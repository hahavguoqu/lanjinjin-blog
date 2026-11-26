---
title: VLM：睁眼看世界，我已觉醒
date: 2026-07-09 00:00:00
categories:
  - 大模型
tags:
  - VLM
  - 多模态
  - ViT
  - CLIP
description: 以一张具体图片的完整前向与推理开销为主线，拆解视觉语言大模型（VLM）的视觉编码、多模态融合、分阶段训练与推理代价。
---

把一张 1024×768 的照片和一句“这张图里有什么？”同时交给大语言模型，问题听起来简单，但底层横着一道硬门槛：语言模型只认离散词表里的 token ID，而图片是一张 $1024 \times 768 \times 3$、共 235 万个连续像素值的三维张量。如果粗暴地把每个像素当成一个 token 塞进 Transformer，自注意力的二次复杂度会直接把显存和算力打穿。

视觉语言大模型（Vision-Language Model, VLM）做的事情，本质上是把这张高维像素矩阵压缩成几十到几千个语义紧凑的视觉向量，再把它们塞进语言模型已经建好的表示空间里，让模型用同一套自回归解码机制把看到的内容“说”出来。

为了不把概念讲散，下文始终围绕一个具体配置展开：一张输入图像进入一个由 **ViT-L/14**（$P=14,\ d_v=1024$）、**两层 MLP Projector** 与 **8B LLM**（32 层、GQA 8 个 KV 头、$d_{\text{llm}}=4096$、BF16）组成的标准三段式 VLM。顺着这张图的张量形状变化，依次看清楚视觉编码、空间对齐、融合路线、分阶段训练以及线上推理时真正花掉的显存与时间。

![一张图片进入标准三段式 VLM 的张量形状变化](/img/posts/vlm/vlm-pipeline.svg)

*图 1：一张 336×336 图片经过 ViT-L/14 与两层 MLP Projector 后变成 576 个视觉 token，与文本 token 拼接进入 LLM。*

---

## 一、把像素压成序列：ViT 与图文对比学习

### 1.1 ViT 为什么能替 LLM 当“眼睛”

ViT（Vision Transformer, 2020）的思路很直接：不逐像素建图，而是把图像切成不重叠的方形图块（patch），把每个图块展平后做一次线性投影，图像就成了一条标准的一维 token 序列。

以 $336 \times 336 \times 3$ 的输入、patch 尺寸 $P = 14$ 为例，图像在宽和高上各切成 $336 / 14 = 24$ 份，得到 $N = 24 \times 24 = 576$ 个 patch。每个 patch 包含 $14 \times 14 \times 3 = 588$ 个数值。把它映射到 $D$ 维向量有两种数学上完全等价的写法：一是把 patch 展平后乘以矩阵 $W_p \in \mathbb{R}^{588 \times D}$；二是直接用一个 `kernel_size=14, stride=14`、输出通道为 $D$ 的二维卷积扫过整张图，输出形状为 $(D, 24, 24)$，再展平成 $(576, D)$。工程上几乎都用后者，因为卷积算子在 GPU 上高度优化。

切完 patch 后，原始 ViT 会在序列头部拼一个可学习的 `[CLS]` token，并加上一维可学习位置编码 $\text{pos\_embed} \in \mathbb{R}^{(N+1) \times D}$，再送入标准的 Pre-LN Transformer Encoder：

$$
\begin{aligned}
z_0 &= [x_{\text{CLS}}; x_1; x_2; \dots; x_N] + \text{pos\_embed} \\
z'_\ell &= z_{\ell-1} + \text{MSA}(\text{LN}(z_{\ell-1})) \\
z_\ell &= z'_\ell + \text{FFN}(\text{LN}(z'_\ell))
\end{aligned}
$$

原始图像分类任务只取 $z_L[0]$（即 `[CLS]` 对应的全局向量）送进分类头；而在 VLM 里，我们需要保留图像里的局部空间细节，因此通常丢弃 `[CLS]`，直接把全部 patch 特征 $z_L[1:] \in \mathbb{R}^{576 \times 1024}$ 交给后续模块。

与卷积网络相比，ViT 几乎没有“局部连接”和“平移等变”这类视觉归纳偏置，一切空间关系都得靠数据学出来。在 ImageNet-1K（约 128 万张图）这种规模上，从零训练的 ViT 往往打不过同量级 ResNet；但当数据量拉到千万乃至数亿级时，全局自注意力的容量上限反而反超 CNN。这也直接决定了现代 VLM 几乎不会从零训 ViT，而是复用在海量图文对上预训练好的视觉编码器。

### 1.2 从 CLIP 到 SigLIP：为什么视觉编码器要和文本对齐

在 2021 年之前，视觉预训练的主流是给 ImageNet 的固定 1000 个类别打标签。这种闭集分类训练出来的特征很强，但只认有限类别，遇到细粒度文字、复杂场景描述或开放语义就捉襟见肘。

CLIP（Contrastive Language-Image Pre-training, 2021）把监督信号换成了互联网上现成的 4 亿图文对。它用一座图像塔（ViT 或 ResNet）和一座文本塔（12 层因果 Transformer，词表 49,152，最大序列长度 77），把图片和对应描述各自投影到同一个 L2 归一化的向量空间。在一个大小为 $N$（原论文取 32,768）的 batch 里，对角线上的 $N$ 对是正样本，其余 $N^2 - N$ 对是负样本，两侧对称地计算 InfoNCE 损失：

$$
\mathcal{L}_i^{(i \to t)} = -\log \frac{\exp(\text{sim}(v_i, w_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(v_i, w_j) / \tau)}, \qquad
\mathcal{L} = \frac{1}{2N} \sum_{i=1}^N \left(\mathcal{L}_i^{(i \to t)} + \mathcal{L}_i^{(t \to i)}\right)
$$

其中 $\tau$ 是可学习的温度系数（初始化为 0.07），用来控制 softmax 分布的尖锐程度。InfoNCE 强依赖大 batch 提供足够多、足够难的负样本，同时每次反向传播都需要跨所有卡同步 $N \times N$ 相似度矩阵做全局 softmax 归一化，训练基础设施开销很大。

SigLIP（2023）改动了一行损失函数，把多分类 softmax 换成逐对（pairwise）二分类 sigmoid 损失：

$$
\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^N \log \frac{1}{1 + \exp\left(z_{ij}(-t \cdot v_i \cdot w_j + b)\right)}
$$

其中正样本 $z_{ij}=1$（$i=j$），负样本 $z_{ij}=-1$（$i \neq j$），$t$ 和 $b$ 为可学习的缩放与偏置参数。这里常有一个误传，说 SigLIP 把复杂度从 $O(N^2)$ 降到了 $O(N)$——其实两两配对的相似度项依然是 $O(N^2)$，真正的区别在于**不再需要跨设备做全局 softmax 归一化**。去掉全局归一化后，SigLIP 可以用分块流水通信把显存压下来，对超大 batch size 的依赖也明显减弱。随后出现的 DFN（用小模型清洗过滤低质图文对）和 InternViT（把视觉编码器拉到 6B 规模并支持动态分辨率），则进一步把视觉特征的上限往“更干净的数据”和“更高的分辨率”推进。

不过，像 CLIP/SigLIP 这样的双塔模型只适合做特征提取或图文检索，无法直接回答复杂问题：整张图被压成一个全局向量后，物体计数、相对方位和密集 OCR 细节都会大幅丢失。要让模型真正看懂细节并组织语言回答，必须把视觉特征接进 LLM。

---

## 二、怎么把视觉接进 LLM：四条融合路线

把视觉编码器和语言模型拼在一起，核心设计决策只有一个：**视觉信息在什么位置、以什么方式与文本交互？** 过去几年社区演化出四条典型路线。

![四种多模态融合架构路线对比](/img/posts/vlm/vlm-fusion-routes.svg)

*图 2：从末层点积、输入拼接、层间交叉注意力到原生统一多模态的四条技术路线。*

1. **双塔浅融合（CLIP / SigLIP）**：两座编码器独立跑完，只在最后做一次余弦相似度点积。适合检索与零样本分类，不具备生成能力，但为后面两条路线提供了现成的视觉骨干。
2. **Projector 输入层拼接（LLaVA / Qwen-VL / InternVL）**：目前开源界最主流的方案。视觉编码器输出 $H_v \in \mathbb{R}^{N \times d_v}$，经由一个轻量连接器（Projector）映射到语言模型的隐藏维度 $H_v' \in \mathbb{R}^{N \times d_{\text{llm}}}$，再和文本 embedding 直接拼成一条长序列送进 LLM。它最大的工程红利是**完全不改动 LLM 主干**——只要换上更强的基座 LLM 或视觉塔，整套推理引擎（vLLM、SGLang 等）几乎零成本复用。代价同样直接：所有视觉 token 都要挤占 LLM 的上下文窗口和 KV Cache 显存。
3. **层间交叉注意力（Flamingo / Llama 3.2 Vision）**：不把视觉 token 塞进文本序列，而是在 LLM 内部每隔若干层插入 Gated Cross-Attention 层，用文本隐藏状态作 Query，视觉特征作 Key 和 Value：

$$
x' = x + \tanh(\alpha) \cdot \text{CrossAttn}(x, H_v)
$$

其中门控标量 $\alpha$ 初始化为 0，保证训练起步时不破坏原 LLM 的语言能力。这种写法让视觉信息在每一层都能被直接读取，且不挤占文本自注意力的序列长度；缺点是必须侵入修改 LLM 架构，推理框架和算子适配都更重。
4. **原生统一多模态（Chameleon / Gemini 等）**：不再区分“外挂视觉塔 + 文本模型”，而是把图像离散化或连续编码后与文本一起放进统一的 Transformer 中联合预训练，天然支持图文交错输入与生成。但不同模态的梯度尺度和收敛速度差异很大，训练稳定性与数据门槛都极高。

在最常见的前置拼接路线中，Projector 本身也经历过三代演进：

- **单层线性映射（LLaVA v1）**：$H_v' = H_v W_p + b_p$，其中 $W_p \in \mathbb{R}^{d_v \times d_{\text{llm}}}$。对 $d_v=1024, d_{\text{llm}}=4096$ 来说只有约 4.2M 参数，极简但拟合跨模态非线性映射的能力有限。
- **两层 MLP（LLaVA-1.5 / Qwen2-VL / InternVL 1.5+ 主流）**：

$$
H_v' = \text{GELU}(H_v W_1 + b_1) W_2 + b_2, \qquad W_1 \in \mathbb{R}^{d_v \times d_{\text{llm}}},\ W_2 \in \mathbb{R}^{d_{\text{llm}} \times d_{\text{llm}}}
$$

加一层非线性之后，在 VQA 等基准上就能稳定提升几个点，且几乎不增加推理延迟。
- **Q-Former / Resampler（BLIP-2 / Qwen-VL v1）**：用一组固定数量（如 32 或 64 个）的可学习 query，通过 cross-attention 把几百个视觉 patch 压缩成固定长度。它能大幅压低送进 LLM 的视觉 token 数，但在密集 OCR 和小物体定位任务上容易丢空间细节，因此近两年的主流多模态模型又普遍回到了 **MLP + 局部空间合并（如 2×2 pixel shuffle / token merge）** 的折中方案。

---

## 三、为什么不能一步训到底：三阶段训练

既然 ViT 和 LLM 都是现成预训练好的，为什么不直接拿图文数据全参数端到端微调？直接训会立刻撞上两个工程问题：刚随机初始化的 Projector 梯度噪声极大，反向传播一冲进 LLM 和 ViT，很容易把原本学好的语言推理和视觉表征一起冲坏；另一方面，高质量多模态指令数据远比纯文本语料稀缺，直接全参训极易过拟合和灾难性遗忘。

因此，从 LLaVA 到 Qwen2.5-VL，工业界普遍把训练拆成“先接好接口，再联合放开，最后校准偏好”的递进流程。

![VLM 三阶段训练的模块冻结与解冻策略](/img/posts/vlm/vlm-training-stages.svg)

*图 3：不同训练阶段中 Vision Encoder、Projector 与 LLM 主干的冻结/解冻状态。*

**第一阶段：特征对齐（Alignment）**。冻结 Vision Encoder 和 LLM，只放开 Projector（学习率通常给到 $1\times 10^{-3}$ 量级），用大规模图文描述对（如 LLaVA-1.5 的 558K，或现代模型千万级图文对）做标准的下一 token 预测，且只对回答部分计算交叉熵损失：

$$
\mathcal{L} = -\sum_{t \in \text{Answer}} \log P(a_t \mid I, q, a_{<t})
$$

这一步并不指望模型学会复杂推理，只是给随机初始化的 Projector 做好“热身”，让视觉特征先落到 LLM 词向量空间的正确邻域里。

**第二阶段：多模态指令微调（Multimodal SFT）**。放开 Projector 和 LLM（学习率压低到 $2\times 10^{-5}$ 左右），把视觉问答、OCR、图表理解、多轮对话和推理数据混在一起训练。LLaVA-1.5 在这一步只用了约 665K 混合指令数据，依然冻结 ViT；而像 Qwen2.5-VL、InternVL2.5 这类强调高分辨率 OCR 与文档解析的现代模型，会在大规模多模态继续预训练与 SFT 阶段连同视觉编码器一起低学习率解冻，让 ViT 真正适应细粒度文字与非正方形分辨率。

**第三阶段：偏好对齐（DPO / RLHF）**。SFT 之后的模型经常出现“看图编造”（幻觉）或回答过度冗长的问题。通过构造同一张图下的正负回答对 $(y_w, y_l)$，用 DPO 直接拉大好回答相对参考模型的对数概率优势：

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta \left(\log\frac{\pi_\theta(y_w \mid I, x)}{\pi_{\text{ref}}(y_w \mid I, x)} - \log\frac{\pi_\theta(y_l \mid I, x)}{\pi_{\text{ref}}(y_l \mid I, x)}\right)\right)
$$

相比需要额外训练 Reward Model 的 PPO，DPO 显存开销更小、超参更稳，已经是目前多模态后训练的标配环节。

---

## 四、把账算清楚：推理阶段的分辨率与显存代价

回到开头的配置：**ViT-L/14 + 两层 MLP + 8B LLM（32 层，GQA 8 个 KV 头，$D=128$，BF16）**。当一张图片和 10 个字的提问发给线上服务时，真正决定延迟和吞吐的往往不是 ViT，而是视觉 token 数量对 LLM Prefill 和 KV Cache 的放大效应。

先看固定分辨率 $336 \times 336$ 的情况：

1. ViT 输出 576 个视觉 token，拼接 10 个文本 token，送入 LLM 的总输入长度 $S = 586$。
2. LLM 先做一次 **Prefill**：586 个 token 并行通过 32 层 Transformer，计算完整的注意力并写入 KV Cache。每层每个 token 需要存下 K 和 V 两个张量，对于 GQA（$H_{kv}=8, D=128$）的 8B 模型，单个 token 在全部 32 层占用的 KV Cache 为：

$$
2 \times 32 \times 8 \times 128 \times 2\ \text{bytes} = 131\,072\ \text{bytes} = 128\ \text{KiB}
$$

所以 586 个输入 token 只占约 **73 MiB** KV Cache，即使上下文拉到 $S = 4096$，也只需要 **512 MiB**（0.5 GiB）。如果误按标准 MHA（$d_{\text{model}} = 4096$）去估算，会得出 2 GiB，足足差出 4 倍——这正是现代开源 VLM 基座全面转向 GQA 的直接原因。
3. 随后进入 **Decode**：逐 token 自回归生成回答。Prefill 是矩阵-矩阵乘（算力受限，Compute-bound），Decode 每步只算 1 个新 token 却要搬运全模型权重和历史 KV（显存带宽受限，Memory-bound）。

但一旦换成高分辨率文档、网页截图或 4K 照片，固定缩放到 $336 \times 336$ 会把小字直接糊成马赛克。因此现代 VLM（LLaVA-NeXT、InternVL 1.5+、Qwen2-VL / Qwen2.5-VL）普遍转向**动态分辨率**：保持原始宽高比，按规则把大图切成若干局部块，再附带一张全局缩略图（或者像 Qwen2.5-VL 那样用原生动态分辨率 + 2×2 空间池化压缩 token）。

视觉 token 数量膨胀带来的推理账非常直观（以单块 576 token、未做额外空间压缩估算）：

| 输入分辨率 | 切块数（含局部块） | 视觉 token 数 | Prefill 注意力相对开销 | 8B GQA 仅图像 KV 占用 |
| :--- | ---: | ---: | ---: | ---: |
| 336 × 336 | 1 | 576 | 1× | 72 MiB |
| 672 × 672 | 4 | 2,304 | ~16×（二次项） | 288 MiB |
| 1024 × 768 | 6 | 3,456 | ~36×（二次项） | 432 MiB |
| 1920 × 1080 | 20 | 11,520 | 极高，需强制限流 | 1.41 GiB |

这也是为什么所有高分辨率 VLM 都必须配两道保险：一是**视觉 token 压缩**（例如 2×2 相邻 patch 拼接成一个 token，直接把 $N$ 砍掉 75%，Prefill 的注意力计算量降为原来的 $1/16$）；二是**设置单图最大像素/最大切块上限**，防止一张 4K 图瞬间把整张卡的显存池和 TTFT 打崩。在多轮看图对话场景下，推理引擎还会对视觉前缀启用 **Prefix Caching**，避免用户针对同一张图连续追问时反复重跑 ViT 和 Prefill。

---

## 五、主流开源演进脉络速览

把前面拆完的模块放回近两年的代表模型里，就能看清各家真正卷的差异点在哪里：

| 模型系列 | 关键演进节点 | 视觉编码与融合特点 | 核心取舍 |
| :--- | :--- | :--- | :--- |
| **LLaVA 家族** | v1 (2023.04) → 1.5 (2023.10) → NeXT (2024.01) → OneVision (2024.08) | CLIP-ViT/SigLIP + MLP Projector + 输入拼接；NeXT 引入 AnyRes 动态切块，OneVision 统一单图/多图/视频 | 验证了极简三段式 + 高质量指令数据的性价比上限 |
| **InternVL 系列** | v1 (2023.12) → 1.5 (2024.04) → 2/2.5 (2024.09–2025.01) → 3 (2025.04) | 自研大视觉塔 InternViT-300M/6B + Pixel Shuffle 压缩 + MLP Projector；InternVL3 沿用三段式架构但改为原生多模态联合预训练 | 用大视觉编码器 + 动态切块硬拉 OCR 与细粒度理解 |
| **Qwen-VL 系列** | v1 (2023.08) → Qwen2-VL (2024.08) → Qwen2.5-VL (2025.01) → Qwen3-VL (2025) | 从早期的 Q-Former 全面转向原生动态分辨率 ViT（M-RoPE + 2×2 视觉 token 合并）+ MLP 拼接 | 不强制固定尺寸切块，原生适配任意长宽比、长视频与文档 |
| **Llama 3.2 Vision** | 11B / 90B (2024.09) | 自研 ViT + **Cross-Attention** 层间注入，冻结原有 Llama 3.1 文本权重 | 保证纯文本能力零退化，但推理框架需专门适配交叉注意力 |
| **DeepSeek-VL / VL2** | VL (2024.03) → VL2 (2024.12) | 早期双路低高分辨率编码，VL2 转向动态切块 + MoE 语言基座 | 用稀疏激活的 MoE 压低多模态推理单步计算成本 |

拉远了看，VLM 这几年的收敛方向其实很明确：**架构上越来越简洁**（三段式拼接 + MLP + 视觉 token 压缩几乎成了默认底座），真正拉开代差的，已经从“怎么把两个模型接在一起”，转移到了**高分辨率/视频下的视觉 token 效率、多模态数据清洗质量，以及后训练阶段对幻觉和推理链的校准**。

---

## 参考论文

1. Vaswani et al., *Attention Is All You Need* (2017)
2. Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT, 2020)
3. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP, 2021)
4. Zhai et al., *Sigmoid Loss for Language Image Pre-Training* (SigLIP, 2023)
5. Liu et al., *Visual Instruction Tuning* (LLaVA, 2023) & *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5, 2023)
6. Bai et al., *Qwen2.5-VL Technical Report* (2025)
7. Chen et al., *Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling* (InternVL2.5, 2024)
