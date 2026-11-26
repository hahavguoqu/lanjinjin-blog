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
description: 视觉语言大模型（VLM）从基础模块、标准架构、多模态融合、训练与推理阶段到主流模型的系统梳理。
---
**前言**：好久没更新了......这篇文章作为一个初级的总结笔记，部分内容是ai生成的，不过都经过人类校验，可以放心阅读。

## 第一章：引入

视觉语言大模型（Vision-Language Model, VLM）要解决的问题即为：**睁眼看世界，祖国放心，我已觉醒**

### VLM 要解决的核心矛盾

VLM 面临的根本矛盾是**模态差异**：

- **语言模型理解的输入**：离散的 token ID 序列（来自有限词表）
- **图片的原始形式**：连续的像素矩阵（$H \times W \times 3$ 的数值矩阵）

解决该矛盾路径是：

像素矩阵 ($H \times W \times 3$) → 图像预处理（归一化）→ **视觉编码器**（例如 CLIP-ViT、SigLIP-ViT、EVA-CLIP、InternViT 等）输出视觉特征序列 ($N \times d_v$) → **Projector**（连接器）映射为 LLM 可理解的**视觉 embedding** ($N \times d_{\text{llm}}$) → 与文本 embedding **拼接**形成多模态 token 序列 → LLM 自回归解码输出文本。

### 本文的组织结构

本文按照"从基础到前沿，从理论到实践"的原则组织：

- **第一章**：引入（你在这里）
- **第二章**：基础模块 —— Token、Transformer、ViT、CLIP
- **第三章**：VLM 标准架构 —— 三模块范式
- **第四章**：多模态融合方式 —— 不同技术路线的对比
- **第五章**：训练阶段 —— 从对齐到微调到对齐
- **第六章**：推理阶段 —— 从 prefill 到 decode
- **第七章**：具体模型深度解析 —— 解剖主流模型
- **附录**：相关论文、常见名词解释

建议按顺序阅读。如果你对某个部分已经熟悉，可以直接跳转。

---



## 第二章：基础模块

### 视觉编码器：ViT 及其现代变体

视觉编码器是 VLM 的"眼睛"。它把图片转换成 Transformer 可以处理的 token 序列。

#### 为什么需要专门的视觉编码？

直观地看，给 LLM 直接输入像素值是不现实的——一张 $1024 \times 1024$ 的 RGB 图片，直接展平后有 3,145,728 个值。如果每个像素当做一个 token（3M 个 token），计算 self-attention 的复杂度是 $O(n^2)$，需要的代价非常高。

所以我们需要一种方法，**把图片压缩成语义丰富的 token 序列**。这就是视觉编码器的作用。

#### ViT 的设计思想

ViT（Vision Transformer，2020 年 Google 提出）的核心思想极其简洁：**既然 Transformer 可以处理序列，那把图片变成序列不就行了？** 图片 → 切 patch → flatten → 线性投影 → token 序列 → Transformer。

这和 NLP 中的操作完全对应：

- NLP：文本 → 切词 → embedding 查表 → token 序列 → Transformer
- CV：图片 → 切 patch → 线性投影 → token 序列 → Transformer

#### ViT 的详细结构

**输入处理**：

输入图片 $I \in \mathbb{R}^{H \times W \times C}$，通常 $C=3$（RGB）。选择 patch size $P$：标准 ViT 使用 $P=16$，CLIP ViT-L/14 使用 $P=14$，现代高分辨率模型使用 $P=14$ 或 $P=8$。patch 数量 $N = (H/P) \times (W/P)$，例如 $336 \times 336$ 图片，$P=14$ 时 $N = 24 \times 24 = 576$。

**Patch Embedding**：

每个 patch 是一个 $P \times P \times C$ 的小区域。有两种等价实现：

**方法 1（显式展平 + 线性层）**：对于一个patch，展平成一维向量，然后映射到D维embedding 向量。$x_i = W_p \cdot \text{flatten}(\text{patch}_i)$，其中 $W_p \in \mathbb{R}^{D \times P^2 C}$。

**方法 2（等价于卷积）**：使用 stride=P, kernel=P, 输出通道=D 的 2D 卷积（stride即为每次移动的像素，kernel即为卷积核大小，所以在这个例子中每个卷积核之间不会重叠，且每个卷积核正好覆盖一个patch）。卷积权重 $W_{\text{conv}}$ 的 shape 为 $(D, C, P, P)$，输出 shape 为 $(D, H/P, W/P)$，再展平为 $(N, D)$。

方法 2 在工程上更高效（卷积是高度优化的算子），两者数学等价。

**加入特殊 token 和位置编码**：

ViT 借鉴 BERT，在序列开头加入 $[CLS]$ token：

$$
z_0 = [x_{CLS}; x_1; x_2; \dots; x_N] + \text{pos\_embed}
$$

$x_{CLS} \in \mathbb{R}^D$ 是可学习的特殊 token，其最终表示用于分类（ViT 原始用法）。$\text{pos\_embed} \in \mathbb{R}^{(N+1) \times D}$ 是位置编码，ViT 使用可学习的一维位置编码。注意 ViT 使用 1D 而非 2D 位置编码，实验表明 2D 位置编码对性能提升不明显。

**Transformer Encoder**：

与标准 Transformer 完全相同。对于 $L$ 层：

$$
\begin{aligned}
z'_\ell &= z_{\ell-1} + \text{MSA}(\text{LN}(z_{\ell-1})) \\
z_\ell &= z'_\ell + \text{FFN}(\text{LN}(z'_\ell))
\end{aligned}
$$

最终输出 $z_L \in \mathbb{R}^{(N+1) \times D}$，其中 $z_L[0]$ 是 $[CLS]$ token 的最终表示（全局图像表示），$z_L[1:]$ 是各 patch 的最终表示（局部表示）。

**ViT 的输出在 VLM 中的使用**：

在 VLM 中，我们通常不使用 $[CLS]$ token，而是使用所有 patch token：$z_L[1:] \in \mathbb{R}^{N \times D}$。这就是 "visual features"，然后通过 Projector 映射到 LLM 的 embedding space。

#### 为什么 ViT 需要大量数据？

这是理解 ViT 最关键的 insight 之一：

- **CNN 有归纳偏置（Inductive Bias）**：局部连接（卷积核只看局部区域）、平移等变性、权值共享。这些先验知识让 CNN 在数据有限时也能学得不错。
- **ViT 没有归纳偏置**：全部是全局自注意力，没有任何 CV 相关的先验，一切都需要从数据中学习。

所以 ViT 需要大量数据才能收敛：ViT 论文发现，在 ImageNet-1M 上训练不如 ResNet，但在 JFT-300M 或 ImageNet-21K 上预训练后远超 ResNet。**ViT 是"数据饥饿"的，给的数据越多，效果越好**。这也是 CLIP 使用 4 亿图文对训练 ViT 的原因之一。

#### 现代 ViT 变体

**第一代：原始 ViT（2020）**

| 模型 | 参数 | 层数 | $d$ |
|------|------|------|------|
| ViT-B/16 | 86M | 12 | 768 |
| ViT-L/16 | 307M | 24 | 1024 |
| ViT-H/14 | 632M | 32 | 1280 |

训练数据：ImageNet-21K / JFT-300M。

**第二代：DeiT（Data-efficient Image Transformers, 2021）**

通过知识蒸馏（teacher CNN → student ViT）减少数据需求。DeiT-B 在 ImageNet-1K（仅 1.3M 图片）上即可训练。

**第三代：CLIP-ViT（2021）**

使用对比学习在 4 亿图文对上训练，成为 VLM 视觉编码器的标准选择。CLIP ViT-L/14：307M 参数，224px 输入。不过在后文会详细讲解CLIP和SigLIP。

**第四代：SigLIP（Sigmoid Loss for Language Image Pre-training, 2023）**

Google 提出，用 sigmoid loss 替代 softmax（InfoNCE）loss。对于图文对 $(x_i, y_i)$：

$$
\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N} \sum_i \log\sigma(s_i \cdot t_i - b) - \frac{1}{N} \sum_i \sum_{j \neq i} \log\sigma(-s_i \cdot t_j + b)
$$

其中 $s_i = \text{ImageEncoder}(x_i)$，$t_i = \text{TextEncoder}(y_i)$，$\sigma$ 是 sigmoid 函数，$b$ 是可学习的偏置参数。

关键区别：CLIP 的 InfoNCE 需要对 batch 内 $N \times N$ 相似度矩阵做 softmax 归一化；SigLIP 将其改为 pairwise sigmoid loss，不需要全局 softmax 归一化，因此对 batch size 的依赖更弱，也更适合分布式/小 batch 训练。复杂度不能简单概括为 O(N) vs O(N²)，更准确的区别是“是否需要全局归一化”。优势包括：不需要大 batch size（CLIP 需要 32K batch）、训练更稳定、效果持平或更好、可以轻松扩展到更大的数据。SigLIP 已经成为现代 VLM 的事实标准视觉编码器。

**第五代：DFN（Data Filtering Networks, 2024）**

Apple 提出，核心贡献不是模型架构，而是数据清洗方法。使用一个小型 CLIP 模型作为"过滤器"：对每个图文对计算相关性分数，过滤掉低分的数据，用高质量数据训练最终模型。DFN-5B 使用 5B 过滤后的数据训练，在多个基准上超越 CLIP ViT-L/14 和 SigLIP。关键洞察是：**对于视觉编码器，数据质量 > 数据量**。

**第六代：InternViT（2024-2025）**

InternVL 团队开发的视觉编码器，是目前最大的开源视觉编码器：InternViT-300M 与 ViT-L 参数量级相当，InternViT-6B 有 60 亿参数。特点包括动态分辨率支持、结合了 SigLIP 的训练策略、超大参数带来的强视觉理解能力。

### CLIP 与后续发展

#### CLIP 要解决的问题

在 CLIP 出现之前（2021 年以前），计算机视觉模型的训练范式是：收集标注数据（如 ImageNet 1000 类）、定义固定的类别集合、训练模型做分类。问题是：标注成本高、类别集合固定无法处理"开放类别"、模型学到的表示是"封闭的"。

CLIP 提出了一种全新的范式：**不用人工标注的类别标签，而是用互联网上的图文对**。一张图片 + 一段描述文本 = 一个训练样本。学习目标是让图片和对应的文本在向量空间中靠近，让图片和不对应的文本在向量空间中远离。

#### CLIP 的架构

CLIP 是一个**双塔模型**（Two-Tower Model）：

图片 $I$ → Image Encoder → $v \in \mathbb{R}^d$；文本 $T$ → Text Encoder → $w \in \mathbb{R}^d$。

CLIP 原版的 Image Encoder 有两种选择：ViT（推荐：ViT-B/32、ViT-B/16、ViT-L/14）或 ResNet（RN50、RN50x4 等）。Text Encoder 是 Transformer（GPT 风格，因果掩码），12 层、512 维、8 头，词表 49,152（BPE tokenizer），最大序列长度 76。

**最后的投影层**（很关键）：Image Encoder 和 Text Encoder 的输出各自通过一个投影层映射到共享的多模态空间 $d=512$：$v = \text{proj}_i(E_{\text{img}}(I))$，$w = \text{proj}_t(E_{\text{text}}(T))$。实际使用中 $v$ 和 $w$ 会做 L2 归一化：$v = v / \|v\|$, $w = w / \|w\|$。

#### CLIP 的训练：对比学习

CLIP 使用**对比学习**（Contrastive Learning）训练。一个 batch 有 $N$ 个图文对 $(I_i, T_i)$，共 $N \times N$ 个可能组合，其中 $N$ 个是正样本对（匹配），$N^2-N$ 个是负样本对（不匹配）。CLIP 的 batch size $N = 32,768$。

**训练目标：InfoNCE Loss**

对于第 $i$ 个样本：

$$
\mathcal{L}_i^{(i \to t)} = -\log \frac{\exp(\text{sim}(v_i, w_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(v_i, w_j) / \tau)}
$$

这是 Image→Text 方向（图片找对应的文本）。对称地，Text→Image 方向：

$$
\mathcal{L}_i^{(t \to i)} = -\log \frac{\exp(\text{sim}(w_i, v_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(w_i, v_j) / \tau)}
$$

总损失（所有样本两个方向平均）：

$$
\mathcal{L} = \frac{1}{2N} \sum_{i=1}^N (\mathcal{L}_i^{(i \to t)} + \mathcal{L}_i^{(t \to i)})
$$

其中 $\text{sim}(v,w) = v \cdot w / (\|v\| \cdot \|w\|)$ 是余弦相似度，$\tau$ 是 temperature 参数（可学习），控制分布的"锐利度"。

**理解 Temperature $\tau$**：$\tau$ 控制 softmax 分布的集中程度。$\tau$ 很小（如 0.01）时，相似度差异被放大，softmax 输出接近 one-hot，模型对负样本"过于自信"；$\tau$ 很大（如 0.1）时，相似度差异被缩小，softmax 输出更平滑，模型学习更"温和"。CLIP 中 $\tau$ 是可学习的参数，初始化为 0.07。

**为什么需要大 batch size？** InfoNCE 的效果依赖于负样本的数量和质量。更多的负样本 = 更丰富的对比信号。如果 batch 太小，负样本太少，学习信号弱。CLIP 使用 32,768 的 batch size 经过实验验证。这也是 CLIP 训练需要大量 GPU 的原因——OpenAI 用 592 台 V100（共 12.8 天）训练 CLIP。

#### CLIP 的零样本分类能力

CLIP 训练完成后，可以不需要任何训练数据就进行分类。方法是将所有候选类别放入 prompt 模板（如 "a photo of a {class}"），用 Text Encoder 编码所有类别文本，用 Image Encoder 编码待分类图片，取与图片最相似的类别：

$$
\text{pred}=\arg\max_{c}\ \mathrm{sim}\left(
f_{\text{img}}(I),
f_{\text{text}}(p_c)
\right)
$$

使用多个 prompt 模板集成可以提高效果。

#### CLIP 的问题与局限

那其实我们很容易发现CLIP的问题，例如：

1. **全局表示，缺乏细粒度理解**：CLIP 把整张图片编码为一个向量，对于"图片中左下方的物体是什么颜色"这种问题不擅长
2. **物体计数和空间关系弱**：可能识别出猫和桌子，但无法精确计数和定位
3. **对分布外数据泛化有限**
4. **分辨率受限**：CLIP ViT 的输入分辨率通常是 224px，对于需要精细细节的任务（如 OCR）效果差
5. **缺乏生成能力**：CLIP 只能做表示学习（embedding），不能生成文本

#### 后 CLIP 时代：SigLIP 的崛起

到 2025-2026 年，CLIP 作为 VLM 视觉编码器的主力地位正在被 SigLIP 和 DFN 取代。

SigLIP 的四个关键优势：训练效率更高（$O(N)$ vs $O(N^2)$）、不需要大 batch size、可以持续训练而不饱和、与 VLM 兼容性好。

**SigLIP-2（2025）**进一步改进：引入正则化、更好的分辨率扩展、局部-全局对齐、多尺度特征，在多个基准上超越原始 SigLIP 和 CLIP。

#### 从表示学习到生成：CLIP 在 VLM 中的角色

在 VLM 中，CLIP/SigLIP 的定位是**特征提取器**——为 LLM 提供高质量的视觉表示。使用方式有三种：

- **固定/冻结的方式**（大多数情况）：加载预训练权重，在 VLM 训练期间冻结，只训练 Projector 和 LLM
- **全微调的方式**（少数情况，如 InternVL）：视觉编码器随训练一起更新，效果更好但计算成本更高
- **LoRA 微调**（折中方案）：对视觉编码器应用 LoRA，参数量小，训练效率高

---



## 第三章：VLM 标准架构

掌握了基础模块之后，我们来看 VLM 的标准架构。

### 三模块范式

几乎所有现代 VLM 都遵循一个三模块架构：

- **Vision Encoder**：把图片变成视觉特征。输入 $I \in \mathbb{R}^{H \times W \times C}$，输出 $H_v \in \mathbb{R}^{N \times d_v}$
- **Projector / Connector**：把视觉特征映射到语言空间。输入 $H_v \in \mathbb{R}^{N \times d_v}$，输出 $H_v' \in \mathbb{R}^{N \times d_{\text{llm}}}$
- **LLM**：融合图文信息并生成回答。输入 $[H_v'; T_{\text{emb}}] \in \mathbb{R}^{(N+M) \times d_{\text{llm}}}$，输出文本

**这个设计的美妙之处**在于模块化：每个模块可以独立替换/升级，可以复用现成模型（Vision Encoder 来自 CLIP，LLM 来自 Llama/Qwen），灵活性高。

### Projector（连接器）的详细设计

为什么需要 Projector？因为视觉编码器的输出空间和 LLM 的输入空间不同，维度不同（$d_v$ vs $d_{\text{llm}}$），语义也不同（视觉语义空间 vs 语言语义空间）。

#### 三种 Projector 设计

**1. Linear Projector（最简洁）**

$$H_v' = H_v \cdot W_p + b_p, \quad W_p \in \mathbb{R}^{d_v \times d_{\text{llm}}}, b_p \in \mathbb{R}^{d_{\text{llm}}}$$

参数量：$d_v \times d_{\text{llm}} + d_{\text{llm}}$。例如 $d_v=1152, d_{\text{llm}}=4096$ 时约 4.7M 参数。优点：简单、不易过拟合。缺点：表达能力有限。

**2. MLP Projector（LLaVA-1.5 标准）**

$$H_v' = W_2 \cdot \text{GELU}(W_1 H_v + b_1) + b_2$$

其中 $W_1 \in \mathbb{R}^{d_v \times d_{\text{llm}}}$，$W_2 \in \mathbb{R}^{d_{\text{llm}} \times d_{\text{llm}}}$。两层 MLP 比单层 Linear 效果好约 3%（在 VQA 基准上），是当前主流选择。

**3. Q-Former Style（BLIP-2 设计）**

使用一组可学习的 query token 从视觉特征中"提取"信息。输入 learnable queries $Q \in \mathbb{R}^{N_q \times d_q}$ 和 visual features $H_v \in \mathbb{R}^{N \times d_v}$。Q-Former 内部通过 self-attention 和 cross-attention 压缩信息，输出 $R \in \mathbb{R}^{N_q \times d_q}$，其中 $N_q \ll N$（典型 $N_q=32$，而 $N$ 可达 256+）。

三种方案的对比：Linear 参数量最小、表达能力弱；MLP 参数量适中、表达能力强，是当前主流；Q-Former 参数量最大、表达能力最强，但训练也更复杂。当前趋势是大多数现代 VLM（Qwen-VL、InternVL）使用 MLP Projector。

### 信息流分析：VLM 的完整前向过程

以图片 $I$（$1024 \times 768$）+ 文本 $Q$（"这张图片里有什么？"）为例：

1. **图像预处理**：固定分辨率模型会 resize 到 $336 \times 336$；动态分辨率模型保持宽高比后切块
2. **Vision Encoder 前向**：$336/14 = 24$，$24 \times 24 = 576$ 个 patch，经过 24 层 Transformer，输出 $576 \times 1024$
3. **Projector 映射**：$576 \times 1024 \to 576 \times 4096$
4. **与文本 token 拼接**：系统 prompt + 用户 query → 文本 embedding → 拼接视觉和文本 token，总长 586
5. **LLM Prefill**：586 个 token 一次性进入 LLM，计算所有 attention
6. **自回归 Decode**：逐 token 生成，每一步利用 KV cache

### LLaVA 家族演进

LLaVA 系列是开源 VLM 中最有影响力的家族之一：

**LLaVA v1（2023 年 4 月）**：CLIP-ViT-L/14 + Linear Projector + Vicuna-13B。贡献在于将多模态指令微调引入 VLM，使用 GPT-4 生成 158K 多模态指令数据。局限是 Linear Projector 表达能力有限、224px 低分辨率。

**LLaVA-1.5（2023 年 10 月）**：MLP Projector、AnyRes 高分辨率（AC）、数据扩展到 665K、使用 Vicuna v1.5。

**LLaVA-NeXT（2024 年 5 月）**：支持更大的 LLM 基座（LLaMA 3 8B、Qwen 1.5 72B），改进的高分辨率支持，更强的数据混合。

**LLaVA-OneVision（2025 年 1 月）**：跨图像、多图像和视频理解的统一框架，同一个模型可以处理单图问答、多图对比、视频理解。

数据是第一推动力，每一代都在扩展数据，架构趋于稳定，分辨率持续提升，能力边界不断扩展（单图→多图→视频）。

---



## 第四章：多模态融合方式

"融合"是 VLM 技术中最为核心的设计决策——**视觉信息如何与文本信息交互？**

### 融合的本质是什么？

融合要解决的问题是：给定视觉特征序列 $V = [v_1, ..., v_N]$ 和文本特征序列 $T = [t_1, ..., t_M]$，如何让两者充分交互、产生理解？

按融合深度可分为：浅融合（最后才交互，如 CLIP 点积）、中融合（在输入层拼接，如 LLaVA）、深融合（在各层不断交互，如 cross-attention）。一般来说，**交互越早、越频繁，模型的多模态理解能力越强，但计算成本和工程复杂度也越高**。

### 路线 A：双塔检索模型（浅融合）

代表：CLIP、SigLIP。图片 → Image Encoder → $v \in \mathbb{R}^d$；文本 → Text Encoder → $w \in \mathbb{R}^d$。融合：$\text{sim} = v \cdot w / (\|v\| \cdot \|w\|)$ — 纯粹的点积。

交互阶段仅在最后一步，交互方式为简单的点积，交互次数仅 1 次。没有跨模态的 attention，没有信息融合，只有"匹配度"。适用于图文检索、零样本分类、作为 VLM 的视觉编码器。

双塔模型不能用于生成，因为没有"融合"——两个模态的表示是独立计算的。LLM 需要的不是"图片和文本有多相似"，而是"图片中的具体物体、空间关系、属性等细节"。

### 路线 B：Projector 拼接式（中融合）——当前主流

代表：LLaVA、Qwen-VL、InternVL。Vision Encoder → Projector → 视觉 token，与文本 token embedding 拼接后送入 LLM。

融合发生在 LLM 的每一层（通过 self-attention），但视觉信息只在最开始进入一次。一旦进入 LLM，视觉和文本 token 通过同一个 self-attention 充分交互。交互次数 = $L$ 层 $\times$ 每层一次 self-attention。

**为什么这个方案是当前主流？** 极简（不需要修改 LLM 结构）、复用（可以直接使用任何预训练的 LLM）、有效（简单拼接 + self-attention 就足以实现强大的多模态推理）、易扩展（换更好的 LLM 或视觉编码器即可提升性能）。

**局限**：视觉 token 数量影响效率（高分辨率下 $N$ 可能很大），视觉信息只在输入层注入一次（后续信号可能逐层衰减），序列总长度受限于 LLM 的最大上下文长度。

### 路线 C：Cross-Attention 式（深融合）

代表：Flamingo（DeepMind, 2022）、BLIP-2（Salesforce, 2023）。

**Flamingo 的核心设计**：
1. **Perceiver Resampler**：将可变长度的视觉特征压缩为固定数量（64 个）的视觉 token
2. **Gated Cross-Attention**：在冻结的 LLM 的每一层插入 cross-attention：

$$x' = x + \tanh(\alpha) \cdot \text{CrossAttn}(x, \text{vision\_tokens})$$

其中 $\alpha$ 初始化为 0（使得新增层在训练初期不干扰 LLM）。

**Cross-Attention 的数学形式**：

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

在 Flamingo 中，$Q$ = 文本 hidden state（来自 LLM 某一层），$K, V$ = 视觉特征（经过 Perceiver Resampler）。

Cross-Attention 方案的优势：视觉信息在每一层都注入、压缩机制减少 LLM 计算负担、可以冻结 LLM 只训练 cross-attention 层。局限：需要修改 LLM 结构、训练和工程复杂度更高。

### 路线 D：原生统一多模态模型（最深度融合）

代表：Chameleon（Meta, 2024）、Gemini（Google, 2023-2026）、GPT（OpenAI, 2024-2026）。

核心思想：不把"视觉"和"语言"当作两个分开的模块，而是从一开始就设计一个可以处理多模态 token 的统一模型。文本 token 和图像 token 在同一个词表中，使用同一个 Transformer 处理。

挑战：训练数据需求巨大、训练不稳定（不同模态的训练动态不同）、架构设计复杂（视觉 tokenization、序列长度管理）、细节不公开（这些模型都是闭源的）。

从信息论角度：拼接式（LLaVA）中视觉信息是 $N$ 个独立 token，靠 self-attention 融合；Cross-attention（Flamingo）视觉信息在每层交叉注意力注入；原生多模态中视觉信息是模型内生的表示，不存在"两个空间的映射"问题。理论上原生多模态可以做到最充分的信息融合，但代价是训练成本和工程难度极大。

---



## 第五章：训练阶段

### VLM 训练的整体图景

现代 VLM 的训练通常分为多阶段：

1. **阶段一：视觉-语言对齐**（Pretraining / Alignment）—— 训练 Projector，冻结 Vision Encoder 和 LLM，使用图文对数据
2. **阶段二：多模态指令微调**（SFT）—— 训练 Projector + LLM（或全参数），使用多模态指令问答数据
3. **阶段三（可选）：偏好对齐**（RLHF / DPO）—— 训练 LLM（部分或全部），使用人类偏好数据

**为什么需要多阶段训练？** 直接端到端训练所有参数的问题包括：训练不稳定（视觉编码器和 LLM 的参数空间差异大）、过拟合（指令数据量有限）、灾难性遗忘（同时训练可能导致 LLM 丢失语言能力）。分阶段训练的好处：逐步对齐、冻结的模块保持能力、数据效率高。

### 阶段一：视觉-语言对齐

可训练参数仅 Projector（通常几百万到几千万），冻结 Vision Encoder 和 LLM。训练数据是图文对（典型：CC-3M、CC-595K）。LLaVA-1.5 使用 558K 图文对。

**训练数据格式**：固定模板 "User: 请详细描述这张图片。 Assistant: {caption}"。为什么使用固定指令？这是一种"教学模板"——让 Projector 学习将视觉特征对应到文本描述。模型在这个阶段还没有学会理解多样化的指令。

**损失函数**：自回归语言建模（标准 cross-entropy loss），只对回答（assistant）部分的 token 计算 loss：

$$
\mathcal{L} = -\sum_{t=1}^T \log P(a_t \mid I, q, a_{<t})
$$

通常训练很短（几 k 到几十 k 步），因为参数量很小且图文对数据的质量有限。

### 阶段二：多模态指令微调

这是 VLM 训练的核心阶段。可训练参数包括 Projector + LLM（Vision Encoder 大多数模型仍然冻结）。训练数据是多模态指令数据：图文问答对、多轮对话、OCR + 理解、区域理解、推理数据。

**LLaVA-1.5 的数据混合配方（665K 条）**：

| 数据来源 | 规模 | 内容 |
|---------|------|------|
| LLaVA-Instruct-158K | 158K | GPT-4 生成的多模态指令数据 |
| VQA-v2 | 80K | 视觉问答 |
| GQA | 72K | 基于场景图的推理 |
| OCR-VQA | 80K | 图片中的文字理解 |
| RefCOCO + Visual Genome | 区域理解 | 指代表达、区域描述 |

损失函数与阶段一相同：$\mathcal{L} = -\sum_{t \in \text{Answer}} \log P(\text{token}_t \mid \text{Image}, \text{Query}, \text{token}_{<t})$。

为什么阶段二不冻结 LLM？阶段一的训练后，Projector 已经能将视觉特征映射到语言空间，但模型只学会了"看图说话"。要让它能回答多样化的问题（VQA、推理、OCR 等），需要 LLM 也参与训练。微调 LLM 时学习率比预训练小得多（$2\times 10^{-5}$ vs $3\times 10^{-4}$）。

### 阶段三：偏好对齐（RLHF/DPO）

阶段二训练出来的模型可能存在冗长、幻觉、过度自信等问题。偏好对齐旨在让模型输出更加符合人类期望。

**RLHF（Reinforcement Learning from Human Feedback）**：三阶段流程——SFT → 训练 Reward Model → PPO 优化。优势是可以显著改善输出质量、减少幻觉。局限是需要训练 Reward Model（额外开销）、PPO 训练不稳定。

**DPO（Direct Preference Optimization）**：不需要训练 Reward Model，直接从偏好对中优化。偏好数据 $(y_w, y_l)$（$y_w$ 是更好的回答）：

$$
\mathcal{L}_{\text{DPO}} = -\log\sigma\left(\beta \cdot \left(\log\frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)
$$

其中 $\pi$ 是当前模型，$\pi_{\text{ref}}$ 是参考模型，$\beta$ 控制偏离参考模型的程度。DPO 训练稳定、超参数少，效果与 RLHF 相当或更好。

趋势：偏好对齐正在成为 VLM 训练的标准环节。

### 现代 VLM 训练的完整配方

以 Qwen2.5-VL 为例，其训练流程代表了 2025 年 VLM 训练的主流实践：

1. **Visual Encoder 预训练**：大规模图文对（数十亿），使用 SigLIP 式对比学习
2. **Visual-Language 对齐**：冻结视觉编码器，训练 Projector + Qwen LLM
3. **多模态预训练**：全参数训练，数据包括图文对 + 交错图文 + 视频 + PDF
4. **多模态指令微调**：高质量多模态指令数据
5. **偏好对齐**（可选）：使用 DPO 或 RLHF

训练数据规模估算：对齐阶段约 10M-100M 图文对，多模态预训练约 1B+ 样本，指令微调约 1M-10M 样本。

---



## 第六章：推理阶段

###  推理的总体流程

VLM 推理分为两个子阶段：

- **Prefill**：一次性处理所有输入 token（包括视觉和文本），建立 KV cache。计算密集型，复杂度 $O(S^2 \cdot d)$
- **Decode**：逐 token 生成回答，利用 KV cache 避免重复计算。访存密集型，每步复杂度 $O(S \cdot d)$

### Prefill 阶段

输入序列长度 $S = N_{\text{visual}} + N_{\text{text}}$。例如一张 $336 \times 336$ 图片 + 10 个文本 token，$S = 576 + 10 = 586$。

Prefill 过程中，所有 token 通过 Embedding 层后进入 Transformer。每层计算所有 token 的 Q, K, V，计算完整的 $S \times S$ 注意力矩阵。

**Prefill 的计算量分析**：Self-Attention 部分（每层）的 Q, K, V 投影消耗 $3 \times S \times d_{\text{model}}^2$ 次运算，注意力分数和聚合各消耗 $S^2 \times d_{\text{model}}$。当 $S$ 很大时（如高分辨率 $S=2304$），$S^2$ 项占主导。

**Flash Attention** 是 Prefill 加速的关键技术：使用 tiling（分块）技术，在 SRAM（片上缓存）中完成 attention，避免大的中间矩阵（$S \times S$）写入 HBM，加速 2-4 倍。

### Decode 阶段

Decode 阶段逐 token 生成回答。每一步只计算新 token 的 Q, K, V，追加到 KV cache，然后计算 attention（利用缓存中的旧 K,V）。

**Decode 的计算量分析**：每步（每层）的 Q, K, V 投影消耗 $3 \times d_{\text{model}}^2$，attention 计算消耗 $O(S)$（与所有历史 K 点积）。相比 Prefill 的 $O(S^2)$，Decode 每步只需 $O(S)$ 计算。

瓶颈分析：Prefill 是 compute-bound（计算瓶颈），Decode 是 memory-bound（访存瓶颈）。这是因为 Decode 每步的计算量很小，但需要从 KV cache 中读取 $S \times d$ 的 K,V 矩阵。

### KV Cache 深入分析

KV Cache 是自回归解码的关键优化技术。没有 KV Cache 时，每一步都要重新计算所有 token 的 K,V，复杂度 $O(T^2 \times S)$。有 KV Cache 时，复杂度为 Prefill $O(S^2)$ + Decode $O(T \times S)$。

**KV Cache 的内存占用公式**：

$$
\text{KV Cache} = 2 \times L \times S \times d_{\text{model}} \times \text{bytes\_per\_param}
$$

其中 $L$ 是层数，$S$ 是当前序列长度（随生成增长），$d_{\text{model}}$ 是每个 token 的维度。示例：LLaMA-3-8B（$L=32$, $d_{\text{model}}=4096$, bfloat16 2 bytes）在 $S=4096$ 时 KV cache = $2 \times 32 \times 4096 \times 4096 \times 2 = 2.15$ GB，$S=8192$ 时约 4.29 GB。

**KV Cache 的优化技术**：

1. **GQA（Grouped Query Attention）**：让多个 Q head 共享一组 K,V，KV cache 减少到原来的 $1/8$（8 head 共享）。Llama 3、Qwen 2.5 都使用 GQA
2. **KV Cache 量化**：从 bfloat16 量化为 INT8 或 INT4，减少 50%-75% 的内存占用
3. **PagedAttention / vLLM**：操作系统"分页"思想，消除 KV cache 碎片化，支持内存共享
4. **Prefix Caching**：多个请求共享相同图片的 KV cache

### 推理优化全景

现代 VLM 推理优化涉及多个层面：

- **模型架构优化**：GQA（减少 KV cache）、MoE（只激活部分参数）、滑动窗口注意力
- **系统级优化**：vLLM / SGLang（优化的 inference serving 框架）、Flash Attention 2/3、Continuous Batching
- **算法级优化**：推测解码（Speculative Decoding，用小模型"草稿"加速大模型，通常 1.5x-2.5x）、Medusa（多头并行解码）、量化（INT4/INT8）
- **VLM 专用优化**：视觉 token 压缩（空间池化、attention pooling、token merge）、动态分辨率优化、visual cache

### 动态分辨率推理

现代 VLM（Qwen2.5-VL、InternVL2/3）支持动态分辨率，不再强制将图片 resize 到固定尺寸，而是根据图片原始尺寸动态决定切块方案。

推理过程：根据原始尺寸确定切块方案（基础块尺寸 336×336），生成局部块 + 全局视图，每块通过 ViT 编码，记录每个块在原始图片中的位置信息，所有 token 输入 LLM。

视觉 token 数与处理时间的关系（近似线性）：

| 分辨率 | 块数 | 视觉 token | 相对推理时间 |
|-------|------|-----------|------------|
| 336×336 | 1 | 576 | 1x |
| 672×672 | 4 | 2304 | 2-3x |
| 1024×768 | 6 | 3456 | 3-4x |
| 1920×1080 | 20 | 11520 | 8-12x |
| 4K (3840×2160) | 72 | 41472 | 25-40x |

所以动态分辨率虽然灵活，但也需要精心管理 token 数量。许多模型设置了最大 token 数限制。

---



## 第七章：具体模型深度解析

### InternVL 系列

InternVL 是上海 AI Lab 开发的视觉语言模型系列，以其大规模视觉编码器和强劲性能闻名。

**InternVL 的假设**：到 2023 年末，文本模型已经达到数百亿参数，而视觉编码器仍然停留在 300M-1B 参数级别。更大的视觉编码器能提取更丰富的视觉信息。

**InternVL v1（2024 年 5 月）**：发布 InternViT-6B（60 亿参数的视觉编码器），在 14 亿图文对上预训练。在多个基准上达到当时开源最优，证明了大视觉编码器的有效性。

**InternVL 1.5（2024 年 7 月）**：动态分辨率支持，从 QLLaMA 切换到 MLP Projector（更简洁）。

**InternVL2（2024 年 9 月）**：多尺寸版本（1B-76B），原生动态分辨率，统一的多模态预训练范式，多图理解。支持 4K 图片输入。

**InternVL2.5（2025 年 1 月）**：更高质量的多模态预训练数据，改进训练策略（三阶段训练），增强 OCR 能力，支持 Agent 工具调用。InternVL2.5-78B 在 30+ 个基准上超越或接近 GPT-4V。

**InternVL3（2025 年 6 月）**：视觉编码器与 LLM 原生融合（不再使用三段式），统一的多模态 token 空间，支持图文混合生成，极长上下文支持（1M+ tokens）。在多项基准上达到或超过 GPT-4o。

### Qwen-VL 系列

Qwen-VL 是阿里通义千问团队开发的多模态模型系列。

**Qwen-VL v1（2023 年 8 月）**：CLIP ViT-bigG（约 1.8B）+ Q-Former + Qwen-7B。训练分三阶段：对齐预训练（1.4B 图文对）、多任务预训练、指令微调。支持多图输入和中英文双语。

**Qwen2-VL（2024 年 8 月）**：动态分辨率（224px 到 4K），改进的 SigLIP 风格视觉编码器，原生多图/视频支持，LLM 基座升级为 Qwen2。

**Qwen2.5-VL（2025 年 1 月）**：重大更新。自研视觉编码器超越 SigLIP，超大规模多模态训练数据（数亿级），能力扩展包括多图理解、30 分钟+长视频理解、Agent 能力、30+ 种语言。多尺寸版本：3B（端侧）、7B（标准）、72B（最强）。

Qwen2.5-VL 的动态分辨率不是简单的切块：保持原始宽高比，计算最优切块数和排列方式，最小化 padding。加上一个全局缩略图提供整体感知。

**Qwen3-VL（2025 年发布技术报告）**：支持文本、图像、视频的交错长上下文，原生 256K tokens；模型家族包括 dense 2B/4B/8B/32B 和 MoE 30B-A3B/235B-A22B 版本。

### LLaVA 家族

（已在前面详细讨论，此处补充 LLaVA-OneVision 的技术细节）

LLaVA-OneVision（2025 年 1 月）的核心创新：Any Resolution + Any Frames，通过"拼接"策略将多帧视为"超长序列"，为视频帧设计特殊的 3D 位置编码（时间 + 空间），训练分单图对齐和多图/视频微调两个阶段。实现了"一个模型处理所有视觉任务"。

### 其他重要 VLM 模型

**Llama 3.2 Vision（Meta, 2024）**：Meta 的第一代官方多模态模型。SigLIP-based 视觉编码器 + MLP Projector + Llama 3.1 8B/90B。特点是与 Llama LLM 深度整合（无缝兼容 Llama 的生态），支持多图理解，11B 版本在多个基准上与 GPT-4V 相当。

**Molmo（Ai2, 2024）**：强调"数据质量"的开源 VLM。核心理念是"不是更多的数据，而是更好的数据"。约 100K 高质量的图文数据（每个样本都经过人工审核）。Molmo-7B 在多个基准上超过使用数百万数据训练的模型，Molmo-72B 接近 GPT-4V 水平。启示：数据质量远比数量重要。

**Phi-3 Vision / Phi-4 Vision（Microsoft, 2024-2025）**：小模型系列，强调"参数效率"。Phi-3.5 Vision（4.2B 参数）达到 7B 级别模型的效果，适合端侧部署。Phi-4 Vision（14B 参数，2025 年发布）更强的视觉推理能力。验证了"在高质量训练数据下，小模型也能表现优异"。

**Pixtral（Mistral, 2024-2025）**：Mistral 的多模态模型。Pixtral 12B（2024 年 9 月）首次将 Mistral 扩展到视觉领域。Pixtral Large（2025 年）更强的 OCR 和文档理解能力。

**DeepSeek-VL / DeepSeek-VL2**：DeepSeek-VL（2024 年 3 月）使用 SigLIP-L + Projector + DeepSeek-LLM-7B，混合分辨率（低+高双路编码）；DeepSeek-VL2（2024 年 12 月）使用 MoE 架构，更高的效率。

---



## 附录

### 相关论文

1. **Attention Is All You Need** (Vaswani et al., 2017) —— Transformer 基础
2. **An Image is Worth 16x16 Words** (Dosovitskiy et al., 2021) —— ViT 基础
3. **Learning Transferable Visual Models From Natural Language Supervision** (Radford et al., 2021) —— CLIP 基础
4. **Visual Instruction Tuning** (Liu et al., 2023) —— LLaVA v1，多模态指令微调的开端
5. **Improved Baselines with Visual Instruction Tuning** (Liu et al., 2024) —— LLaVA-1.5，VLM 训练的"标准配方"
6. **Sigmoid Loss for Language Image Pre-Training** (Zhai et al., 2023) —— SigLIP
7. **Qwen2.5-VL Technical Report** (Bai et al., 2025) —— 现代 VLM 的技术全景
8. **How Far Are We to GPT-4V?** (Chen et al., 2024) —— InternVL 技术报告

### 常见名词解释

| 缩写 | 全称 | 说明 |
|------|------|------|
| VLM | Vision-Language Model | 视觉语言模型 |
| ViT | Vision Transformer | 视觉 Transformer |
| CLIP | Contrastive Language-Image Pre-training | 对比图文预训练 |
| SigLIP | Sigmoid Loss for Language-Image Pre-training | 改进版对比学习 |
| LLaVA | Large Language and Vision Assistant | 经典开源 VLM |
| RoPE | Rotary Position Embedding | 旋转位置编码 |
| GQA | Grouped Query Attention | 分组查询注意力 |
| GELU | Gaussian Error Linear Unit | 高斯误差线性单元 |
| MoE | Mixture of Experts | 混合专家（稀疏激活） |
| DPO | Direct Preference Optimization | 直接偏好优化 |
| RLHF | RL from Human Feedback | 基于人类反馈的强化学习 |
| SFT | Supervised Fine-Tuning | 监督微调 |
| InfoNCE | Info Noise-Contrastive Estimation | 信息噪声对比估计 |
| MHA/MQA/GQA | Multi-Head / Multi-Query / Grouped-Query Attention | 注意力的三种变体 |
| BPE | Byte-Pair Encoding | 字节对编码（主流分词算法） |
| KV Cache | Key-Value Cache | 注意力键值缓存 |
| Flash Attention | Flash Attention | 高效注意力实现 |

