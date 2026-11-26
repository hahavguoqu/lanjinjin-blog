---
title: KV Cache
date: 2026-07-14 00:00:00
categories:
  - AI Infra
tags:
  - KV Cache
  - LLM 推理
  - Prefill
  - Decode
description: 从显存账与带宽账出发，系统拆解 KV Cache 的数学原理、容量与吞吐模型、架构压缩（GQA/MLA）、分页调度与分布式解耦。
---

把一个 70B 的大语言模型部署到 4 张 80 GB 的 H100 上，模型权重按 BF16 切分后每张卡只占 35 GB，看起来还剩一半多显存。可真把线上流量接进来，如果用户发来几条 32K 的长上下文请求，吞吐很快就会掉下去，再多来几条并发甚至直接 OOM。

吃掉剩下显存、又卡住每秒吐字速度的元凶，就是 **KV Cache**。它最初只是一个为了省掉自回归重复计算的“空间换时间”技巧，但当上下文从 4K 拉长到 128K、并发从单请求变成在线集群时，KV Cache 已经不再是一块简单的临时数组，而是决定并发上限、首字延迟（TTFT）、吐字速度（TPOT）乃至集群拓扑的核心系统状态。

下文先从计算本身讲清楚为什么只缓 K 和 V、Prefill 与 Decode 为什么脾气截然相反，再拿 **Llama 3 70B（80 层、64 个 Q 头、8 个 KV 头、$D=128$、BF16）运行在 H100（80 GB / 3.35 TB/s / 989 TFLOPS FP16）** 这组具体条件把容量账和带宽账算透，最后顺着“模型架构 → 显存管理 → 调度复用 → 量化卸载 → 分布式集群”把一整套优化拆开。

---

## 一、为什么缓存的是 K 和 V，而不是 Q

主流大语言模型采用 Decoder-only 架构与因果掩码（Causal Mask），以自回归方式逐个生成 token：每步吐出 1 个新 token，拼回序列末尾，再跑一次前向去预测下一个。

如果每一步都把前面已经生成好的 $t-1$ 个 token 重新过一遍全部 Transformer 层，单步自注意力的计算复杂度就是 $O(t^2)$，生成 $T$ 个 token 的总开销会膨胀到 $O(T^3)$。但仔细看因果掩码下的依赖关系就会发现大量重复计算：对于历史位置 $j < t$ 的 token，它在第 $l$ 层的隐藏状态 $h_j^{(l)}$ 只取决于 $1 \dots j$ 这些更早的 token，后面新生成的 token 永远不会反过来修改 $h_j^{(l)}$。因此，由 $h_j^{(l)}$ 线性投影得到的 $k_j^{(l)} = h_j^{(l)} W_K^{(l)}$ 与 $v_j^{(l)} = h_j^{(l)} W_V^{(l)}$ 一旦算完就终身不变。

至于历史 token 的 Query 向量 $q_j^{(l)}$，虽然同样固定，但它在第 $j$ 步算完该位置的注意力输出后就已经完成了使命——第 $t$ 步计算新 token 的输出时，只需要用当前步的新查询向量 $q_t$ 去和全部历史 Key 做点积，再对全部历史 Value 加权求和：

$$
o_t = \operatorname{softmax}\!\left(\frac{q_t [K_{\text{cache}};\, k_t]^T}{\sqrt{d_k}}\right) [V_{\text{cache}};\, v_t]
$$

这就是**只缓存历史 K、V 而不缓存历史 Q** 的唯一原因。又因为 Transformer 的 $L$ 层各自拥有独立的投影权重 $W_K^{(l)}, W_V^{(l)}$，每一层都必须维护自己独立的 KV Cache，总缓存体积天然与层数 $L$ 成正比。

---

## 二、Prefill 与 Decode：同一个模型，两种相反的负载

引入 KV Cache 之后，一条推理请求的执行过程裂成了特征截然相反的两个阶段。

![Prefill 与 Decode 阶段的计算形态与硬件瓶颈对比](/img/posts/kv-cache/kv-prefill-decode.svg)

*图 1：Prefill 以矩阵-矩阵乘为主、吃满算力；Decode 每步退化为矩阵-向量乘，主要时间花在从 HBM 搬运权重与历史 KV。*

**Prefill（预填充）阶段**一次性接收全部输入 Prompt（长度 $S$）。由于所有输入 token 已知，层内可以沿序列维度高度并行，计算形态是稠密的矩阵-矩阵乘法（GEMM），权重从 HBM 读进片上缓存后可以被 $S$ 个 token 共同复用，算术强度很高，GPU 的 Tensor Core 通常能跑到 60%–90% 的利用率。这一阶段走完后会建立起完整的初始 KV Cache，并吐出第一个输出 token，它直接决定了用户的首字延迟 **TTFT（Time to First Token）**。

**Decode（解码）阶段**每次只进 1 个新 token。单请求下，计算形态退化成矩阵-向量乘法（GEMV）：为了算这 1 个 token，GPU 既要把整套模型权重完整读一遍，还要把当前上下文全部历史 KV 从 HBM 搬进片上 SRAM。算力单元大部分时间在空等数据，属于典型的**显存带宽受限（Memory-bound）**。它决定了相邻两个输出 token 的间隔 **ITL（Inter-Token Latency）** 以及平均每 token 耗时 **TPOT（Time Per Output Token）**。

| 维度 | Prefill 阶段 | Decode 阶段 |
| :--- | :--- | :--- |
| 输入形态 | 整段 Prompt（$S$ 个 token 并行） | 每步 1 个新 token |
| 核心算子 | 矩阵-矩阵乘（GEMM） | 矩阵-向量乘（GEMV） |
| 硬件瓶颈 | 算力受限（Compute-bound） | 显存带宽受限（Memory-bound） |
| 关键指标 | TTFT（首字延迟） | ITL / TPOT（吐字间隔） |
| 典型优化 | FlashAttention、Prefix Cache | 大 Batch 合批、GQA/MLA、KV 量化 |

---

## 三、把显存账和带宽账算清楚

### 3.1 容量公式与典型模型对比

逻辑上，单条序列每一层的 K、V 张量形状为 `[batch_size, num_kv_heads, seq_len, head_dim]`。对于长度为 $S$ 的单条序列，全部 $L$ 层占用的 KV Cache 总字节数为：

$$
M_{\text{KV}} = 2 \times L \times S \times H_{kv} \times D \times b
$$

其中系数 $2$ 对应 K 和 V 两个矩阵，$b$ 为单个元素的字节数（BF16/FP16 为 2 字节，FP8/INT8 为 1 字节）。套进几个代表模型，差距非常直观（均按 BF16 计算）：

| 模型 | 层数 $L$ | KV 头数 $H_{kv}$ | 头维 $D$ | 单 token KV | 8K 序列 KV | 32K 序列 KV |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama 3 8B（GQA） | 32 | 8 | 128 | 128 KiB | 1.0 GiB | 4.0 GiB |
| Llama 3 70B（GQA） | 80 | 8 | 128 | 320 KiB | 2.5 GiB | 10.0 GiB |
| GPT-3 175B（MHA） | 96 | 96 | 128 | 4.5 MiB | 36.0 GiB | 144.0 GiB |
| DeepSeek-V3（MLA） | 61 | 压缩潜向量 | $d_c+d_r=576$ | ~68.6 KiB | ~549 MiB | ~2.14 GiB |

注意 DeepSeek-V3 的算法不同：它主模型共 61 层，每层不再缓存多头 K、V，而是只缓存宽度为 $d_c + d_r = 512 + 64 = 576$ 的压缩潜向量与共享位置 Key，因此单 token 字节数为 $61 \times 576 \times 2 = 70\,272\ \text{bytes} \approx 68.6\ \text{KiB}$，比同量级 GQA 模型还要再小近一个数量级。

### 3.2 4 卡 H100 上部署 Llama 3 70B 的真实并发上限

回到开头的部署场景：4 张 80 GB H100，以张量并行 **TP = 4** 运行 BF16 精度的 Llama 3 70B。先算**单张卡**上的显存账：

- 可用总显存：80 GiB
- 模型权重分片：$70\text{B} \times 2\ \text{bytes} / 4 \approx 35\ \text{GiB}$
- 运行时开销（临时激活、CUDA Context、NCCL 通信缓冲、CUDA Graph 预留）：约 10 GiB
- 安全余量（防突发分配与碎片）：约 8 GiB
- **单卡留给 KV Cache 的空间**：$80 - 35 - 10 - 8 = 27\ \text{GiB}$

接下来最容易算错的地方在于分母：Llama 3 70B 有 8 个 KV 头，在 TP = 4 下每张卡只存 $8 / 4 = 2$ 个 KV 头，因此**单卡上的单 token KV 体积是 $320 / 4 = 80\ \text{KiB}$**（4 张卡合起来才构成完整的 320 KiB/token）。于是整组 4 卡实例能同时容纳的理论 token 上限是：

$$
\frac{27 \times 1024 \times 1024\ \text{KiB}}{80\ \text{KiB/token}} \approx 353\,894\ \text{tokens}
$$

如果线上请求的平均总长度（Prompt + Output）为 8K，理论可支撑约 43 个并发请求；如果拉到 32K，就只剩约 10 个并发。这还只是静态理论上限——真实服务里请求长度动态增长，不能按“最大长度”死板预留（会浪费绝大多数空间），也不能只按平均值算（流量尖峰立刻 OOM），工程上通常会按 **P95 长度** 设水位线，并配合调度器的排队与抢占机制兜底。

### 3.3 为什么 Decode 一定撞上显存带宽墙

再算带宽账。H100 的 FP16 稠密算力为 989 TFLOPS，HBM 带宽为 3.35 TB/s，硬件的算力/带宽拐点（Roofline 脊点）约在 $989 / 3.35 \approx 295\ \text{FLOPs/Byte}$。也就是说，每从显存搬 1 字节数据，至少要做近 300 次浮点运算才能把算力喂饱。

但在单请求 Decode 时，Attention 部分每步必须把全部 $L$ 层历史 KV 读一遍（以 70B、8K 上下文、单卡视角计，单卡读取量为 $2.5\ \text{GiB} / 4 = 640\ \text{MiB}$；全模型 4 卡合计 2.5 GiB）。而对每个字节的 KV 数据，它只和 $q_t$ 对应的几个 Query 头做一次乘加（对于 GQA $H_q / H_{kv} = 8$，BF16 每元素 2 字节，算术强度只有 $2 \times 8 / 2 = 8\ \text{FLOPs/Byte}$），连 H100 脊点的零头都够不到。

只看 Attention 读取 KV 这一项的理论耗时下限（单卡 3.35 TB/s，TP=4 下单卡读 1/4 KV）：

| 上下文长度 | 全模型 KV（4 卡合计） | 单卡每步读 KV | H100 单卡仅 KV 读取下限 |
| :--- | ---: | ---: | ---: |
| 1K | 0.31 GiB | 80 MiB | ~0.025 ms |
| 8K | 2.50 GiB | 640 MiB | ~0.20 ms |
| 32K | 10.0 GiB | 2.50 GiB | ~0.80 ms |
| 128K | 40.0 GiB | 10.0 GiB | ~3.20 ms |

这还只是 1 条请求的 KV 搬运时间；实际单步 TPOT 还要加上权重读取（35 GiB / 3.35 TB/s ≈ 10.4 ms）、Kernel 启动和跨卡 All-Reduce 通信。**增大 Batch 可以把权重读取的 10.4 ms 摊薄给几十条请求，但不同请求的 KV 内容互不相同，总 KV 读取量会随 Batch 线性上涨。** 因此，小 Batch 时瓶颈在权重读取，大 Batch 长上下文时瓶颈会迅速转移到 KV 读取，最优并发点必须靠压测在吞吐和 TPOT 之间找平衡。

---

## 四、从模型结构源头给 KV Cache 瘦身

推理框架只能管好显存，真正把单 token 字节数砍掉几倍甚至几十倍的，是模型架构本身的演进。

![MHA、GQA、MQA 与 MLA 单 token KV 缓存结构对比](/img/posts/kv-cache/kv-mha-gqa-mla.svg)

*图 2：从标准多头（MHA）、分组共享（GQA）、全共享（MQA）到低秩潜变量（MLA）的每层缓存结构对比。*

### 4.1 MHA → GQA → MQA 与张量并行的约束

标准 **MHA** 给每个 Query 头都配了独立的 K、V 投影，KV Cache 最大。**MQA** 走到另一个极端，所有 Query 头共享同一组 K、V，缓存缩到 $1/H_q$，但训练稳定性和长文质量有时会打折扣。**GQA（Grouped-Query Attention）** 取中间路线，把 $H_q$ 个 Query 头分成 $H_{kv}$ 组共享 K、V，在几乎不掉点的前提下把 KV Cache 砍到 $1/4$ 或 $1/8$，已经成为 Llama 3、Qwen 2/3、Mistral 等模型的默认配置。

GQA 在多卡 **Tensor Parallel（TP）** 部署时有一条非常实际的硬约束：**TP 切分数最好不要超过 KV 头数 $H_{kv}$**。比如模型有 8 个 KV 头、TP=8 时，每张卡刚好分 1 个头，单卡 KV 降为 $1/8$；可如果一个模型只有 4 个 KV 头却强行开 TP=8，每张卡分不到完整的 1 个 KV 头，只能把同一个 KV 头复制到两张卡上，导致**卡数翻倍，单卡 KV 占用却一字节没少**，总显存消耗反而翻倍。

### 4.2 滑动窗口（Sliding Window）与 MLA

**Sliding Window Attention**（如 Mistral 7B，窗口 $W=4096$）通过直接丢弃窗口外的历史 KV，给每层的缓存钉死一个常数上限 $\min(S, W)$。很多号称支持超长上下文的模型，实际采用的是“少数层全局注意力 + 多数层滑动窗口”的混合结构，评估长文召回能力时必须看清它的逐层配置。

**MLA（Multi-head Latent Attention）** 则换了一个维度压缩：GQA 减的是“头的数量”，MLA 减的是“每个 token 存下来的表示宽度”。在 DeepSeek-V2/V3 中，第 $t$ 个 token 的隐藏状态先通过下投影矩阵压缩成一个低维联合潜向量：

$$
\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \in \mathbb{R}^{d_c}
$$

推理计算时，理论上再通过上投影矩阵恢复各头的内容 Key 与 Value（工程上还会利用矩阵乘法结合律，把上投影矩阵直接吸收到 Query 权重与输出权重里，避免在显存里显式展开完整多头 K/V）。为了兼容旋转位置编码 RoPE，MLA 把 Key 拆成内容部分与带 RoPE 的共享位置部分 $\mathbf{k}_t^R \in \mathbb{R}^{d_r}$，最终每个 token 每层只需缓存 $\mathbf{c}_t^{KV}$ 和 $\mathbf{k}_t^R$，总宽度为 $d_c + d_r = 512 + 64 = 576$，而不是传统 MHA 下的 $2 \times H_q \times D$（例如 $2 \times 128 \times 128 = 32\,768$）。MLA 的代价在于标准 FlashAttention 无法直接复用，必须编写融合解压/矩阵吸收的专用算子（如 FlashMLA）。

---

## 五、显存管理与调度：分页、前缀共享与分块预填充

即便模型本身已经用了 GQA，早期推理引擎按“最大序列长度”预先分配连续张量（Static Cache），在请求长度参差不齐时依然会造成 60%–80% 的显存空置浪费；而按需动态扩容（Dynamic Cache）又会带来频繁拷贝与外部碎片。

### 5.1 PagedAttention 与 Prefix Cache

vLLM 提出的 **PagedAttention** 借用了操作系统虚拟内存分页的思路：每条请求在逻辑上仍然看到一段连续的 token 序列，但物理上被切分成固定大小（如 16 个 token）的 **Physical Block**，分散存放在显存池的任意位置，由每个请求自己的 **Block Table** 维护逻辑块到物理块的映射。

![PagedAttention 块表映射与 Prefix Cache 共享机制](/img/posts/kv-cache/kv-paged-prefix.svg)

*图 3：两条共享 System Prompt 的请求通过 Block Table 指向同一组物理块（引用计数 ref=2），各自后缀独立分配不连续物理块。*

分页顺带打开了 **Prefix Cache（前缀缓存）** 的大门。在多轮对话、共享长 System Prompt 或 Agent 携带大量工具定义的场景下，只要按块计算链式哈希（第 $k$ 块的哈希依赖第 $k-1$ 块的哈希，确保前缀严格连续），新请求命中后直接把自己的 Block Table 指向已有物理块、递增引用计数，就能跳过对应前缀的全部 Prefill 计算，大幅压低 TTFT。如果某条请求要往一个尚未写满的共享尾块追加新 token，系统会触发 **Copy-on-Write（写时复制）**，拷贝出一份新物理块再写入，保证其他请求的共享前缀不被污染。

使用 Prefix Cache 时有两条边界必须注意：第一，命中前提是 **token 级完全一致**，只要 System Prompt 开头塞了当前时间戳或随机请求 ID，整条前缀都会失效；第二，它只省 Prefill（压 TTFT），对后续 Decode 的 TPOT 几乎没有直接加速作用。

### 5.2 Continuous Batching、Chunked Prefill 与正确性细节

传统静态批处理必须等同一批里最长的请求吐完，短请求才能退出，GPU 大量空转。**Continuous Batching（迭代级调度）** 让请求在任意一步完成后立即释放物理块，新请求在下一轮迭代即可插队进入 batch。

但合批之后会冒出一个新矛盾：一条 32K 的长 Prompt 进来做 Prefill，可能霸占 GPU 几百毫秒甚至数秒，导致同卡上正在 Decode 的几十条请求全部卡死，P99 ITL 瞬间飙升。**Chunked Prefill** 的解法很干脆——把长 Prompt 切成固定大小（常见 512–2048 token）的 chunk，每轮迭代只算一个 chunk，并和当前活跃请求的 Decode token 拼在同一个 batch 里执行。这样既把长 Prefill 的卡顿摊平了，又顺便用 Prefill 的大矩阵乘把 Decode 空闲的算力填满。

在工程落地中，维护 KV Cache 最隐蔽的 bug 往往出在**位置编码与掩码的对齐**上：

- **Chunked Prefill 的位置连续性**：第 2 个 chunk 的 `position_ids` 必须严格接在第 1 个 chunk 末尾（从 `chunk_size` 起步而非从 0 重来），否则 RoPE 角度全乱。
- **Sliding Window 的位置编号**：窗口外的旧 KV 被丢弃后，留下的 token 依然保持它们在原始序列里的真实位置，绝不能重新从 0 编号。
- **Beam Search 的重排**：每步淘汰分支时，必须同步按保留索引重排 KV Cache，否则上下文与分支会张冠李戴。

---

## 六、逼近极限时的手段：量化、淘汰与分层卸载

当分页和前缀共享都用满，显存或带宽依然顶不住时，系统层还有三张牌可打，但每一张都有明确代价：

1. **KV Cache 量化**：与一次离线完成的权重量化不同，KV 量化是在线动态执行的。目前 H100 上最实用的是 **FP8 KV Cache**（显存和访存减半，硬件原生支持、精度几乎无损）；再往下压到 INT4 通常需要 Per-Group 分组量化，且因为 Key 比 Value 更容易出现通道级异常值（Outlier），前沿方案往往对 K 用更保守的精度或更细的分组。需要警惕的是：**显存省下来不等于推理一定变快**——如果反量化没有融合进 Attention Kernel，额外启动一次反量化算子反而可能把省下的带宽吐回去。
2. **Token Eviction 与稀疏注意力**：前者（如 StreamingLLM 保留开头 Sink Token + 最近窗口、H2O 保留高累积注意力分数的 Heavy Hitter）是真把旧 KV 扔掉以省显存；后者（Sparse Attention）仍保留全部 KV，只在计算时挑子集读取以省带宽。两者都面临“大海捞针”类任务漏掉关键细节的风险。
3. **CPU Offload 与分层存储**：把暂时不活跃的冷对话 KV 异步换出到 CPU DRAM（PCIe Gen4/5 几十 GB/s 级）甚至 NVMe SSD，需要续聊时再拉回 GPU。调度器在显存告急时面临“直接丢弃将来重算 Prefill”还是“Offload 到 CPU”的权衡：短 Prompt 直接重算更快，长上下文（如 32K+）则往往是异步 Offload 更划算。

| 优化手段 | 作用对象 | 核心收益 | 主要代价 |
| :--- | :--- | :--- | :--- |
| GQA / MLA | 模型架构 | 从源头缩小单 token KV 4×–50× | 需训练阶段决定；MLA 需专用 Kernel |
| PagedAttention | 显存分配 | 消除预留碎片，并发提升 2×–4× | 块表间接寻址，Kernel 实现变复杂 |
| Prefix Cache | 重复前缀 | 跳过重复 Prefill，大幅压低 TTFT | 依赖严格前缀命中，需维护淘汰策略 |
| Chunked Prefill | 调度粒度 | 消除长 Prompt 对 Decode 的卡顿 | chunk 过小会拉低 Prefill 算力利用率 |
| KV 量化（FP8/INT4） | 元素位宽 | 容量与带宽占用减半至 1/4 | 精度风险；需融合反量化算子 |
| Token Eviction | 缓存长度 | 控住超长上下文显存增长 | 不可逆丢失被淘汰的历史信息 |
| CPU Offload | 存储层级 | 突破单机 HBM 容量墙 | 受限于 PCIe/RDMA 传输延迟 |

---

## 七、走出单机：P/D 分离与分布式 KV 状态

把视角放大到多机集群，前面提到的 Prefill（吃算力）与 Decode（吃带宽）之间的资源冲突，催生了 **Prefill/Decode 分离（Disaggregated Serving）** 架构——代表系统包括学术界的 **DistServe**（北京大学与字节跳动）、工业界的 **Mooncake**（月之暗面与清华大学，以分布式 KV 缓存池为核心）以及 **TensorRT-LLM** 等推理框架的原生解耦模式。

思路很清晰：把集群拆成专门跑 Prefill 的 Worker 池和专门跑 Decode 的 Worker 池。请求先到 Prefill 节点算出初始 KV Cache，再通过高速网络（节点内 NVLink，跨节点 InfiniBand/RoCE RDMA）把 KV 传给 Decode 节点继续逐字生成。这样做不仅彻底消除了长 Prompt 对 Decode 尾部延迟的干扰，还能根据业务特点独立扩缩容两类节点的数量，甚至混用不同型号的 GPU。

但 P/D 分离绝不是免费午餐。拿前面算过的 Llama 3 70B 32K 上下文（10 GiB KV Cache）算一笔传输账：即便走 200 Gbps 的 RDMA 网络，单次整包传输的理论下限也要 $10 \times 8 / 200 \approx 400\ \text{ms}$。如果请求本身 Prompt 很短，传输耗时甚至可能超过 Prefill 本身；高并发下跨节点 KV 流量还会打满网络带宽，而且 Decode 节点一旦故障，上面的在途 KV 状态全部丢失，只能退回 Prefill 重算。因此工程上正在叠加三层补救：一是**按层或按 block 流水传输**，把传输延迟藏在计算背后；二是 **KV-aware Routing**，路由器维护全局前缀目录，优先把请求发往已经缓存了对应前缀的节点（并在“命中率”与“负载均衡”间设阈值防热点）；三是**动态解耦**，短 Prompt 就地跑一体式，长 Prompt 或长输出才走分离路径。

走到这一步，KV Cache 已经完全演变成了一个真正的**分布式存储与状态管理系统**——分配、寻址、哈希共享、写时复制、跨节点迁移、LRU 淘汰与故障恢复，传统分布式存储里的经典命题，几乎全部在推理引擎里重演了一遍。
