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
description: 从自回归生成、Prefill 与 Decode 到显存容量模型与优化技术的 KV Cache 完整解析。
---
# 第 1 章：从自回归到KV Cache

## 1.1 自回归

几乎所有当前主流的大语言模都采用了 **Decoder-only Transformer** 架构，并以 **自回归（autoregressive）** 的方式生成文本。自回归的含义即为：模型每次只产生一个 token，然后将这个新 token 拼接到已有的序列末尾，再将扩展后的序列作为下一轮的输入，去预测更后面的 token。然而Attention 部分的计算量约为O（n^2）的开销。

值得在这里指出的一点是：这种"逐 token 生成"的方式与人类写作有本质的不同。人类在写一句话时，脑子里通常已经有了整句话的大致轮廓，然后逐字写出来。标准自回归解码不会预先固定完整输出，而是在每一步根据当前上下文重新计算下一个 token 的条件分布；模型可以隐式表示未来结构，但其后续内容和停止位置通常不是预先确定的。

---

## 1.2 Attention 中的 Q、K、V 与 KV Cache的具体计算

我们来看 Self-Attention 的完整计算过程。

$$
\operatorname {A t t e n t i o n} (Q, K, V) = \operatorname {s o f t m a x} \left(\frac {Q K ^ {T}}{\sqrt {d _ {k}}}\right) V
$$


现在来回答本章最核心的一个问题：**为什么缓存的是 K、V，而不是 Q？**

推理时，对于一个已经生成的历史 token $j$（假设它在序列中的位置是 $p_j$，内容也已经确定），它在第 $l$ 层的隐藏表示 $h_j^{(l)}$ 在以下意义上是不变的：

- $h_j^{(l)}$ 只依赖于 $j$ 之前的 token 和 $j$ 自身的 embedding。
- 一旦 $j$ 通过了第 $l$ 层，$h_j^{(l)}$ 就被固定了下来。后续加入的新 token 不会"穿越回去"改变已经计算好的 $h_j^{(l)}$。
- 既然 $h_j^{(l)}$ 固定，那么 $k_j^{(l)} = h_j^{(l)} \cdot W_K^{(l)}$ 和 $v_j^{(l)} = h_j^{(l)} \cdot W_V^{(l)}$ 也就是固定的（其实$q_j^{(l)}$也是固定的）。

而 Query 则完全不同。**每个新来的 token 会生成自己的 Query**，并且这个 Query 会与**所有历史 Key**（包括上述 $k_j$）做交互。历史 token 的 Query 已经完成了它的历史使命（用于计算它所在位置的输出），对后续 token 的推理没有用处。

例如针对第三个token:
$$
o_{3}=\operatorname {s o f t m a x} \left(\frac{q_{3}[k_{1},k_{2},k_{3}]^{T}}{\sqrt{d_{k}}}\right)[v_{1},v_{2},v_{3}]
$$


因此：**缓存历史 K、V，而不缓存历史 Q**。

Transformer 有 $L$ 层，每一层都有自己独立的 $W_Q^{(l)}$、$W_K^{(l)}$、$W_V^{(l)}$，因此每一层都产生自己独立的 K、V 矩阵。这意味着：

- 每一层都需要自己独立的 KV Cache。
- 不同层的 KV Cache 之间**没有共享关系**——它们是针对不同特征空间的。

这导致了**总缓存大小与层数成正比**，模型层数越深，KV Cache 越大的直接原因。

假设当前已经生成了 $t-1$ 个 token，某一层的缓存中已经保存了这些 token 的 Key 和 Value：

```
K_cache = [k_1, k_2, ..., k_{t-1}]  形状: [t-1, head_dim]
V_cache = [v_1, v_2, ..., v_{t-1}]  形状: [t-1, head_dim]
```

当第 $t$ 个 token 到达该层时：

$$\text{output} = \text{softmax}\left(q_t \cdot [K_{cache}; k_t]^T\right) \cdot [V_{cache}; v_t]$$

---

## 1.3 Prefill 与 Decode

### Prefill（预填充）阶段

**定义与过程：**

Prefill 阶段处理整个输入 Prompt，一次性为所有 Prompt token 计算出注意力，并生成第一个输出 token。在这个过程中，完整的初始 KV Cache 被建立。

从计算的角度看，Prefill 做的是**矩阵-矩阵乘法**：

$$Q_{seq} \cdot K_{seq}^T$$

其中 $Q_{seq}$ 的形状为 $[S_{prompt}, H_q, D]$，$K_{seq}$ 的形状为 $[S_{prompt}, H_{kv}, D]$（$H_q$即为Query头数，$D$即为每个头的维度）。

**Prefill 阶段的计算特征：**

- **并行度高**：所有 Prompt token 可以同时参与token投影等 Attention 的计算（层与层之间串行，层内token并行）。
- **计算量大**：对于长 Prompt（如 32K token），Prefill 的计算量是可观的。
- **典型耗时**：几十毫秒到几秒，取决于 Prompt 长度和 GPU 算力。

**Prefill 建立的缓存：**

Prefill 完成后，每一层都建立了包含所有 Prompt token KV 的缓存：

```text
K_cache: [S_prompt, H_kv, D]  ← 所有 Prompt token 的 Key
V_cache: [S_prompt, H_kv, D]  ← 所有 Prompt token 的 Value
```

**TTFT（Time to First Token）：**

TTFT：从用户发出请求到收到**第一个返回 token 的时间**。

TTFT 由以下因素共同决定：

- **Prefill 计算时间**（主要因素）：取决于 Prompt 长度、模型大小和 GPU 算力。
- **排队时间**：请求在调度器中等候处理的时间。
- **Prefix Cache 命中**：如果命中，跳过部分或全部 Prefill 计算。

### Decode（解码）阶段

**定义与过程：**

Decode 阶段逐个生成后续 token。每次只处理一个 token，使用已缓存的 KV 来计算当前 token 的注意力。

从计算的角度看，Decode 做的是**矩阵-向量乘法**：

$$q_t \cdot K_{cache}^T$$

其中 $q_t$ 的形状为 $[1, H_q, D]$（单 token），$K_{cache}$ 的形状为 $[S_{total}, H_{kv}, D]$（所有历史 token）。

**Decode 阶段的计算特征：**

- **并行度低**：每次只有一个 token 参与计算，GPU 的算力资源无法被充分利用。
- **算力利用率低**：矩阵-向量乘法无法充分利用 Tensor Core，SM 利用率通常只有 10–30%。
- **内存带宽敏感**：每步需要从 HBM （高带宽内存）读取大量 KV 数据。
- **典型耗时**：每 token 5–100ms，取决于上下文长度和 GPU 带宽。

**TPOT（Time Per Output Token）：**

TPOT：**平均**每生成一个输出 token 所需的时间。

TPOT 由以下因素决定：

- **上下文长度**：每步读取的 KV 数据量随上下文线性增长。
- **HBM 带宽**：更快的 HBM 直接降低 TPOT。
- **模型大小**：更大模型有更多层，每层都需要读取 KV。
- **Batch 大小**：增大 batch 可以分摊权重读取开销，但 KV 读取量也会增加。

### 两个阶段的全面对比

| 维度                  | Prefill                              | Decode                        |
| --------------------- | ------------------------------------ | ----------------------------- |
| 输入                  | 整个 Prompt（$S_{prompt}$ 个 token） | 单 token                      |
| 计算模式              | 矩阵-矩阵（$Q_{seq}K_{seq}^T$）      | 矩阵-向量（$q_tK_{cache}^T$） |
| 并行度                | 高（seq 维度并行）                   | 低（单 token）                |
| 主要瓶颈              | 算力（Compute-bound）                | 显存带宽（Memory-bound）      |
| GPU SM 利用率（大概） | 60–90%                               | 10–30%                        |
| 关键指标              | TTFT                                 | TPOT                          |
| 对 batch 的敏感性     | 受 batch 影响较大                    | 受 batch 影响相对小           |
| 典型优化方式          | 矩阵乘法加速、FlashAttention         | KV 量化、稀疏 Attention       |

### P/D 分离：将两者分开部署

Prefill 和 Decode 对硬件资源的需求几乎相反，这自然引出一个问题：**是否应该把它们部署在不同的 GPU 上？**

#### 混合执行的问题

在一体式架构（单 GPU 同时处理 Prefill 和 Decode）中，存在几个难以解决的矛盾：

**资源竞争**：当 Prefill 和 Decode 共享同一 GPU 时，Prefill 的大矩阵乘法会占用 Tensor Core，而 Decode 需要持续读取 HBM。两者对硬件资源的工作负载特征和服务目标不同。在同一时间步内，GPU 很难同时高效处理 Prefill 和 Decode。

**长 Prompt 阻塞**：假设 GPU 当前正在处理多个请求的 Decode，此时一个新请求到达，需要进行 Prefill。如果这个新请求的 Prompt 很长（如 32K token），它的 Prefill 可能持续数秒。在这几秒内，正在 Decode 的其他请求都需要等待——它们的 P99 延迟会瞬时飙升。

**扩缩容僵化**：如果业务的特点是"Prompt很长但输出较短"，需要更多 Prefill 算力。但在混合架构中，不能单独增加 Prefill 容量——必须整台 GPU 一起增加，这意味着 Decode 资源也会被同时浪费。

#### 分离后的架构

P/D 分离将 Prefill 和 Decode 分配到不同的 GPU 集群上：

```
                       ┌──────────────────────┐
                       │   Global Scheduler   │
                       │  (路由 + 负载均衡)     │
                       └──────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
      ┌───────▼───────┐ ┌────▼────────┐ ┌───▼────────┐
      │ Prefill       │ │ Prefill     │ │ Prefill    │
      │ Worker 0      │ │ Worker 1    │ │ Worker 2   │
      │ (算力型GPU)    │ │ (算力型GPU)  │ │ (算力型GPU) │
      └───────┬───────┘ └────┬────────┘ └────┬───────┘
              │              │               │
              └──────────────┼───────────────┘
                             │ KV Cache 传输
                             │ (RDMA / NVLink)
                             ▼
              ┌──────────────┼───────────────┐
              │              │               │
      ┌───────▼───────┐ ┌────▼────────┐ ┌───▼────────┐
      │ Decode        │ │ Decode      │ │ Decode     │
      │ Worker 0      │ │ Worker 1    │ │ Worker 2   │
      │ (带宽型GPU)    │ │ (带宽型GPU)  │ │ (带宽型GPU) │
      └───────────────┘ └─────────────┘ └────────────┘
```

工作流程如下：

1. 请求到达 Global Scheduler。
2. 调度器根据当前负载和缓存位置，将请求分配给一个 Prefill Worker。
3. Prefill Worker 完成 Prompt 编码，建立初始 KV Cache，并生成第一个 token。
4. KV Cache 通过高速网络传输给一个 Decode Worker。
5. Decode Worker 从已缓存的 KV 开始逐 token 解码，直到请求完成。

#### 关键的代表性系统

在实际工程中，P/D 分离已经在多个系统中得到实现和验证：

- **DistServe**（2024 年，北京大学与字节跳动）：最早系统性地展示了 P/D 分离可以在一体式架构的基础上大幅降低 P99 延迟。其核心发现在于：分离后，长 Prompt 的 Prefill 不再阻塞 Decode 请求，尾部延迟得到了根本性的改善。

- **Mooncake**（2024 年，DeepSeek）：以 KV Cache 为中心的分离式推理系统。Mooncake 将 KV Cache 视为"分布式状态"，引入了全局缓存目录和基于 RDMA 的跨节点传输。DeepSeek 的生产推理集群基于 Mooncake 架构。

- **TensorRT-LLM**（2025 年，NVIDIA）：在 NVIDIA 的生产推理栈中提供了原生的 disaggregated serving 支持，允许用户配置 Prefill 和 Decode 的独立实例组。

#### P/D 分离的代价

P/D 分离引入了一系列新的挑战，需要在系统设计中加以解决：

**KV Cache 传输延迟**

Prefill Worker 完成计算后，需要将整个 KV Cache 传输给 Decode Worker（虽然这不是唯一的方式）。对于一个长上下文请求，KV Cache 可以非常庞大：

```text
70B 模型、32K 上下文、BF16 精度：
  KV Cache ≈ 10 GB

通过 200 Gbps RDMA 传输：
  理论传输时间 ≈ 10 GB × 8 / 200 Gbps ≈ 400ms
  
通过 800 Gbps NVLink（同节点）：
  理论传输时间 ≈ 10 GB × 8 / 800 Gbps ≈ 100ms
```

在某些场景下（特别是 Prompt 本身就不太长时），传输时间可能超过 Prefill 时间本身。

**网络带宽压力**

在高并发场景下，多个 Prefill Worker 同时完成计算，都需要将缓存传输到 Decode Worker。如果并发数很大，瞬间的网络流量可能达到数十甚至数百 GBps，对网络架构提出很高的要求。

**Decode Worker 的空闲等待**

如果 Prefill Worker 繁忙或网络拥塞，Decode Worker 可能处于空闲状态（没有请求可以解码），造成 GPU 资源的浪费。这需要在调度器中做好 Prefill 和 Decode 容量匹配。

**节点故障与缓存恢复**

在 P/D 分离架构中，如果一个 Decode Worker 崩溃，正在其上解码的所有请求的 KV Cache 都丢失了。调度器需要将这些请求重新发送给 Prefill Worker 重新 Prefill，这会显著增加端到端延迟。

**动态配比问题**

业务负载会随时间变化。例如，白天可能以短 Prompt 对话为主（需要更多 Decode Worker），晚上以长文档处理为主（需要更多 Prefill Worker）。如果 P/D 配比固定，很难适应这种变化。

#### 动态聚合与解耦

前沿研究与工程实践正在探索一种更灵活的模式：**动态聚合与解耦（Dynamic Disaggregation）**。其核心思想是：不把 P/D 分离当作一个"有或无"的二值选择，而是根据每个请求的特征动态决定。

决策因子可能包括：

- **Prompt 长度**：长 Prompt → 分离（充分利用 Prefill Worker 算力）；短 Prompt → 聚合（避免传输开销）。
- **输出长度**：长输出 → 分离（Decode 时间远大于传输时间，净收益为正）。
- **当前网络负载**：网络拥塞 → 聚合（避免进一步加重网络负担）。
- **SLO 要求**：TTFT 敏感 → 聚合（消除传输延迟）。TPOT 敏感 → 分离（保证稳定的 Decode 速度）。

通过这种方式，系统可以在一体式和分离式之间动态切换，试图对每个请求都做出最优的选择。

# 第 2 章：KV Cache 的结构、容量与性能模型

## 2.1 KV Cache 的张量结构

### 逻辑形状

KV Cache 本质上是一个多维数组。对于单个序列而言，K Cache 和 V Cache 最常见的逻辑形状为：

```text
K Cache: [batch_size, num_kv_heads, seq_len, head_dim]
V Cache: [batch_size, num_kv_heads, seq_len, head_dim]
```

各维度具体说明如下：

- **batch_size（B）**：同时处理的序列数量。在预填充阶段，可以同时处理多条独立 prompt；在解码阶段，多个请求共享同一个 batch 的不同槽位。
- **num_kv_heads（Hₖᵥ）**：Key/Value 注意力头的数量。在 MHA 中这个值与 Query Head 数相同，在 GQA 中则更小。**这是决定 KV Cache 大小的关键设计参数之一。**
- **seq_len（S）**：当前已缓存的 token 数量（从 1 到模型支持的最大长度）。在推理过程中这一维度不断增长。
- **head_dim（D）**：每个注意力头中向量的维度。常见的取值为 64（如 GPT-3）、96（如 LLaMA）、128（如 LLaMA 2/3）。

### 框架间的差异

需要特别说明的是，上述形状只是**逻辑视图**。在实际框架中，KV Cache 的物理布局可能完全不同：

- **HuggingFace Transformers**：使用逻辑上最直观的张量，形状为 `[batch, num_kv_heads, seq_len, head_dim]`。动态缓存时使用 Python list，静态缓存时预分配最大张量。
- **vLLM（PagedAttention）**：KV Cache 按 block 存储，物理地址不连续。逻辑上连续的 token 序列，在物理上可能分布在显存的各个角落，由 block table 来维护映射关系。
- **TensorRT-LLM**：使用预分配的连续内存池，通过 slot 机制管理。内存池大小在初始化时确定，运行时通过引用计数和空闲列表管理分配与回收。
- **CUDA Graph 场景**：要求所有张量的 shape 在 graph 捕获后不再变化，因此需要静态预分配最大长度的缓存。

### 额外的维度：Tensor Parallel 分片

当使用 Tensor Parallel（TP）时，KV Cache 还会多出一个"分片"维度。KV Head 在多个 GPU 之间分布，每张 GPU 只持有自己负责的那部分 KV Head。这一点将在第 3 章和第 7 章中详细讨论。

---

## 2.2 KV Cache 容量公式

### 基础公式

单个序列的 KV Cache 总字节数可以用以下公式估算：

$$M_{KV} = 2 \times L \times S \times H_{kv} \times D \times b$$

各变量含义如下表所示：

| 变量     | 含义                  | Llama 3 8B 示例值 | Llama 3 70B 示例值 |
| -------- | --------------------- | ----------------- | ------------------ |
| $2$      | K 和 V 两个矩阵的系数 | 2                 | 2                  |
| $L$      | Transformer 层数      | 32                | 80                 |
| $S$      | 缓存的 token 数       | 8192              | 8192               |
| $H_{kv}$ | KV Head 数            | 8                 | 8                  |
| $D$      | Head Dimension        | 128               | 128                |
| $b$      | 每元素字节数（BF16）  | 2                 | 2                  |

### 不同模型的对比

| 模型        | $L$  | $H_{kv}$    | $D$  | $b$  | 每 token KV | 8K 序列 | 说明                      |
| ----------- | ---- | ----------- | ---- | ---- | ----------- | ------- | ------------------------- |
| Llama 3 8B  | 32   | 8           | 128  | 2    | 128 KB      | 1 GB    | GQA 经典配置              |
| Llama 3 70B | 80   | 8           | 128  | 2    | 320 KB      | 2.5 GB  | 层数多，KV Head 少        |
| Mistral 7B  | 32   | 8           | 128  | 2    | 128 KB      | 1 GB    | Sliding Window 进一步减少 |
| GPT-3 175B  | 96   | 96          | 128  | 2    | 4.5 MB      | 36.9 GB | MHA，KV Cache 巨大        |
| DeepSeek V3 | 67   | 1 (MQA+MLA) | 128  | 2    | ~17 KB      | ~136 MB | MLA 压缩，极小            |

从表中可以清晰地看出：**MHA 模型的 KV Cache 远大于 GQA 和 MQA 模型**。这就是为什么几乎所有新模型都转向 GQA 或更激进的压缩方案。

---

## 2.3 模型权重和 KV Cache 的共同显存预算

一张 GPU 的显存远不止是给 KV Cache 准备的。以 H100（80 GB）上部署 Llama 3 70B（BF16、TP=4）为例：

```text
可用的 H100 显存: 80 GB

减去模型权重: 70B × 2 字节 ÷ 4（TP 分片）= 35 GB

减去运行时开销:
  ┣ 临时激活值（Activation Memory）：推理时中间结果，取决于 batch 和序列长度
  ┣ CUDA Context 和 Workspace：~1–2 GB
  ┣ NCCL 通信 Buffer：~1–2 GB
  ┣ CUDA Graph 预留（如果启用）：数百 MB
  ┗ 框架显存池（如 vLLM 的预先分配）
  总计运行时开销 ≈ 8–12 GB（随配置变化）

减去安全余量：建议保留 5–10% 以应对突发分配和碎片 → 约 4–8 GB

可用于 KV Cache 的显存 ≈ 80 - 35 - 10 - 8 = 27 GB
```

基于这个数字，再考虑单 token KV 的大小（320 KB / token），可以做出一阶估算：

```text
最大可缓存 token 数 ≈ 27 GB ÷ 320 KB ≈ 86,400 tokens

如果平均请求长度为 8K：
  ≈ 86,400 ÷ 8,192 ≈ 10 个并发请求

如果平均请求长度为 32K：
  ≈ 86,400 ÷ 32,768 ≈ 2–3 个并发请求
```

值得注意的是：这只是一个**理论上限**。实际可支持的并发数通常低于此值，原因将在下一节讨论。

---

## 2.4 KV Cache 与最大并发

#### 常见误区

很多人倾向于使用这样的公式来估算并发能力：

```text
最大并发 ≈ 剩余显存 ÷ (单请求最大 KV Cache)
```

例如：27 GB ÷ 2.5 GB ≈ 10 个并发请求。但这个估算存在多个理想化假设：

- **假设所有请求都达到了最大长度**：实际上，大多数请求的长度远低于最大长度。如果 P50 请求只有 4K，而最大值是 32K，按最大长度估算并发会浪费 87.5% 的潜在并发能力。
- **假设请求是静态的**：实际上请求在推理过程中不断增长，缓存需求也在动态变化。前几秒只需要几千 token 的缓存，后几秒可能就需要几万。
- **假设碎片为零**：在 PagedAttention 中，如果 block size 为 16，实际只用了 17 个 token 就需要 2 个 block（浪费 15/32 ≈ 47% 的空间）。

#### 更贴近实际的估算方法

一种更稳健的做法是基于**业务长度分布的分位值**来估算：

```text
基于平均值的估算：
  平均请求长度 = 平均 Prompt 长度 + 平均 Output 长度
  最大并发 ≈ 可用 token 数 ÷ 平均请求长度

基于 P95 的估算（作为水位线）：
  P95 请求长度 = P95 Prompt + P95 Output
  水位线并发 ≈ 可用 token 数 ÷ P95 请求长度
```

使用 P95 而非最大值来设定水位线，可以在"不浪费大量显存"和"留有余量"之间取得更好的平衡。当然，当系统负载超过水位线时，调度器需要做好排队或抢占的准备。

#### 其他影响并发的因素

除了缓存容量外，实际并发还受到以下因素的共同约束：

- **Prefix Cache 共享**：如果多个请求共享前缀（如 System Prompt），实际的缓存消耗会显著降低。
- **请求完成时的缓存释放**：Continuous Batching 中，请求完成后立即释放 KV block，新请求可以即时接入。
- **抢占策略**：缓存不足时，可以选择重计算（丢弃缓存后重新 Prefill）或 Offload（将缓存迁移到 CPU 内存），不同的策略对有效并发有不同的影响。

---

## 2.5 Decode 的内存带宽瓶颈

#### 为什么 Decode 是 Memory-bound

在 Decode 阶段，每生成一个 token，Attention 计算需要读取当前层**全部历史 KV**。对于给定的上下文长度 S：

```text
每步 Decode 从 HBM 读取的 KV 数据量 = 2 × H_kv × S × D × b
```

以 70B 模型、8K 上下文为例，每步需要读取 2.5 GB 的 KV 数据。这些数据需要从 GPU HBM（约 3.35 TB/s）搬运到片上 SRAM，然后参与计算。

但搬运这些数据后，GPU 执行的计算量是多少？每个 token 的 Decode 注意力大致需要：

```text
计算量 ≈ 2 × S × D（点积）+ S × D（加权求和）
       = 3 × S × D 次浮点运算
       = 3 × 8192 × 128 ≈ 3.1 × 10⁶ FLOPs
```

算力/带宽比可以帮助我们理解瓶颈所在。H100 的官方规格：

- 算力 = 989 TFLOPS（FP16 Tensor Core）
- HBM 带宽 = 3.35 TB/s
- 算力/带宽比 = 989 × 10¹² ÷ (3.35 × 10¹²) ≈ 295 FLOPs/Byte

每从 HBM 读取 1 字节，GPU 需要做 295 次浮点运算才能"不浪费"算力。但在 Decode 注意力中，读取 1 字节 KV 伴随的计算量远低于这个值。结果是：**GPU 的 Tensor Core 大部分时间在等待数据到达，而非真正计算**。这就是"Memory-bound"的含义。

#### 上下文长度对 TPOT 的影响

可以建立一个简化的性能模型来估计 TPOT 的下限：

```text
TPOT_attention ≈ (每步读取的 KV 数据量) ÷ (HBM 带宽)
```

| 上下文长度 | 每步读取 KV | H100 理论 TPOT | H200 理论 TPOT | 说明               |
| ---------- | ----------- | -------------- | -------------- | ------------------ |
| 1K         | ~0.3 GB     | ~0.1 ms        | ~0.07 ms       | H200 带宽 4.8 TB/s |
| 8K         | ~2.5 GB     | ~0.8 ms        | ~0.5 ms        | GQA 下显著降低     |
| 32K        | ~10 GB      | ~3.1 ms        | ~2.1 ms        | 长上下文开始变慢   |
| 128K       | ~40 GB      | ~12.2 ms       | ~8.3 ms        | 带宽瓶颈极其明显   |

需要强调的是，这只是 Attention 部分的**理论下限**。实际的 TPOT 还包括 MLP 层计算、模型权重读取、Kernel 启动开销、跨卡通信（TP 下）等，通常比上表数值高 2–5 倍。

#### Batch 大小的影响

增大 Batch 可以摊薄权重读取的开销，提高整体硬件利用率。但每个请求的 KV Cache 必须被独立读取——因为不同请求的 KV 内容不同。因此：

- **小 Batch**：GPU 算力利用率低，大量 Tensor Core 空闲，TPOT 主要被权重读取和 Kernel 启动开销主导。
- **大 Batch**：总 KV 读取量随 Batch 线性增长，HBM 带宽成为刚性约束。Batch 增大带来的延迟增加可能超过吞吐收益。
- **最优 Batch**：通常在算力利用率和内存带宽之间做权衡，经验上通过压测寻找拐点。

# 第 3 章：模型结构如何减少 KV Cache

理解 KV Cache 的成本不仅由推理框架和运行时决定，更根本地由模型架构决定。选择 MHA、MQA、GQA 还是 MLA，意味着 KV Cache 的大小可能相差数倍甚至数十倍。此外，Tensor Parallel 下的分片策略、Sliding Window 等结构设计也会显著影响缓存的规模。

---

## 3.1 MHA、MQA 与 GQA

#### 三者的结构差异

三种注意力机制的核心区别在于 **Q、K、V 的 Head 数是否一致**。

**Multi-Head Attention（MHA）**：
这是原始 Transformer 论文中提出的结构。每个注意力头都拥有**独立的** Query、Key、Value 投影矩阵。Query Head 数与 KV Head 数相等。

**Multi-Query Attention（MQA）**：
MQA 走了一个极端，所有 Query Head 共享同一组 Key 和 Value（也就是$W_K$和$W_V$都是相同的）。KV Cache 立即缩小为 MHA 的 $1/n_{heads}$。

**Grouped-Query Attention（GQA）**：
GQA 是 MHA 和 MQA 之间的折中方案：将 Query Head 分成 $g$ 组，每组内的 Query Head 共享一组 K、V。KV Cache 大小为 MHA 的 $g/n_{heads}$。

LLaMA 2 70B 率先在 2023 年采用了 GQA（8:64），此后 LLaMA 3、Mistral、Qwen 2、Gemma 等主流模型纷纷跟进。GQA 已成为当前大模型的事实标准。

#### 为什么 GQA 能保持模型质量

一个关键问题：减少 KV Head 数为什么不会显著影响模型质量？

直觉上的理解是：Key 和 Value 在注意力机制中的角色是"提供信息"，而 Query 的角色是"检索信息"。多个不同的 Query Head 可以从同一组 K、V 中提取不同的特征。因为注意力权重由每个 Query Head 独立计算，softmax 后的分布不同，加权求和的结果自然也不同。

实验表明，KV 空间的"多样性"需求远低于 Query 空间。将 KV Head 降低到 Query Head 的 1/4 或 1/8 时，大多数标准评测任务上观察不到可测量的质量下降。当然，在极端长上下文或对细粒度注意力要求极高的任务上，差异仍然可能存在——这正是实际部署前需要进行充分评估的原因。

---

## 3.2 Tensor Parallel 下的 KV Head

#### Head 分片规则

当使用 Tensor Parallel（TP）将模型分布到多张 GPU 时，Attention Head 被分片。分片的方式取决于模型结构：

**MHA（KV Head 数 >= TP Size）：**

```text
TP=4, Q Heads=32, KV Heads=32
  GPU 0: Q Heads[0-7],  KV Heads[0-7]
  GPU 1: Q Heads[8-15], KV Heads[8-15]
  GPU 2: Q Heads[16-23], KV Heads[16-23]
  GPU 3: Q Heads[24-31], KV Heads[24-31]
  每卡缓存 = 完整缓存的 1/4
```

**GQA（KV Head 数 < TP Size，但能整除）：**

```text
TP=8, KV Heads=8
  GPU 0: KV Head[0], GPU 1: KV Head[1], ... 每卡 1 个 KV Head
  每卡缓存 = 完整缓存的 1/8
```

**GQA/MQA（KV Head 数 < TP Size，不能整除）：**

```text
TP=8, KV Heads=4
  每 GPU 分不到 1 个完整的 KV Head
  → 部分 KV Head 需要在多张 GPU 上复制
  → 总缓存占用增加（相同的 KV Head 存了多份）
```

#### 关键问题：为什么增加 TP Size 不一定线性降低每卡 KV Cache

当 KV Head 数小于 TP Size 时，会出现"反直觉"的现象：**增加 TP Size 不会减少每张卡的 KV Cache 大小**。

例如：KV Heads=4、TP Size=8：即使有 8 张 GPU，每张卡仍然需要保存 4 个 KV Head 中的某几个——因为无法均匀分片，部分 Head 需要复制到多卡上。一张 GPU 上的 KV Cache 可能是 "完整缓存的一半" 而非 "完整缓存的 1/8"。

这意味着在实际部署中，TP Size 的选择需要与 KV Head 数配合。**一个经验建议**：TP Size 应不超过 KV Head 数（对于 GQA），否则复制带来的额外缓存开销和通信代价可能超过并行计算带来的收益。

#### 补充：Transformer中如何使用TP

Transformer 里通常把两个线性层配成一组——前一个做 **Column Parallel**，后一个做 **Row Parallel**。这样中间结果可以一直留在各张 GPU 本地，直到模块末尾才通信一次。

---

## 3.3 Sliding Window 与局部注意力

并非所有注意力层都需要看到完整的上下文。在 **Sliding Window Attention** 中，每个 token 只关注最近的 $W$ 个 token，窗口之外的 token 不可见：

```
位置 0 ... 1000 ... 2000 ... 3000 ... 5000
对于 W=4096、位置 5000 的 token:
  可关注范围 = [5000 - 4096 + 1, 5000] = [905, 5000]
  位置 0–904：不可见
```

#### 对 KV Cache 的影响

Sliding Window 为 KV Cache 设置了一个**固定的上限**：

```text
最大缓存 = min(实际序列长度, W) × 单 token KV

实际序列长度 = 100K, W = 4K:
  缓存 = 4K × 单 token KV（而非 100K × 单 token KV）
```

这意味着，Mistral 7B 在使用默认窗口大小 4096 时，即使处理长达 100K 的输入，其 KV Cache 的上限也只有 4K 所对应的值。在许多部署场景中，这可以大幅降低显存需求。

#### 全局与局部混合

值得注意的是，许多模型并没有在所有层使用滑动窗口。常见的模式是：

- **底层（如第 0 层）**：全局注意力，缓存所有历史 token。
- **中间层（如第 1 到 L-2 层）**：滑动窗口注意力，只缓存最近 W 个 token。
- **顶层**：全局注意力。

这种混合设计试图兼顾"长距离信息捕捉"和"缓存效率"。Mistral 和 LongLLaMA 等模型采用了类似的策略。

#### 常见误解

"支持超长上下文"这个宣称需要谨慎理解。一些号称支持 128K 或更长上下文的模型，可能只在部分层使用完整注意力，其余层使用滑动窗口或稀疏注意力。在评估模型能力时，了解其**逐层的注意力模式**比只看宣称的上下文长度更有意义。

---

## 3.4 MLA（Multi-head Latent Attention）

传统方法（MHA、GQA、MQA）减少 KV Cache 的策略都是**减少 KV Head 的数量**。而 DeepSeek 在 V2/V3 模型中提出的 **Multi-head Latent Attention（MLA）**是压缩每个 Head 的表示维度。在标准 GQA 中，每个 KV Head 直接缓存完整的 Key、Value 向量（维度为 $D$）。在 MLA 中，每个 token 的 Key、Value 先被**压缩**到一个低维的"潜空间"中，缓存的是这个低维表示。在计算注意力时，再通过一个上投影矩阵将潜变量"解压"回完整维度。

对第 t 个 token，MLA 先计算：

$c\mathbf c_t^{KV} = W^{DKV}\mathbf h_t, \qquad \mathbf c_t^{KV}\in\mathbb R^{d_c}$

这里的 D 表示 down projection。

随后，从同一个潜变量中恢复所有 Head 的 Key 内容部分和 Value：
$$
\mathbf k_{t,i}^{C} = W_{i}^{UK}\mathbf c_t^{KV}\\
\mathbf v_{t,i}^{C} = W_{i}^{UV}\mathbf c_t^{KV}
$$


其中 i 是 Query Head 编号。

但 RoPE 带来了一个特殊问题。MLA 通常把 Key 拆成：
$$
\mathbf k_{t,i}
=
\left[
\mathbf k_{t,i}^{C};
\mathbf k_t^{R}
\right]
$$
$\mathbf k_{t,i}^{C}$：与位置无关的“内容 Key”，从潜变量恢复（模型并不一定必须让 Key 的所有维度都承担位置编码）。

$\mathbf k_t^{R}$：应用了 RoPE 的位置 Key，各个 Query Head 共享。

因此，每个 token 每层真正需要缓存的是：
$$
\mathbf c_t^{KV},\quad \mathbf k_t^{R}
$$
缓存宽度约为：$d_c+d_r$

而不是传统 MHA 的：$d_c+d_r$

DeepSeek-V3 的公开配置中，常见参数是：$d_c=512,\qquad d_r=64$

所以缓存宽度为：$512+64=576$

这也是 FlashMLA 文档中经常出现 `head_dim_k=576`、`head_dim_v=512` 的原因。NVIDIA Megatron Core 的 MLA 配置同样列出了 512 的 KV 低秩维度、128 的内容 QK 维度和 64 的位置编码维度。

- **减少 KV Head 数**（GQA/MQA）：减少的是"多少个 Head"——相当于减少的是"宽度"。
- **MLA**：减少的是"每个 Head 存储多少信息"——相当于减少的是"深度"。

理论上，两者可以结合使用（DeepSeek V3 正是这样做的）。

MLA 有一个工程代价：它需要专门的 CUDA Kernel 实现标准 Attention。在目前的生态中：

- 标准 Attention Kernel（如 FlashAttention）需要修改才能支持 MLA 的潜变量解压。
- Kernel 是否将"解压 + Attention"融合为一个算子，直接影响 MLA 的实际效率。
- 如果解压是一个独立的 Kernel 调用，其启动开销和额外数据传输可能部分抵消缓存减少带来的收益。

# 第 4 章：KV Cache 的工程实现与内存管理

## 4.1 简单连续缓存实现

最简单的 KV Cache 实现是预先分配一个固定大小的连续张量，每次推理时，新 token 的 K、V 被写入对应的位置。这种方法被称为 **Static Cache**，常见于 HuggingFace 的简单推理脚本和实验代码中。

这种实现有两个显著的优点：一是内存连续，访问效率高；二是最大长度固定，适合 **CUDA Graph** 等预先编译的静态 Shape 场景。但它也有明显的缺点：**按最大长度预分配**意味着每个序列都要预留可能远大于实际所需的空间。如果最大上下文设为 32K 而实际平均只有 4K，那么约 87.5% 的预留空间处于闲置状态。

另一种变体是 **Dynamic Cache**，即按需扩容，用多少分配多少。这避免了 Static Cache 的浪费，但动态拼接和重新分配带来了额外的运行时开销和内存碎片。

综合来看，简单连续缓存的适用场景是：单请求实验、固定 batch 的本地推理、以及需要 CUDA Graph 静态 shape 的场景。它不适合请求长度高度动态的高并发生产服务。

---

## 4.2 PagedAttention

PagedAttention 借用了操作系统虚拟内存分页的思想。其核心洞察是：**每个请求在逻辑上需要连续的 KV Cache 序列，但物理上不必如此**。

具体来说：

- **Logical Block（逻辑块）**：请求将自己的 KV Cache 视为一段连续的 token 序列，被划分为固定大小的 block（如每 block 16 个 token）。
- **Physical Block（物理块）**：实际的 GPU 显存中，KV Cache 被存储在不连续的物理 block 中。
- **Block Table（块表）**：维护逻辑 block 到物理 block 的映射关系。请求在计算 Attention 时，通过 block table 找到实际数据所在的位置。

PagedAttention 的主要贡献在于**消除了内部碎片**：请求不再需要按最大长度预留连续空间，而是随着 token 的生成逐 block 分配物理内存。在请求长度差异较大的场景下，这可以显著提高显存利用率。vLLM 在其论文中报告，PagedAttention 可以将 KV Cache 的显存利用率提升至接近 100%（相比连续缓存的 20%–60%）。

此外，PagedAttention 还天然地支持**块级共享**：多个请求的前缀如果相同，它们对应的物理 block 可以共享同一份内存数据，仅通过增加引用计数来管理。

PagedAttention 也有自身的成本：

- **地址间接访问**：每次 Attention 计算都需要通过 block table 查找物理地址，增加了 Kernel 的实现复杂度。
- **Block 对齐浪费**：如果 block size 为 16 而请求只用了 17 个 token，最后一个 block 只有 1 个有效 slot，浪费了 15 个 slot 的空间。block size 的选择需要在碎片率和 Kernel 效率之间权衡。
- **它不减少理论 KV 元素数量**——PagedAttention 优化的是内存管理，而不是 Attention 的数学计算量。

---

## 4.3 Continuous Batching 与缓存生命周期

传统推理中，一个 batch 的请求同时开始、同时结束。**Continuous Batching** 则允许请求动态加入和退出：每个请求完成生成后立即退出 batch，释放的资源可以被新到达的请求使用。这使得 GPU 在每个时间步都能"塞满"尽可能多的待解码 token，从而提高硬件利用率。

每个请求的 KV Cache 经历以下阶段：

1. **分配**：请求被调度器接受后，系统检查剩余缓存容量。如果缓存不足，调度器可能选择排队、抢占已有请求（将其缓存 offload 到 CPU 或直接丢弃以备后续重算），或直接拒绝新请求。
2. **Prefill**：对 Prompt 做预填充计算，建立初始 KV Cache。如果启用 Prefix Cache，命中部分的 block 直接从共享池引用，无需重新计算。
3. **Decode**：每生成一个新 token，在缓存中追加对应的 block。如果请求在 Prefix Cache 中有不完整的最后 block，可能触发 **Copy-on-Write** 机制以确保写隔离。
4. **释放**：请求完成后，其占用的所有物理 block 被回收，引用计数归零。

在 Continuous Batching 中，调度器必须始终感知以下信息：

- 当前剩余可用物理 block 的数量
- 每个请求当前占用的 block 数量及其预测增长趋势
- 哪些请求的前缀可以共享 block
- 当缓存不足时，选择哪条策略（重计算、offload 还是拒绝请求）

---

## 4.4 Position、Mask 与缓存正确性

KV Cache 的正确维护远比分配和释放复杂。最容易出错的环节是**位置编码**。

#### position_ids

每 token 的 RoPE 编码依赖于它的绝对位置。在 Prefill 阶段，输入是一段连续的 token 序列，position_ids 从 0 开始递增。在 Decode 阶段，每个新 token 的位置是当前序列长度之后的第一个整数。如果 position_ids 传递错误，RoPE 编码就会出错，进而导致注意力分数完全错误。

#### Prefill 分块时的位置连续性

当使用 Chunked Prefill 时，一个长 Prompt 被切成多个 chunk 分别处理。每个 chunk 的 token 位置必须保持全局连续——第一个 chunk 的 position_ids 为 [0, 1, ..., chunk_size-1]，第二个 chunk 为 [chunk_size, chunk_size+1, ...]，以此类推。任何一个 chunk 的位置错位都会导致 Attention 计算的异常。

#### Sliding Window 的位置处理

Sliding Window 丢弃了窗口外的 KV，但**剩余 token 的位置编码保持不变**。位置的编号不会被"重新编"成从 0 开始，而是保持其在原始序列中的实际位置。

#### Batch 重排与 Cache Reorder

在 Beam Search 中，每一步都会淘汰低概率的分支。KV Cache 需要按照保留 beam 的索引进行重排。如果重排操作有误，保持的 token 与缓存内容对应不上，生成的文本将出现错乱。

# 第 5 章：缓存复用与推理调度

## 5.1 Prefix Cache

在大模型推理中，许多请求可能共享相同的 Prompt 前缀，例如相同的 **System Prompt**、工具定义、Few-shot 示例或问答模板的固定开头。如果这些请求都独立做 Prefill，相同前缀对应的 KV 会被反复计算，造成算力和时间的浪费。**Prefix Cache** 的核心思想很简单：将已经计算过的 KV block 缓存起来，后续请求如果前缀相同，直接复用这些 block，跳过对应部分的 Prefill。

在实际实现中（如 vLLM 和 TensorRT-LLM），Prefix Cache 的工作流程如下：

1. **Block 级别匹配**：Prompt 按 block size（如 16 token）切分成逻辑块，每个块具有独立的 K、V 数据。
2. **Block Hash**：每个 block 的内容通过一个哈希函数计算出标识。由于块与块之间存在顺序依赖，block N 的哈希通常依赖于 block N-1 的哈希，从而确保前缀的连续性——如果前一个 block 不同，后一个 block 即使 token 相同也不会命中。
3. **引用计数和共享**：每个物理 block 维护一个引用计数。当多个请求共享同一个 block 时，引用计数递增；当某个请求不再需要该 block 时，计数递减，归零时释放。
4. **Copy-on-Write（写时复制）**：如果某个请求需要对共享 block 进行修改（例如该 block 恰好是它的最后一个不完整 block），系统不会直接修改共享 block，而是分配一个新的物理 block 将内容复制过去，再将新请求的映射指向新 block。这保证了共享 block 的稳定性和线程安全。

Prefix Cache 也需要**淘汰机制**来防止无限增长。常见策略包括：LRU、LFU、大小感知策略。

---

## 5.2 Prefix Cache 的收益边界

Prefix Cache 并非在任何场景下都能带来显著收益，理解其边界对于正确配置和使用至关重要。

Prefix Cache 主要影响 **TTFT**——因为它跳过了重复的 Prefill 计算。对于后续的 Decode 阶段（TPOT），由于 Decode 本来就不需要重新计算前缀，Prefix Cache 几乎没有直接影响。

命中 Prefix Cache 的前提是 **token 级别完全匹配**。这意味着：

- 只有完全相同的 token 前缀才能复用。"几乎相同"或"意思相近"是不够的。
- Prompt 中如果包含时间戳、随机 ID、用户特有的 session 信息，或使用不同模板生成，都会破坏前缀的连续性。
- 模型版本、LoRA 配置和 RoPE 基频等训练参数也必须完全一致。一个用不同模型版本预填充的 KV block 不能直接用于另一个版本的推理。

"是否开启 Prefix Cache"不如"命中率有多高"重要。命中率低于一定阈值时，Prefix Cache 可能反而因为元数据存储和哈希计算的开销而得不偿失。设计 Prompt 模板时，如果能将公共部分放在前缀位置，将动态变化的部分放在后缀位置，可以显著提高命中率。

---

## 5.3 典型复用场景

#### 高收益场景

- **共享 System Prompt**：在 ChatML 格式中，用户的系统消息通常出现在 prompt 的最开头且保持不变。这是 Prefix Cache 最高效的应用场景之一。
- **多轮对话**：每一轮对话的 prompt 包含所有历史轮次。后续轮次可以复用前面轮次的全部 KV，只有最新的一轮需要计算。
- **Agent 的工具定义**：AI Agent 通常需要在 prompt 中包含大量工具/函数定义。这些定义在同一 Agent 的不同调用之间是不变的。
- **Few-shot 示例**：固定示例位于 prompt 开头，所有请求共享。
- **批量处理相同模板**：在离线批处理中，所有数据的 prompt 结构相同，只是填充内容不同。

#### 低收益场景

- **随机前缀**：如果每个请求的 prompt 都完全不同，命中率为零。
- **共同前缀很短**：当共享内容只占 prompt 的极小部分时，节省的 Prefill 时间有限。
- **Decode 很长但 Prefill 很短**：当 Prefill 时间本身就可以忽略不计时，即使命中率高，端到端收益也不明显。
- **缓存频繁被淘汰**：如果系统的总缓存池太小，命中的 block 很快被挤出去，下一次请求又要重新 Prefill。

---

## 5.4 Chunked Prefill

在 Continuous Batching 中，一个长 Prompt 的 Prefill 会长时间占用 GPU，导致该 GPU 上正在 Decode 的其他请求被阻塞。这会造成这些请求的 TPOT 出现剧烈抖动。

Chunked Prefill 将一个长 Prompt 的 Prefill 计算分解为多个较小的 chunk，每个 chunk 与其他请求的 Decode 交错执行：

```text
时间 →
请求 A（长 Prompt）: [chunk1] [空闲] [chunk2] [空闲] [chunk3] ...
请求 B（Decode）:     [decode] [decode] [decode] [decode] [decode] ...
```

每个时间步，GPU 在处理 A 的 Prefill chunk 的同时，也会处理 B 的 Decode token。这避免了长 Prefill 对 Decode 的长时阻塞。

#### chunk size 的影响

- **chunk 过小**：矩阵太薄，GPU 算力利用率下降，无法充分利用 Tensor Core。
- **chunk 过大**：对 Decode 的阻塞时间变长，尾部延迟恶化。
- 典型选择范围：256–2048 token 之间。

---

## 5.5 调度目标与缓存约束

推理调度器通常需要在以下目标之间权衡：

- **Token Throughput（token 吞吐）**：每秒生成的总 token 数。
- **Request Throughput（请求吞吐）**：每秒完成的总请求数。
- **TTFT SLO（Service Level Objective）**：首 token 必须在给定时间内返回。
- **TPOT SLO**：每个 token 的生成速度必须维持在给定阈值以下。

在调度决策中，**缓存容量本身就是重要的约束条件**。调度器必须回答这些问题：

- 当前还有多少可用物理 block？
- 如果接入这个新请求，剩余缓存是否足够支撑到它完成？
- 如果不够，应该拒绝它、排队等待、还是抢占一个正在运行的请求？

典型的抢占选择包括：**重计算**（释放请求的 KV block，后续重新 Prefill）、**Offload**（将 KV block 传输到 CPU 内存后再载回）和**直接拒绝**。不同策略在延迟、吞吐和系统复杂度方面各有取舍。

一个先进的调度器会根据每个请求的 SLO 紧迫程度动态分配资源。接近超时的请求获得更高的优先级，而刚到达的请求可以稍微等待。这需要调度器能实时追踪每个请求的 TTFT 和 TPOT 累积情况，并将其与目标值进行比较大。

# 第 6 章：KV Cache 的核心优化技术

## 6.1 KV Cache 量化

需要先做一个重要的区分：**权重量化**和 **KV 量化**是两回事。权重量化对模型的权重矩阵做低位压缩，一次量化后可以使用多次。而 KV 量化针对的是推理过程中动态产生的 K、V 张量，需要在每次推理中都执行量化和反量化操作。

KV Cache 中的数值通常具有相对集中的分布。与模型权重相比，K 和 V 对量化的敏感度有时较低。=，因为注意力计算中的 softmax 操作本身具有一定的"容错性"：注意力分数经过 softmax 归一化后，小范围的量化误差可能对最终的加权求和结果影响有限。当然，这并不意味着量化没有代价。近年来的研究表明，KV 量化导致的误差在某些长上下文或高精度要求的任务上可能会积累并变得显著。

#### 常见精度方案

| 精度 | 每元素字节数 | 举例                  | 说明                                 |
| ---- | ------------ | --------------------- | ------------------------------------ |
| BF16 | 2            | 无量化                | 基线，无精度损失                     |
| FP8  | 1            | H100 原生支持         | 精度损失较小，当前主流方案           |
| INT8 | 1            | 需要 scale/zero point | 量化方法成熟，框架支持广泛           |
| INT4 | 0.5          | 需要分组量化          | 精度损失较大，需要精细的 Kernel 支持 |

#### 量化粒度的权衡

同样一种精度，使用不同量化粒度会产生不同的效果：

- **Per-Tensor**：整个张量共享一个 scale，实现最简单，但精度损失最大。
- **Per-Channel**：每个 Head 有独立的 scale，精度有所改善。
- **Per-Token**：每个 token 独立量化，适合处理异常值。
- **Per-Group**：每 N 个值一组独立 scale（在D维度上切分），精度最接近原始值，但存储 scale 的额外开销也随之增加。

K 和 V 的数值分布存在**明显差异**：K 的分布通常更分散，更容易出现异常值；V 的分布相对更集中。这意味着它们可能需要不同的量化策略——对 K 使用更保守的量化（如更大的分组或更高的精度），而对 V 可以采用更激进的量化。

一个常被忽视的事实是：**理论容量减少不等于推理一定加速**。量化省下的显存是确定的，但省下的带宽是否转化为更快的 token 生成，取决于反量化操作是否能高效地融合到 Attention Kernel 中。如果反量化是一个独立的 Kernel 调用，其启动和运行开销可能抵消大部分带宽收益。

KV 量化的前沿正从"固定统一位宽"走向**按层、按 token、甚至按值重要性动态分配精度**。例如，重要的层使用高精度（FP8），不重要的层使用更低精度（INT4）；最近的 token 保留高精度（即 Residual Cache 方案），较早的 token 使用低位量化。这些自适应方案试图在"尽可能压缩"和"关键信息不丢失"之间找到更优的平衡点。

---

## 6.2 Token Eviction 与稀疏 KV

Token Eviction 的目的是**减少需要缓存的 token 数量**。思路是：并非所有历史 token 对当前输出都有同等重要的影响，一些 token 可能永远不会再被 Query 注意到，可以将其从缓存中删除。

Sliding Window 是最简单的淘汰方式——统一丢弃窗口外的所有 token。而 Token Eviction 试图做得更精细：通过分析注意力模式，只丢弃那些"不太可能被后续 token 关注"的 token。

研究发现，以下类型的 token 在注意力中扮演着重要角色：

- **Recent Token**：最近生成的 token，与当前 token 的语义关联最强。
- **Heavy Hitter**：被很多后续 token 频繁关注的 token，通常是一些主题性的关键信息载体。
- **Sink Token**：序列开头的 token（如 BOS、第一个词），往往天然获得较高的注意力。

#### 常见策略

- **H2O（Heavy Hitter Oracle）**：保留累积注意力分数最高的 token，淘汰最低的。
- **Scissorhands**：通过观察 token 在注意力中被访问的频率来判断其重要性。
- **StreamingLLM**：保留 sink token（开头）和最近的 token，淘汰中间的——简单而有效。

需要明确区分三个概念：

- **KV 量化**：减少每个元素的大小。
- **Token Eviction**：减少保留的 token 数量。
- **Sparse Attention**：仍然存储所有 KV，但计算时只读取一部分。

三者可以组合使用，但优化的目标和代价各不相同。

---

## 6.3 CPU Offload 与分层存储

GPU HBM 虽然速度快（H100 为 3.35 TB/s），但容量有限（80 GB）。当上下文长度超过 HBM 能容纳的范围时，可以考虑将部分 KV 迁移到更低层级的存储：

- **CPU DRAM**：容量可达数百 GB，带宽 ~200 GB/s（PCIe Gen4 x16），延迟 ~10μs。
- **NVMe SSD**：容量可达数 TB，带宽 ~7 GB/s，延迟 ~100μs。

活跃推理中的请求的 KV 属于"热缓存"，应保留在 GPU HBM。那些暂时不活跃但仍可能被复用的 KV（如某轮对话的历史上下文）属于"冷缓存"，可以 offload 到 CPU 内存甚至 NVMe。

Offload 的核心在于**异步传输与计算重叠**：在 GPU 计算当前 token 的同时，通过 DMA 或 RDMA 异步传输下一批需要的 KV block。如果能将传输时间"隐藏在"计算时间背后，offload 对 TPOT 的影响可能降至最低。

当需要释放 GPU HBM 空间时，调度器面临一个选择：是把请求的 KV Cache offload 到 CPU（未来可以恢复），还是直接丢弃（未来需要重新 Prefill）？

通常情况下，offload 需要传输大量数据（延迟高），但恢复快；重新计算不需要传输，但会消耗 GPU 算力。在长上下文中，offload 通常更有优势，因为重新计算长序列的 Prefill 成本很高。

---

## 6.4 各优化技术的统一比较

| 技术           | 主要优化对象       | 主要收益                  | 主要代价                         |
| -------------- | ------------------ | ------------------------- | -------------------------------- |
| GQA/MQA        | KV Head 数         | 从模型结构上减少缓存      | 需模型本身支持，可能影响质量     |
| PagedAttention | 内存碎片与动态分配 | 提高实际显存利用率 2–4 倍 | Kernel 复杂度增加，地址间接访问  |
| Prefix Cache   | 重复 Prompt 计算   | 降低 TTFT 50–90%          | 依赖前缀命中率，增加元数据存储   |
| KV 量化        | 每个元素的位宽     | 降低容量和访存量 2–4 倍   | 精度损失，反量化开销             |
| Token Eviction | 缓存的 token 数    | 控制长上下文下的缓存增长  | 可能丢失重要信息                 |
| CPU Offload    | GPU 驻留容量       | 突破 HBM 容量限制         | 数据传输延迟                     |
| Sliding Window | 历史长度           | 为缓存设置固定上限        | 长距离信息完全丢失               |
| MLA            | KV 表示维度        | 大幅减少单 token 缓存     | 需训练阶段支持，解压 Kernel 开销 |

**这些技术并非互斥**。一套高性能推理系统通常将它们组合使用：模型层用 GQA 或 MLA 从结构上减负，存储层用 PagedAttention 和量化管理物理存储，计算层用 Token Eviction 和 Sparse Attention 优化访存，调度层用 Prefix Cache 和 Offload 扩展容量。

# 第 7 章：分布式 KV Cache 与大规模推理系统

## 7.1 Tensor Parallel 下的缓存

当使用 Tensor Parallel（TP）时，Attention 的 Head 被分布到多张 GPU 上。对于 KV Cache，分布方式取决于模型结构：

- **MHA**：Query Head 和 KV Head 数相等，两者可以一起均匀分片。TP=4、KV Heads=32 时，每卡负责 8 个 KV Head。
- **GQA**：KV Head 数小于 Query Head 数。分片时看 KV Head 能否被 TP Size 整除。可以整除则完美分片，不能整除则需要复制。
- **MQA**：KV Head 只有 1 个，无法分片，GPU 都保存同样的 KV Cache 副本。

#### KV Head 过少时的复制问题

当 KV Head 数小于 TP Size 时，部分 KV Head 必须在多张 GPU 上复制：

```text
TP=8, KV Heads=4（GQA）:
  每张卡分配不到 1 个完整的 KV Head
  → 每张卡都被迫保存完整的 KV Cache（的 KV Head 部分）
  → 增加 TP Size 不减少每卡 KV Cache 大小
```

这解释了为什么：**增加并行度不一定线性降低每卡的缓存负担**。在实际部署中，TP Size 的选择需要兼顾模型结构和硬件配置。通常建议 TP Size 不超过 KV Head 数（对于 GQA），避免不必要的复制。

#### Decode 阶段的通信开销

在 TP 下，Attention 计算完成后需要跨卡做 all-reduce 以合并各头的输出。这个通信开销在 Decode 阶段尤为突出：每生成一个 token 就需要一次 all-reduce。当 TP Size 很大时，通信可能成为瓶颈。因此，大模型推理中常见 TP 与 PP（Pipeline Parallel）的组合使用——TP 控制每张卡的局部计算，PP 控制跨节点的流水线。

---

## 7.2 Prefill/Decode 解耦

#### 动机回顾

Prefill 和 Decode 对硬件的需求截然不同：

- **Prefill** 需要大量算力来处理长序列的矩阵-矩阵乘法，适合算力型 GPU。
- **Decode** 受显存带宽限制，需要快速从 HBM 读取大量 KV 数据，适合带宽型 GPU。

在混合执行时，两者互相干扰：Decode 的低并行度拉低 Prefill 的 GPU 利用率，而 Prefill 的长时执行推高 Decode 请求的 P99 延迟。

#### 解耦架构

P/D 分离将两种计算分配到不同的 GPU 集群：

1. 请求首先到达一个 Global Scheduler。
2. 调度器决定将请求发往某个 **Prefill Worker**。
3. Prefill Worker 完成计算后，将 KV Cache 通过高速网络传输给一个 **Decode Worker**。
4. Decode Worker 从收到的 KV Cache 开始逐 token 解码，直到请求完成。

DistServe 和 Mooncake 等系统是这一方向的早期代表。到 2025–2026 年，TensorRT-LLM 等生产推理栈也提供了解耦式服务能力。解耦的核心优势在于：

- 两类 Worker 可以独立扩缩容。
- 长 Prefill 不会阻塞 Decode 请求，P99 延迟大幅改善。
- 每类 Worker 可以用最适合自身负载的 GPU 型号。

#### KV Cache 传输

Prefill 完成后，KV Cache 需要从 Prefill Worker 传输到 Decode Worker。传输的粒度和方式有多种选择：

- **按请求整体传输**：简单直接，但长上下文的传输延迟可能很高。
- **按层传输**：Decode Worker 可以在收到第一层的 KV 后就开始计算，与后续层的传输重叠。
- **按 block 传输**：最细粒度，与 PagedAttention 的 block 管理天然契合。

传输协议方面，RDMA（如 InfiniBand、RoCE）提供最低的延迟和最高的带宽，NVLink 在节点内传输时表现最优，TCP 则适用于通用网络但延迟最大。

---

## 7.3 解耦架构的代价

#### 主要挑战

P/D 分离并非在所有场景下都优于一体式架构。它带来的额外代价包括：

- **KV 传输延迟**：一个 32K 上下文的 70B 模型 KV Cache 约 10 GB（BF16）。即使通过 200 Gbps RDMA 传输，也需要约 400ms——这可能比 Prefill 本身还要长。
- **网络带宽压力**：高并发下，KV Cache 传输可能占满网络带宽，成为新的瓶颈。
- **Decode Worker 等待**：如果 Prefill Worker 繁忙或 KV 传输延迟，Decode Worker 可能出现空闲等待。
- **节点故障与缓存丢失**：如果 Decode Worker 崩溃，该节点上所有请求的 KV Cache 都丢失了，需要重新 Prefill。
- **配比难确定**：业务的 Prompt/Output 长度分布变化时，Prefill 和 Decode Worker 的最佳配比也随之变化。

#### 短 Prompt 场景的权衡

当 Prompt 很短时，Prefill 本身只需要很少的时间和算力。此时 KV 传输的代价可能超过 Prefill 的节省。前沿研究正探索**动态选择**——根据请求的特征（Prompt 长度、输出长度、网络负载等）决定是走解耦路径还是一体式路径，从而在两种架构之间找到最佳平衡。

---

## 7.4 KV-aware Routing

传统的负载均衡器通常只关心 GPU 的当前负载（如利用率、队列长度），对缓存的分布一无所知。但在 KV Cache 的世界里，"数据在哪里"可能比"负载有多重"更重要。

#### 路由策略

- **Session Affinity（会话亲和性）**：同一用户/同一会话的请求始终路由到同一个节点，最大化多轮对话中的 Prefix Cache 命中率。
- **Prefix-aware Routing**：路由器维护一个全局的缓存目录（或分布式哈希表），记录每个前缀的 KV Cache 所在的节点。新请求到达时，优先路由到缓存命中率最高的节点。
- **热前缀复制**：对于最常出现的前缀（如头部 System Prompt），在多个节点之间主动复制其 KV Cache，使更多节点可以"命中"它。

#### 命中率与负载的平衡

缓存亲和路由可能加剧负载不均衡：拥有热门缓存的节点负载越来越高，而其他节点相对空闲。调度器需要在"缓存命中"和"负载均衡"之间做出权衡。一个常见的做法是：为缓存命中设置一个阈值，只有当命中节省的时间超过某个值时，才容忍一定程度的负载不均衡。

---

## 7.5 KV Cache 作为分布式存储系统

#### 整体视角

当把视野扩大到整个推理集群，KV Cache 已经超越了"每个请求的私有缓存"这一定位，演变为一个**分布式状态系统**。它具备传统分布式存储系统的许多特征：

- **分配**：新的请求需要分配物理 block（在哪里分配？）。
- **定位**：已有的 KV block 分布在不同节点上（在哪里？）。
- **共享**：多个请求可以共享同一个 block（通过引用计数和 Copy-on-Write）。
- **传输**：block 需要跨节点移动（通过 RDMA、NVLink）。
- **淘汰**：需要决定哪些 block 可以移除（LRU、LFU、大小感知）。
- **恢复**：节点故障时，丢失的 KV block 需要重新计算或从备份恢复。

在分布式架构中，**Cache Manager** 逐渐成为独立于推理引擎的系统组件。它维护全局的缓存状态，为调度器和路由器提供查询接口，并负责缓存内容的迁移和复制