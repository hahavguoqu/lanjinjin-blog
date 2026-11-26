---
title: Transformer
date: 2025-10-19 00:00:00
categories:
  - 大模型
tags:
  - Transformer
  - Attention
description: Transformer 论文（Attention Is All You Need）核心概念、架构实现与性能分析笔记。
---
### 一、摘要的摘要

​	We propose a new simple network architecture, the Transformer,based solely on **attention mechanisms**, dispensing with recurrence and convolutions entirely. 

​	完全依赖注意力机制，完全摒弃了循环和卷积操作。



### 二、名词释义

#### sequence transduction

序列转换：将一个输入序列映射为另一个输出序列

演化历史：RNN Encoder-Decoder --> Attention(2015) --> Transformer(2017)

​	RNN Encoder-Decoder用“隐藏状态”处理上下文（时序信息），仍存在“**长距离依赖**”的问题。

​	Attention机制**依附**在RNN/LSTM的编码器-解码器结构上，所以仍存在**计算并行性差**、长距离依赖等问题。Attention解决了“模型该注意哪里”的问题，而Transformer进一步推动了更高效解决问题。注意，Attention机制天然具有并行计算的优势，但是在早期的模型上都是依附于RNN的decoder部分，局部可以做到并行计算，整体流程仍然受限于RNN的序列结构。



#### sequential computation

​	The Extended Neural GPU,ByteNet and ConvS2S这些模型仍旧以CNN为基本模块（并非RNN时间序列模型），可以并行计算所有输入输出的隐藏表示，但是由于CNN的感受野有限，需要多层传播，所以在这些模型中还是会出现问题：将两个任意输入或输出位置的信号关联起来所需的运算量随着位置间距距离的增长而增加，即”**长距离依赖问题**（long-range dependency）“。不过，这些CNN具有**多通道**的优良特性，Transformer在设计 multi-head attention的时候实际上就是实现了多通道的特性，具备了丰富的特征表达能力。



#### memory network

**显式**地为神经网络增加一个**可读写**的**外部**记忆模块。memory network借助了attention发展。

​	RNN并不能满足这个要求，因为RNN的记忆是”隐式“的，连续更新的（递归计算得到的），无法直接访问历史某个具体的信息单元；即使我们存储了所有的历史纪录，也缺乏一种机制选择访问其中哪一个历史状态。

​	在memory network中模型中通常通过**注意力**机制**访问权重**：
$$
p_i=\text{softmax}(q^Tm_i)
$$
​	然后读取加权和：
$$
o=\sum_ip_i\:m_i
$$
其中:

- **$q$**：查询向量，表示当前的问题或模型希望检索的信息，例如输入句子或问题的编码表示。
- **$m_i$**：记忆单元，表示第 $i$ 条存储的信息（如词、句子或知识片段）的向量表示，构成模型的外部记忆库。
- **$p_i$**：注意力权重，对所有匹配分数进行 softmax 归一化后得到，表示模型在第 $i$ 个记忆单元上的关注程度。
- **$o$**：读出结果，对所有记忆单元进行加权求和的结果，作为模型最终“读出”的综合信息。

​	这种形式是 attention 思想的抽象表达，具体实现会根据模型有所不同，例如 Bahdanau Attention (2015) 和 End-to-End Memory Networks (2015) 都有类似思想。需要注意 :memory network是一个体系结构，而attention是其中一种“可微寻址”（通过连续、可微的**权重分布**来决定读写记忆的位置）的机制。

​	同时，Transformer ≠ Memory Network，因为Transformer的self-attention机制输出的是一组上下文相关内部表示（模型前向传播中即时计算得到的**特征向量**），隐式地在内部表征之间做信息检索，而不是一种显式可读写的“外部记忆库”（持续储存的具体数据或独立的参数集合）。



#### Seq2Seq

​	序列到序列模型（Sequence-to-Sequence）,采用**“编码器-解码器”**结构（Encoder-Decoder）。编码器把离散的符号（如单词字符等）转化成连续的向量表示（embedding），解码器将这些连续表示生成目标序列。



#### auto-regressive

自回归：在解码器逐步生成目标序列的时候，每一步生成一个符号，此时解码器会使用编码器生成的上下文信息和之前已经生成的目标序列的符号，即当前输出**依赖于之前**的输出。这样可以捕捉目标序列的前后依赖关系（如上下文信息），然而这**不方便并行化**计算，生成速度较慢。

​	最原始transformer架构同样是采用编码器-解码器，而解码器也是使用自回归的。在编码器阶段使用自注意力机制，此时可以极大加快运行速度。而后有部分transformer架构采用非自回归结构，但是通常会产生精度较低的问题。

​	由于传统transformer解码器采用自回归实现，所以模型在**推理**阶段的自回归生成会有速度瓶颈，这就是扩散LLM等模型研究的动因之一。



### 三、具体实现

![transformer_structure](/img/posts/transformer/transformer_structure.png)



#### embedding

词嵌入：将离散的符号（如单词字符等）转化成连续的向量表示。

​	Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{model}$. 

​	与其他序列转换模型类似，我们使用**可学习**的嵌入，将输入 token 和输出 token 转换为维度为 $d_{model}$ 的向量。
$$
E_{token}=X\times W_e
$$
其中：

- $X$: 输入的 token 序列 ，每个 token 是一个索引

- $W_e$: 词嵌入矩阵，大小是 $(V, d_{model})$ （V 是词表大小，$d_{model}$ 是模型维度）

​	而上面这个式子是有问题，是一种符号化写法，表示用$X$去查embedding矩阵$W_e$。在实际操作中，不会直接创建独热矩阵进行计算（过于稀疏浪费内存与运算速度），使用**查表**得到。同时，在实际操作中，$X$不会是一个单独的句子，多为**batch输入**，即一次喂给模型多条序列，让模型**并行计算**多个样本。

​	假如批量大小B=32，每个句子长度 L=20，词表大小 V=10000，模型维度$d_{model} = 512$

| 名称        | 含义               | 维度                |
| ----------- | ------------------ | ------------------- |
| $X$         | 批输入的 token ID  | $(B, L)$            |
| $W_e$       | 词嵌入矩阵（共享） | $(V, d_{model})$    |
| $E_{token}$ | 查表结果           | $(B, L, d_{model})$ |

​	这个式子在线性代数中是不成立的，因为是查表得到的，只是作为展示。

​	随后有三个问题：

##### **各个句子L并不固定**

​	如果L长度不一样，就无法保证embedding输出矩阵形状一致，后续无法组成batch以及矩阵计算。采取**padding**的方式，找出**当前**batch中的最大句长，然后用特殊符号<pad>将所有句子都padding(填充)到该长度。然后会通过Mask机制忽略Padding，没有必要去关注这些填充信息。然后还会有些优化策略，比如将长度相似的句子分配在一个batch减少padding的浪费。

##### **$d_{model}$长度固定为512的作用意义**

​	确保各层之间张量形状一致，保证各个模块输入、输出一致。

##### **$W_e$是如何得到的**

​	在原始transformer中，模型会通过注意力、解码器等结构，预测输出词分布，再根据结果与真实值的差距，**反向传播学习**得到结果。虽然一开始所有数据都是随机化的，但是在不断地学习中神经网络可以捕捉到语义信息。这里的训练时间就极其之大了。不过，以GPT-3为例，词嵌入矩阵计算量占比仍小于0.1%，最主要的还是transformer层的训练。在后面更现代的模型中，通常会使用**预训练**好的$W_e$，然后结合下游任务进行**微调**。

​	在最后还需要注意，在embedding之前必须要经过**分词**这个操作，然后不同的模型会使用不同的分词方法。



#### positional encoding

​	Since our model contains no recurrence and no convolution, in order for the model to make use of **the order of the sequence**, we must inject some information about the **relative or absolute position** of the tokens in the sequence. To this end, we add **positional encodings** to **the input embeddings** at the bottoms of the encoder and decoder stacks. The positional encodings have the **same dimension**, $d_{model}$, as the embeddings, so that the two can be summed. There are many choices of positional encodings, **learned and fixed**. In this work, we use **sine and cosine functions of different frequencies**.

​	由于我们的模型既不包含循环结构，也不包含卷积结构，因此，为了让模型能够利用序列的**顺序信息**，我们必须在序列中注入一些关于**相对位置**或**绝对位置**的信息。为此，我们在编码器和解码器堆栈的底部，将**位置编码**加入到输入的词嵌入中。位置编码与词嵌入具有相同的维度$d_{model}$，这样二者可以直接相加。位置编码有多种实现方式，可以是学习得到的或固定的。在本研究中，我们使用**不同频率**的**正余弦函数**来构造位置编码。

**公式：**
$$
\begin{aligned}
PE_{(pos,2i)} &= \sin(pos/10000^{2i/d_{\mathrm{model}}}) \\
PE_{(pos,2i+1)} &= \cos(pos/10000^{2i/d_{\mathrm{model}}})
\end{aligned}
$$
其中：

| 符号               | 含义                                    |
| ------------------ | --------------------------------------- |
| pos                | 表示序列中第几个位置（position index）  |
| i                  | 表示 embedding 的维度索引               |
| $d_{\text{model}}$ | 模型的 embedding 总维度，比如 512       |
| 2i, 2i+1           | 表示偶数维和奇数维分别用 sin / cos 编码 |

##### **直观例子：**

​	以$d_{model}$=8为例子

| 维度 | 函数形式                | 波长（变化速度） |
| ---- | ----------------------- | ---------------- |
| 0    | sin(pos / 10000^0)      | 变化快           |
| 1    | cos(pos / 10000^0)      | 变化快           |
| 2    | sin(pos / 10000^{0.25}) | 变化中           |
| 3    | cos(pos / 10000^{0.25}) | 变化中           |

​	公式换个形式可以表达成：
$$
PE(pos)=[\sin(\frac{pos}{K_{0}}),\cos(\frac{pos}{K_{0}}),\sin(\frac{pos}{K_{1}}),\cos(\frac{pos}{K_{1}}),\ldots]
$$
其中：每个维度都对应一个不同的周期（不同频率），$K_i = 10000^{2i / d_{model}}$，$K$与$pos$无关，仅取决于模型本身的$d_{model}$和维度索引$i$。

纵向观察下面这个表格就是三角函数的变化过程。

|         | $dim_0$   | $dim_1$   | $dim_2$  | $dim_3$  | $dim_4$  | $dim_5$  | $dim_6$  | $dim_7$  |
| ------- | --------- | --------- | -------- | -------- | -------- | -------- | -------- | -------- |
| $pos_0$ | 0.000000  | 1.000000  | 0.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| $pos_1$ | 0.841471  | 0.540302  | 0.009999 | 0.999950 | 0.000100 | 1.000000 | 0.000001 | 1.000000 |
| $pos_2$ | 0.909297  | -0.416147 | 0.019998 | 0.999800 | 0.000200 | 1.000000 | 0.000003 | 1.000000 |
| $pos_3$ | 0.141120  | -0.989992 | 0.029997 | 0.999550 | 0.000300 | 1.000000 | 0.000004 | 1.000000 |
| $pos_4$ | -0.756802 | -0.653644 | 0.039996 | 0.999200 | 0.000400 | 1.000000 | 0.000006 | 1.000000 |
| $pos_5$ | -0.958924 | 0.283662  | 0.049995 | 0.998750 | 0.000500 | 1.000000 | 0.000007 | 1.000000 |
| $pos_6$ | -0.279415 | 0.960170  | 0.059994 | 0.998200 | 0.000600 | 1.000000 | 0.000009 | 1.000000 |
| $pos_7$ | 0.657989  | 0.753902  | 0.069993 | 0.997550 | 0.000700 | 1.000000 | 0.000010 | 1.000000 |
| $pos_8$ | 0.989358  | -0.145500 | 0.079992 | 0.996800 | 0.000800 | 1.000000 | 0.000011 | 1.000000 |
| $pos_9$ | 0.412118  | -0.911130 | 0.089991 | 0.995950 | 0.000900 | 1.000000 | 0.000013 | 1.000000 |

​	这组公式表示**绝对位置**，同时也能隐式地表示**相对位置**。将每组 $i$ 想象成一个在坐标系上的**圆**$(\sin(\frac{pos}{K_{i}}),\cos(\frac{pos}{K_{i}}))$，pos不断增加相当于这个圆的角度在以一个固定值增大旋转，所以$PE(pos+\Delta)=\mathcal{R}(\Delta)\:PE(pos)$中的$\mathcal{R}(\Delta)\:$仅取决于相对位置差，同时会有256组 $i$ 的数据，所以不用担心周期性问题。注意，这只是辅助理解方式，transformer并**不会显式**计算这个旋转矩阵，它依赖注意力和线性层隐式捕获。理论上网络可以学习通过线性映射捕捉相对位置信息，但实际上效果如何并不好说，可能不稳定。后续模型在针对**极长序列**或者更加关注**相对位置**的时候，会考虑使用其他的位置编码方式。

##### 优势：

1. 无需学习，效率高，实现简单。
2. 绝对位置表示可能也能感受到相对位置信息。
3. 不同的 $dim$ 之间变化速率并不相同（$K_i = 10000^{2i / d_{model}}$），相当于在**多尺度**上感受信息，慢变化相当于在长距离上感受变化，快变化相当于在更关注局部信息。
4. 所有位置编码向量的范数相同（恒定长度），仅仅是方向在高维空间中旋转，大大提高**模型稳定性**，不会出现“位置越大向量越大”等问题。

##### 缺点：

1. 与输入 embedding 直接相加**可能限制表达能力**，可能不稳定。
2. 对长序列泛化能力有限。

##### **为什么在解码器增加位置编码**

​	To this end, we add **positional encodings** to **the input embeddings** at the bottoms of the encoder and decoder stacks. 

​	为此，我们在编码器和解码器堆栈的底部，将**位置编码**加入到输入的词嵌入中。

​	解码器同样无法在token embedding直接得到位置信息。

##### 位置编码信息如何加入

​	我们得到了通过查表得到embedding$x_{pos}$，并通过计算得到位置编码矩阵$PE$，这两个矩阵形状一致，**直接简单相加**$z=x_{pos}+PE$得到编码器的最初输入。当然，可以使用Gated Sum（$z=\alpha\cdot x+(1-\alpha)\cdot PE$，其中$\alpha$可学习）。在后续的self-attention计算中，$\text{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$,$Q，K，V$都是通过 $z$线性映射得到的，内部隐式包含了位置信息与关系。简单的相加看似破坏了$x_{pos}$和$PE$的直观含义，但是Transformer会通过**注意力机制**和**线性变换**自动**重构**并利用这些信息。（不过确实**可能限制了表达能力**）

​	例如，$\mathrm{score}_{ij}=Q_i\cdot K_j^T$，

​	其中：$Q_i=(x_i+PE_i)W_Q,\quad K_j=(x_j+PE_j)W_K$，展开得到：$Q_iK_j^T=(x_iW_Q)(x_jW_K)^T+(x_iW_Q)(PE_jW_K)^T+(PE_iW_Q)(x_jW_K)^T+(PE_iW_Q)(PE_jW_K)^T$，里面包含了**词语语义相似度**、**位置之间关系和位置**与**语义之间的交互关系**等信息。



#### scaled dot-product attention

​	We call our particular attention "Scaled Dot-Product Attention". The input consists of queries and keys of dimension  $d_k$ , and values of dimension  $d_v$ . We compute the dot products of the query with all keys, divide each by  $\sqrt{d_k}$ , and apply a softmax function to obtain the weights on the values.

​	我们称我们的特殊注意力为“缩放点积注意力”。输入由维度为$d_k$的**查询（query）**和**键（key）**，以及维度为$d_v$的**值（value）**组成。我们计算查询与所有键的点积，分别除以$\sqrt{d_k}$，然后应用softmax函数以获得数值的权重。

​	In practice, we compute the attention function on a set of queries **simultaneously**, packed together into a matrix  $Q$ . The keys and values are also packed together into matrices  $K$  and  $V$ . We compute the matrix of outputs as:

​	在实践中，我们**同时**计算一组查询的注意力函数，将它们打包成矩阵$Q$。键和值也分别打包成矩阵$K$和$V$。输出矩阵的计算公式为：

$$
\operatorname {A t t e n t i o n} (Q, K, V) = \operatorname {s o f t m a x} \left(\frac {Q K ^ {T}}{\sqrt {d _ {k}}}\right) V
$$
​	The two most commonly used attention functions are additive attention , and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for **the scaling factor** of  $\frac{1}{\sqrt{d_k}}$ . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

​	最常用的两种注意力函数是加性注意力与点积（乘法）注意力。点积注意力与本文算法完全一致，仅区别于**缩放因子 $\frac{1}{\sqrt{d_k}}$** 的使用。加性注意力通过含单隐藏层的前馈网络计算兼容性函数。虽然二者理论复杂度相近，但点积注意力在实践中因可采用高度优化的矩阵乘法代码而**显著提升运算速度**并减少内存占用。

​	While for small values of  $d_{k}$  the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of  $d_{k}$ . We suspect that for large values of  $d_{k}$ , the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by  $\frac{1}{\sqrt{d_k}}$ .

​	当$d_{k}$取值较小时，两种注意力机制表现相近；但在$d_{k}$较大时，加性注意力要优于未缩放的点积注意力。我们认为当$d_{k}$较大时，点积的幅值会显著增大，从而将softmax函数推入梯度极小的饱和区间。为抵消这一影响，我们将点积结果缩放$\frac{1}{\sqrt{d_k}}$倍。

##### 解释：

​	Scaled的意思就是除以 $\sqrt{d_k}$，目的是**防止点积结果过大**，softmax梯度过小、训练不稳定。而dot-product就是点积的意思。前面也提到了，attention机制有很多的实现方式，如加性注意力（$Q$、$K$相加）。相比于一般的点积注意力，transformer的特点就是多除一个$\sqrt{d_k}$。注意这里的 $d_k\ne d_{model}$，这里就涉及了**Multi-Head Attention**机制

​	

<img src="/img/posts/transformer/attention.png" alt="attention" style="width: 38%; min-width: 240px; max-width: 100%;" />

​	



#### multi-head attention

​	Instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries,we found it beneficial to **linearly project** the queries, keys and values $h$ times with **different**, **learned** linear projections to $d_{k}$,$d_{k}$ and **$d_{v}$** dimensions,respectively. On each of these projected versions of queries, keys and values, we then perform the attention function **in parallel**, yielding $d_v$-dimensional output values. These are **concatenated** and once again projected, resulting in the final values, as depicted. Multi-head attention allows the model to jointly attend to information from **different representation subspaces** at **different positions**. With a single attention head, averaging inhibits this.

​	与其仅使用一个键、值和查询均为 $d_{model}$维的注意力函数，我们发现将查询、键和值分别通过 $h$ 个**不同**的、**可学习**的**线性映射**投影到 $d_{k}$,$d_{k}$ 和 **$d_{v}$** 维空间中是更有效的。在每组投影后的查询、键和值上，我们分别**并行**执行注意力计算，从而得到 **$d_v$** 维的输出结果。然后将这些并行输出**拼接（concatenate）**在一起，再经过一次线性变换（投影），得到最终的输出结果。多头注意力机制使模型能够在不同位置上**同时**关注来自**不同表示子空间**的信息。而如果只有单个注意力头，这种能力会因为平均化而被抑制。

​	Multi-head attention allows the model to **jointly** attend to information from **different representation subspaces** at **different positions**. With a single attention head, **averaging** **inhibits** this.

​	多头注意力机制使模型能够在**不同的位置**上**同时**关注来自**不同表示子空间**的信息。如果只有单个注意力头，那么这种能力会因为**平均化而受到抑制**。
$$
\begin{aligned}
\mathrm{MultiHead}(Q,K,V) &= \mathrm{Concat}(\mathrm{head}_{1},\dots,\mathrm{head}_{h})W^{O} \\
\mathrm{where\ head}_{i} &= \mathrm{Attention}(QW_{i}^{Q},KW_{i}^{K},VW_{i}^{V})
\end{aligned}
$$
​	这里每个 $head_i$ 的输出维度是 $d_v$，拼接后得到维度$h \times d_v$，$W^O \in \mathbb{R}^{(h d_v) \times d_{model}}$是一个**可学习**的线性变换矩阵,这一层只是将多头输出重新**映射回**模型维度 $d_{model}$，以便和后续层对齐。

![multi_head_attention_block](/img/posts/transformer/multi_head_attention_block.png)

​	这里的**Add**就是残差连接（Residual Connection，和ResNet的思想一脉相承），即 $x\ +\ MHA(x)$。让子层学习的是输入的增量变化量，而**不是完全重构特征**，有助于transformer的**稳定训练**，让输入$x$至少保留原始信息$x$避免丢失。

​	而**Norm**就是层归一化（LayerNorm），标准化输出保证分布**保持均匀**，还是稳定训练的意义。
$$
\mathrm{LayerNorm}(\mathbf{h})=\frac{\mathbf{h}-\mu}{\sigma}\odot\gamma+\beta
$$
其中：

- $\mathbf{h} \in \mathbb{R}^{d_{\text{model}}}$：特征向量
- $\mu = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} h_i$：均值
- $\sigma = \sqrt{\frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} (h_i - \mu)^2}$：标准差
- $\gamma \in \mathbb{R}^{d_{\text{model}}}$：每个特征的**可学习**缩放参数
- $\beta \in \mathbb{R}^{d_{\text{model}}}$：每个特征的**可学习**偏置

##### Q、K、V都是怎么得到的

​	首先是会有一个输入 $X$,而在编码器和解码器中都有multi-head attention机制，结构基本相似，但是发挥的作用和输入输出是不一样的。我们需要对 $X$ 做三个**不同**的**线性变换**。
$$
\begin{aligned}
Q &= XW^Q \\
K &= XW^K \\
V &= XW^V
\end{aligned}
$$
其中：$W^Q,W^K\in\mathbb{R}^{d_{\mathrm{model}}\times d_k}$，且$W^V\in\mathbb{R}^{d_{\mathrm{model}}\times d_v}$，$W^Q,W^K,W^V$都是**可学习**矩阵，这里是一种**降维**操作。

##### Q、K、V的解释

​	从语义粗浅的解释来看，Q，K，V 是同一输入经过不同线性变换得到的三个向量空间，借鉴了**信息检索**中的“query-key-value”模型，但这只是个比喻。这样设计的目的并非语义划分，而是**结构解耦**，通过线性映射分解增强模型的**灵活性**。

| 名称          | 直观含义             | 类比     |
| ------------- | -------------------- | -------- |
| **Q (Query)** | “我要找什么信息？”   | 提问的人 |
| **K (Key)**   | “我能提供什么信息？” | 资料标签 |
| **V (Value)** | “具体的信息内容”     | 资料内容 |

##### Q、K、V的形状	

​	关于$W_Q$,$W_K$和$W_V$的**形状**是**可以**是不一样的，从解释性来看Q，K形成匹配空间，决定”谁和谁相关“，表示**相似度**，每个头关注关系的复杂程度；而V形成信息空间，存放被**提取**的信息内容，决定输出的**信息量**。虽然在transformer中，$h$, $d_{model}$, $W_Q$, $W_K$ 和 $W_V$都是**超参数**，设置的具体大小分别为：$d_{model}=512$，$h=8$，$d_k=d_v=\frac{d_{model}}h=\frac{512}8=64$。

##### multi-head attention的作用

​	**多角度**捕捉信息，不同头关注不同类型的依赖（因为存在随机的初始化、位置编码、dropout等众多原因），增强表达能力。

##### 不同位置的应用（applications）

The Transformer uses multi-head attention in three different ways:

​	In **"encoder-decoder attention"** layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models. 

​	The **encoder** contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.

​	 Similarly, self-attention layers in the **decoder** allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to **prevent leftward information flow** in the decoder to preserve the **auto-regressive property**. We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections. 
Transformer 在三种不同的方式中使用多头注意力：

1. **“编码器-解码器注意力”层**中，queries来自前一层解码器，memory keys 和 values来自编码器的输出。（由于这里的Q、K、V来源不同，所以称为**cross-attention**）这使得解码器中的每个位置都可以关注输入序列中的所有位置。这模仿了序列到序列模型中典型的编码器-解码器注意力机制。
2. **编码器**包含自注意力层。在自注意力层中，所有的键、值和查询都来自同一来源（即**self-attention**），这里是编码器前一层的输出。编码器中的每个位置都可以关注编码器前一层中的所有位置。（即可以关注**所有token**，**没有mask机制**）
3. **解码器**中的自注意力层（同样也是**self-attention**）允许解码器的每个位置关注解码器中直到该位置（包括该位置）的所有位置。为了保持**自回归**性质，我们需要**阻止信息向左流动**（即**防止看到未来token**）。我们在缩放点积注意力中实现这一点，通过**masking** softmax 输入中对应非法连接的所有值（**设为 $-\infty$**）。

##### mask机制

​	出现在**解码器**的注意力机制。公式如下：
$$
\text{masked\_scores}_{ij} =
\begin{cases}
\dfrac{(QK^T)_{ij}}{\sqrt{d_k}}, & j \le i \\
{-\infty}, & j > i
\end{cases}
$$
​	为什么把对应非法连接的所有值设置为 $-\infty$？

​	因为在**softmax层**公式如下：
$$
\operatorname{softmax}(z)_j=\frac{e^{z_j}}{\sum_ke^{z_k}}
$$
​	很自然$e^{-\infty}=0$，这个信息就被完全屏蔽了。



#### encoder

​	The encoder is composed of a stack of N = 6 **identical** layers. Each layer has two sub-layers. The first is a **multi-head self-attention mechanism**, and the second is a simple, position-wise **fully connected feed-forward network**. We employ a **residual connection** around each of the two sub-layers, followed by **layer normalization** .That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{model}$ = 512.

​	编码器由一堆 N=6 个**相同**的层组成。每一层都包含两个子层：第一个是**多头自注意力**机制，第二个是一个简单的、按位置独立的**全连接前馈网络**。我们在每个子层外都采用了**残差连接**，并在其后进行**层归一化**。也就是说，每个子层的输出形式为 LayerNorm(x + Sublayer(x))，其中 Sublayer(x) 表示该子层自身实现的函数。为了便于这些残差连接，模型中所有的子层以及嵌入层的输出维度都被设定为$d_{model}$。



#### decoder

​	The decoder is also composed of a stack of N=6 **identical** layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. **Similar to the encoder**, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, **combined** with the fact that **the output embeddings are offset by one position**, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

​	解码器同样由 N=6 层**相同**的堆叠层组成。除了编码器层中的两个子层之外，解码器增加了第三个子层，对编码器堆叠的输出执行多头注意力。**与编码器类似**，我们在每个子层周围使用残差连接，然后进行层归一化。我们还对解码器堆叠中的自注意力子层进行了修改，以防止某个位置关注后续位置（就是**mask**机制）。这种掩码（masking），**结合** **输出嵌入向右偏移一个位置**的事实（就是**shifted right**），保证了位置 i 的预测只能依赖于位置小于 i 的已知输出（就是说防止看到未来的信息**不只是**mask机制起到作用）。

![shift_right](/img/posts/transformer/shift_right.png)

##### shift-right embedding

​	注意，在训练阶段和推理阶段shift-right embedding是不同的。在**训练阶段**，知道目标序列 **Y** ，所以输入就是特殊符号\<start>（也是一个**可训练**embedding + 位置编码）和下一个token前的所有目标序列的token。而在**推理阶段**，输入就是前面解码器输出的所有token。



#### feed-forward networks

​	In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

​	除了注意力子层之外，我们的编码器和解码器中的每一层都包含一个**全连接前馈网络**，该网络对每个位置分别且相同地应用（即与位置无关，不考虑上下文连接）。该网络由两个线性变换组成，中间夹着一个 ReLU 激活函数：
$$
\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$
​	While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is $d_{\text{model}} = 512$, and the inner-layer has dimensionality $d_\text{ff} = 2048$.

​	虽然线性变换在不同位置上相同，但不同层使用的参数不同（Transformer 有 **L 层** Encoder/Decoder，每层都有自己的 FFN）。另一种描述方式是，这相当于两个卷积核大小为 1 的卷积操作（意思就是与位置无关，不考虑上下文连接）。输入和输出的维度为 $d_{\text{model}} = 512$，中间层的维度为 $d_\text{ff} = 2048$（即FFN 第一层线性升维到 $d_\text{ff} = 2048$ → ReLU → 第二层线性降回 512）。

##### FFN维度大小具体变化

**1.输入 → 第一层线性变换**

$X W_1 + b_1$

- 输入矩阵 $n \times 512$，其中n为序列长度
- 权重矩阵 $W_1: 512 \times 2048$
- 结果矩阵维度：$n \times 512 \cdot 512 \times 2048 = n \times 2048$
- 加上 bias 后，每行加 $b_1$ → 仍然是 $n \times 2048$

第一层线性：**升维** 512 → 2048

**2.ReLU 激活**
$\text{H} = \text{ReLU}(X W_1 + b_1)$

- 维度不变：$n \times 2048$
- 激活只改变数值，不改变矩阵形状

**3.第二层线性变换**
  $\text{FFN}(X) = H W_2 + b_2$

- $H: n \times 2048$
- $W_2: 2048 \times 512$
- 输出矩阵：$n \times 2048 \cdot 2048 \times 512 = n \times 512$

第二层线性：**降维** 2048 → 512



#### 最后的Linear&softmax

​	在所有Decoder之后，我们得到最终矩阵:$H\in\mathbb{R}^{n\times d_{\mathrm{model}}}$，但是这个矩阵还不是**词概率分布**，只是每个 token 的**向量表示**。最后使用一个全连接层（Linear）将 $d_\text{model}$ 映射到 **词表大小 |V|**：$Z=HW_o+b_o$。其中：$W_o\in\mathbb{R}^{d_{\mathrm{model}}\times|V|}$。这一步就是将**连续的向量空间**转换成**离散的token空间**。

​	在最后进行softmax。如果是**训练**阶段，即用于计算**交叉熵损失**；如果是**推理**阶段，用于选择贪婪搜索或者采样或者Beam Search。

##### 贪婪搜索 (Greedy Search)

​	最简单粗暴的办法。永远选择当前**概率最高**的那个Token。这种方法只看眼前最好的，不考虑全局。它可能会导致重复、无聊或陷入局部最优。

##### 束搜索 (Beam Search)

​	束搜索会同时保留 $k$ 个（$k$ 就是“束宽”）当前最优的“候选句子”。在每一步，它都会为这 $k$ 个候选句子分别预测下一个Token，然后从所有（$k \times$ 词汇表大小）的可能性中，选出 $k$ 个**累积概率最高**的新候选句子。这种方法考虑了“全局最优”，生成的句子通常更流畅、更合理，在机器翻译等任务中非常常用，但是计算量更大。


##### 采样 (Sampling) 

​	在像聊天机器人或创意写作这样的任务中，我们不希望答案总是“最正确”但“最无聊”的。我们需要引入一些随机性。

**温度 (Temperature)：** 在 $softmax$ 之前调整Logits（就是最后一层网络输出的**原始分数**），公式如下：
$$
P(y=i)=\frac{e^{(z_i/T)}}{\sum_{j=1}^Ve^{(z_j/T)}}
$$

- 高温 (T > 1)： 概率分布更“平坦”，使得低概率的词（更有创意的词）也有机会被选中。
- 低温 (T < 1)： 概率分布更“尖锐”，使得高概率的词更容易被选中。

**Top-k 采样：** 只在概率排名前 $k$ 个的Token中进行随机采样。这排除了那些概率极低的“垃圾词”。

**Top-p (Nucleus) 采样：** 一种更智能的方法。它从概率最高的词开始，不断累加概率，直到总和达到一个阈值 $p$（例如 $p=0.95$）。然后模型只在这批（累积概率达到95%）的词中进行采样。这是一个动态的列表，有时可能只有3个词，有时可能有30个词。

##### 输出Token的长度如何确定

Transformer模型是自回归的，输出token的过程就像一个循环：

[输入] -> 预测Token A -> [输入 + Token A] -> 预测Token B -> [输入 + Token A + Token B] -> ...

停止条件：

1. [EOS] 停止符 (End-of-Sequence Token)

​	在模型的词汇表中，有一个（或多个）特殊的Token，叫做 **[EOS]**（End-of-Sequence）或类似的标记。模型在训练时，会**学习**到“当一句话说完了”的时候，就应该输出 [EOS] 这个Token。因此，在生成时，当模型在某一步最终选择的Token是 [EOS] 时，生成过程就自动停止了。

2. 达到最大长度 (Max Length)

​	模型有一个**最大长度限制**（例如 `max_length=2048`）。如果模型在生成了2048个Token之后，仍然没有生成 [EOS] Token（比如模型陷入了重复、或者在处理一个极其复杂的主题），系统会“强行”切断生成过程，防止它无限循环下去，耗尽资源。



### 四、性能分析

**(这部分没有学习整理比较细，留给之后再慢慢看看)**

| Layer Type              | Complexity per Layer | Sequential Operations | Maximum Path Length |
|-------------------------|----------------------|-----------------------|---------------------|
| Self-Attention          | $O(n^2 \cdot d)$      | $O(1)$                | $O(1)$              |
| Recurrent               | $O(n \cdot d^2)$      | $O(n)$                | $O(n)$              |
| Convolutional           | $O(k \cdot n \cdot d^2)$ | $O(1)$              | $O(\log_k(n))$      |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$          | $O(n/r)$            |

​	In this section, we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations $(x_1, \dots, x_n)$ to another sequence of equal length $(z_1, \dots, z_n)$, with $x_i, z_i \in \mathbb{R}^d$, such as in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

​	在本节中，我们将自注意力层与循环层和卷积层的各个方面进行比较。这两种传统网络层通常用于将一个可变长度的符号表示序列 $(x_1, \dots, x_n)$ 映射到另一个等长的序列 $(z_1, \dots, z_n)$，其中 $x_i, z_i \in \mathbb{R}^d$。这在典型的序列转导编码器或解码器的隐藏层中很常见。我们之所以采用自注意力机制，主要基于三个期望的特性。

​	One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

​	第一个特性是每层的总计算复杂度。第二个是可以并行的计算量，这通常由所需的最小顺序操作数量来衡量。第三个是网络中长距离依赖关系之间的路径长度。在许多序列转导任务中，学习长距离依赖是一个关键挑战。影响学习这种依赖能力的一个关键因素是前向和后向信号在网络中必须遍历的路径长度。输入和输出序列中任意位置组合之间的路径越短，学习长距离依赖就越容易。因此，我们还比较了由不同层类型组成的网络中，任意两个输入和输出位置之间的最大路径长度。

​	As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires $O(n)$ sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translation, such as word-piece and byte-pair representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in the input sequence centered around the respective output position. This would increase the maximum path length to $O(n/r)$. We plan to investigate this approach further in future work.

​	如表1所示，一个自注意力层用固定的顺序执行操作次数连接所有位置，而一个循环层需要 $O(n)$ 的顺序操作。在计算复杂度方面，当序列长度 $n$ 小于表示维度 $d$ 时，自注意力层比循环层更快。在机器翻译的最新模型中，对于句子表示（例如 word-piece 和 byte-pair 表示），通常都满足这个条件。为了提高处理极长序列任务的计算性能，可以将自注意力机制限制为只考虑输入序列中以相应输出位置为中心的、大小为 $r$ 的邻域。这会将最大路径长度增加到 $O(n/r)$。我们计划在未来的工作中进一步研究这种方法。

​	A single convolutional layer with kernel width $k < n$ does not connect all pairs of input and output positions. Doing so requires a stack of $O(n/k)$ convolutional layers in the case of contiguous kernels, or $O(\log_k(n))$ in the case of dilated convolutions, increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of k. Separable convolutions, however, decrease the complexity considerably, to $O(k \cdot n \cdot d + n \cdot d^2)$. Even with $k=n$, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

​	一个核宽度 $k < n$ 的单层卷积层不会连接所有输入和输出位置对。要做到这一点，在连续卷积核的情况下需要一个包含 $O(n/k)$ 个卷积层的堆栈，或者在扩张卷积（dilated convolutions）的情况下需要 $O(\log_k n)$ 个卷积层，这增加了网络中任意两个位置之间的最长路径长度。卷积层通常比循环层计算成本更高，大约是 $k$ 倍。然而，可分离卷积（separable convolutions）将复杂度显著降低到 $O(k \cdot n \cdot d + n \cdot d^2)$。尽管如此，即使 $k=n$，可分离卷积的复杂度也与自注意力层和我们模型中采用的逐点前馈层（point-wise feed-forward layer）的组合相当。

​	As a side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.	

​	作为一个额外的好处，自注意力机制可能会产生更具可解释性的模型。我们检查了模型中的注意力分布，并在附录中展示和讨论了一些例子。不仅单个注意力头（attention heads）清晰地学会了执行不同的任务，许多注意力头似乎还表现出与句子的句法和语义结构相关的行为。