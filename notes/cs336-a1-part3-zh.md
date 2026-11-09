# CS336 作业 1（basics）讲义译文 · 第 3 部分：Transformer 语言模型架构

> **原文**：`cs336_assignment1_basics.pdf`，版本 26.0.3，Spring 2026，第 13–28 页（§3 全文，至 `transformer_accounting` 结束）
> **性质**：个人学习用非官方译文，逐段对照原文翻译；术语优先保留英文原词。原文为准。
> **接续**：上接 `cs336-a1-part1-2-zh.md`（§1–§2）。
> **注意**：本文件不属于作业提交物。提交前请把 `notes/` 加入 `make_submission.sh` 的排除列表（如 `-x 'notes/*'`）。

---

## 3 Transformer 语言模型架构

语言模型以一批整数 token ID 序列作为输入（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`），并返回词表上的（批处理的）归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch Tensor），其中对每个输入 token，所预测的分布都是关于**下一个词**的。训练语言模型时，我们用这些下一词预测来计算真实下一词与预测下一词之间的交叉熵损失。推理阶段从语言模型生成文本时，我们取**最后一个时间步**（即序列中的最后一项）所预测的下一词分布来生成序列中的下一个 token（例如取概率最大的 token、或从分布中采样等），把生成的 token 追加到输入序列上，然后重复。

在本部分作业中，你将**从零**构建这个 Transformer 语言模型。我们会先给出模型的高层描述，再逐步细化各个组件。

### 3.1 Transformer LM

给定一个 token ID 序列，Transformer 语言模型先用输入嵌入把 token ID 转换为稠密向量，把嵌入后的 token 依次送入 `num_layers` 个 Transformer 块，然后施加一个可学习的线性投影（"输出嵌入"或"LM head"）来产生预测的下一 token logits。示意图见图 1。

> **图 1：我们的 Transformer 语言模型概览。**
> 流程：Inputs → Token Embedding →（`num_layers` 个 Transformer Block）→ Norm → Linear（Output Embedding）→ Softmax → Output Probabilities。

#### Token 嵌入

在最开始的一步，Transformer 把（批处理的）token ID 序列嵌入为一系列携带 token 身份信息的向量（图 1 中的红色方块）。

更具体地说，给定一个 token ID 序列，Transformer 语言模型用 token 嵌入层产出一系列向量。每个嵌入层接受形状为 `(batch_size, sequence_length)` 的整数张量，产出形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

#### Pre-norm Transformer 块

嵌入之后，激活值会经过若干个结构相同的神经网络层。一个标准的 decoder-only Transformer 语言模型由 `num_layers` 个相同的层（通常称为 Transformer"块"）组成。每个 Transformer 块接受形状为 `(batch_size, sequence_length, d_model)` 的输入，返回相同形状的输出。每个块通过自注意力在整个序列上聚合信息，并通过前馈层对其进行非线性变换。

经过 `num_layers` 个 Transformer 块之后，我们取最终的激活值，把它变为词表上的分布。

我们将实现"pre-norm"Transformer 块（详见 §3.4），它额外要求在最后一个 Transformer 块之后再做一次层归一化，以确保其输出被适当缩放。

完成这次归一化之后，我们会用标准的可学习线性变换把 Transformer 块的输出转换为预测的下一 token logits（例如见 A. Radford et al. [7] 的式 2）。

> **图 2：一个 pre-norm Transformer 块。**
> 输入张量形状 `(batch_size, seq_len, d_model)` → Norm → Causal Multi-Head Self-Attention w/ RoPE → Add（残差）→ Norm → Position-Wise Feed-Forward → Add（残差）→ 输出张量形状 `(batch_size, seq_len, d_model)`。

### 3.2 附注：批处理、Einsum 与高效计算

在整个 Transformer 中，我们会把同一套计算施加到许多"类批处理"的输入上。以下是几个例子：

- **批次中的元素**：我们对每个 batch 元素施加相同的 Transformer 前向运算。
- **序列长度**：像 RMSNorm 与前馈网络这类"逐位置（position-wise）"运算，对序列的每个位置都完全一致地作用。
- **注意力头**：在"多头"注意力运算中，注意力操作在各注意力头上批处理执行。

拥有一套既能充分利用 GPU、又易于阅读和理解的表达方式来执行这类运算，是很有用的。许多 PyTorch 运算可以在张量开头接受多余的"类批处理"维度，并在这些维度上高效地重复／广播运算。

例如，假设我们在做一个逐位置、批处理的运算。我们有一个形状为 `(batch_size, sequence_length, d_model)` 的"数据张量" $D$，并希望与形状为 `(d_model, d_model)` 的矩阵 $A$ 做批处理的向量-矩阵乘法。此时 `D @ A` 会做一次批处理矩阵乘法，这是 PyTorch 中的一个高效原语，其中 `(batch_size, sequence_length)` 这两个维度被批处理掉。

正因如此，比较好的做法是假定你的函数可能被传入额外的类批处理维度，并把这些维度保持在 PyTorch 形状的**开头**。为了把张量组织成能以这种方式批处理的形式，可能需要经过许多步 `view`、`reshape` 与 `transpose`。这有点麻烦，而且常常让人难以看懂代码在做什么、你的张量究竟是什么形状。

更舒适的选择是在 `torch.einsum` 中使用 einsum 记号，或者干脆使用与框架无关的库，如 `einops` 或 `einx`。两个关键运算是 `einsum`（可对输入张量的任意维度做张量收缩）与 `rearrange`（可重排、拼接与拆分任意维度）。事实证明，机器学习中几乎所有运算都是"维度搬移 + 张量收缩"的某种组合，偶尔再加上一个（通常是逐点的）非线性函数。这意味着，使用 einsum 记号后，你的大量代码可以更易读、更灵活。

我们强烈建议你学习并在课程中使用 einsum 记号。此前没接触过 einsum 记号的同学应使用 `einops`（文档见此处）；已经熟悉 `einops` 的同学则应学习更通用的 `einx`（见此处）⁴。这两个包都已安装在提供给你们的运行环境中。

下面给出几个使用 einsum 记号的例子。它们是对 `einops` 文档的补充，你应当先读那份文档。

> **示例（einstein_example1）：用 `einops.einsum` 做批处理矩阵乘法**
>
> ```python
> import torch
> from einops import rearrange, einsum
>
> ## 基本实现
> Y = D @ A.T
> # 很难看出输入与输出的形状，也看不出它们各自的含义。
> # D 和 A 可以是什么形状？这些情况里有没有哪种会出人意料？
>
> ## Einsum 自带文档性，且更稳健
> #                          D                A     ->          Y
> Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
>
> ## 或者写成批处理版本：D 可以带任意多个前置维度，而 A 受约束。
> Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
> ```

> **示例（einstein_example2）：用 `einops.rearrange` 做广播运算**
>
> 我们有一批图像，想对每张图基于某个缩放因子生成 10 个变暗版本：
>
> ```python
> images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
> dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
>
> ## 重塑后相乘
> dim_value = rearrange(dim_by,    "dim_value              -> 1 dim_value 1 1 1")
> images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
> dimmed_images = images_rearr * dim_value
>
> ## 或者一步到位：
> dimmed_images = einsum(
>     images, dim_by,
>     "batch height width channel, dim_value -> batch dim_value height width channel"
> )
> ```

> **示例（einstein_example3）：用 `einops.rearrange` 做像素混合**
>
> 假设我们有一批形状为 `(batch, height, width, channel)` 的图像，希望在图像的全部像素上做一次线性变换，但该变换对每个通道独立进行。该线性变换由形状为 `(height * width, height * width)` 的矩阵 $B$ 表示。
>
> ```python
> channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
> B = torch.randn(32*32, 32*32)
>
> ## 重排图像张量，以便在所有像素上混合
> channels_last_flat = channels_last.view(
>     -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
> )
> channels_first_flat = channels_last_flat.transpose(1, 2)
> channels_first_flat_transformed = channels_first_flat @ B.T
> channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
> channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
> ```
>
> 改用 einops 则可以写成：
>
> ```python
> height = width = 32
>
> ## rearrange 取代了笨重的 torch view + transpose
> channels_first = rearrange(
>     channels_last,
>     "batch height width channel -> batch channel (height width)"
> )
> channels_first_transformed = einsum(
>     channels_first, B,
>     "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
> )
> channels_last_transformed = rearrange(
>     channels_first_transformed,
>     "batch channel (height width) -> batch height width channel",
>     height=height, width=width
> )
> ```
>
> 或者，如果你够大胆，可以用 `einx.dot`（`einops.einsum` 的 einx 对应物）一步到位：
>
> ```python
> height = width = 32
> channels_last_transformed = einx.dot(
>     "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
>     "-> batch row_out col_out channel",
>     channels_last, B,
>     col_in=width, col_out=width
> )
> ```
>
> 上面第一种实现可以靠在前前后后写注释来标明输入输出形状，但那样很笨重、也容易出 bug。用 einsum 记号时，**文档即实现**！

Einsum 记号能处理任意输入批处理维度，还有一个关键好处：自带文档性。在使用 einsum 记号的代码里，输入与输出张量的相关形状要清楚得多。对于其余张量，你可以考虑使用 Tensor 类型标注，例如借助 `jaxtyping` 库（并非 JAX 专属）。

关于使用 einsum 记号带来的性能影响，我们会在作业 2 中进一步讨论；现在你只需知道：它们几乎总是优于替代写法！

#### 3.2.1 数学记号与内存排布

许多机器学习论文在记号中使用**行向量**，这样的表示与 NumPy、PyTorch 默认的**行主序**（row-major）内存排布十分契合。使用行向量时，线性变换写作

$$y = xW^\top, \tag{1}$$

其中 $W \in \mathbb{R}^{d_{out} \times d_{in}}$ 为行主序，$x \in \mathbb{R}^{1 \times d_{in}}$ 为行向量。注意这样写可以很方便地对输入做批处理：只需把 $x$ 的最外层维度加大，即把向量输入 $x$ 换成矩阵输入 $X \in \mathbb{R}^{batch \times d_{in}}$。

在线性代数中，更常见的是使用**列向量**，此时线性变换写作

$$y = Wx, \tag{2}$$

其中 $W \in \mathbb{R}^{d_{out} \times d_{in}}$ 为行主序，$x \in \mathbb{R}^{d_{in}}$ 为列向量。在这种写法下要对输入做批处理，批次维度就必须放在 $x$ 的**最后**，即把 $x$ 换成矩阵 $\tilde{X} \in \mathbb{R}^{d_{in} \times batch}$。

本作业的数学记号将主要使用**列向量**，因为数学上通常这样写。你应记住：如果你想用普通的矩阵乘法记号，就得像第 1 式的行向量约定那样对矩阵做转置后再施加，因为 PyTorch 采用行主序内存排布。如果你用 einsum 完成线性代数运算，只要各轴标注正确，这就不成问题。顺带一提：Matlab、Julia 与 Fortran 等其他语言／线性代数包都采用**列主序**内存排布，也就是批次维度放在最后，而 Python 及相关包沿用了 C 标准的行主序。

### 3.3 基础构件：Linear 与 Embedding 模块

#### 3.3.1 参数初始化

要有效地训练神经网络，往往需要谨慎地初始化模型参数——糟糕的初始化会导致梯度消失或梯度爆炸等不良行为。pre-norm Transformer 对初始化异常地稳健，但初始化仍可能对训练速度与收敛产生显著影响。由于本作业篇幅已经很长，我们把这些细节留到作业 3，这里先给出一些在多数情形下应能良好工作的近似初始化。现在请使用：

- **Linear 权重**：$\mathcal{N}\left(\mu = 0,\ \sigma^2 = \dfrac{2}{d_{in} + d_{out}}\right)$，并在 $[-3\sigma, 3\sigma]$ 处截断。
- **Embedding**：$\mathcal{N}(\mu = 0,\ \sigma^2 = 1)$，在 $[-3, 3]$ 处截断。
- **RMSNorm**：$\mathbf{1}$（即全 1）。

你应当用 `torch.nn.init.trunc_normal_` 来初始化这些截断正态权重。

#### 3.3.2 Linear 模块

线性层是 Transformer 乃至一般神经网络的基本构件。首先，你要实现自己的 `Linear` 类，它继承 `torch.nn.Module` 并执行线性变换：

$$y = Wx. \tag{3}$$

注意我们不含偏置项，这与大多数现代 LLM 一致。

**问题（linear）：实现线性模块（1 分）**

**交付物：** 实现一个继承 `torch.nn.Module` 并执行线性变换的 `Linear` 类。你的实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口，只是没有 bias 参数或偏置项。我们推荐以下接口：

- `def __init__(self, in_features, out_features, device=None, dtype=None)` — 构造一个线性变换模块。该函数应接受以下参数：
  - `in_features: int` — 输入的最后一维大小
  - `out_features: int` — 输出的最后一维大小
  - `device: torch.device | None = None` — 存放参数的设备
  - `dtype: torch.dtype | None = None` — 参数的数据类型
- `def forward(self, x: torch.Tensor) -> torch.Tensor` — 对输入施加线性变换。

务必做到：

- 继承 `nn.Module`；
- 调用父类构造函数；
- 把参数构造并存为 $W$（**不是** $W^\top$），并放进 `nn.Parameter`；
- 当然，不要使用 `nn.Linear` 或 `nn.functional.linear`。

初始化请使用上述设置，并用 `torch.nn.init.trunc_normal_` 初始化权重。

要测试你的 `Linear` 模块，请实现 `[adapters.run_linear]` 处的测试适配器。适配器应把给定权重加载进你的 `Linear` 模块，为此可以使用 `Module.load_state_dict`。然后运行 `uv run pytest -k test_linear`。

#### 3.3.3 Embedding 模块

如上所述，Transformer 的第一层是一个嵌入层，把整数 token ID 映射到维度为 `d_model` 的向量空间。我们将实现一个自定义的 `Embedding` 类，继承 `torch.nn.Module`（因此你不应使用 `nn.Embedding`）。`forward` 方法应通过用形状为 `(batch_size, sequence_length)` 的 `torch.LongTensor` token ID 去索引形状为 `(vocab_size, d_model)` 的嵌入矩阵，来取出每个 token ID 的嵌入向量。

**问题（embedding）：实现嵌入模块（1 分）**

**交付物：** 实现继承 `torch.nn.Module` 并执行嵌入查表的 `Embedding` 类。你的实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。我们推荐以下接口：

- `def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)` — 构造一个嵌入模块。该函数应接受以下参数：
  - `num_embeddings: int` — 词表大小
  - `embedding_dim: int` — 嵌入向量的维度，即 $d_{model}$
  - `device: torch.device | None = None` — 存放参数的设备
  - `dtype: torch.dtype | None = None` — 参数的数据类型
- `def forward(self, token_ids: torch.Tensor) -> torch.Tensor` — 查找给定 token ID 的嵌入向量。

务必做到：

- 继承 `nn.Module`；
- 调用父类构造函数；
- 把嵌入矩阵初始化为 `nn.Parameter`；
- 存储嵌入矩阵时把 `d_model` 作为最后一维；
- 当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`。

初始化同样使用上述设置，并用 `torch.nn.init.trunc_normal_` 初始化权重。

要测试你的实现，请实现 `[adapters.run_embedding]` 处的测试适配器，然后运行 `uv run pytest -k test_embedding`。

### 3.4 Pre-Norm Transformer 块

每个 Transformer 块包含两个子层：一个多头自注意力机制，以及一个逐位置前馈网络（[A. Vaswani et al., 2017]，第 3.1 节）。

在原始 Transformer 论文中，模型在这两个子层周围各有一条残差连接，之后跟一个层归一化。由于层归一化施加在子层输出上，这种架构通常被称为"post-norm"Transformer。然而，多项工作发现：把层归一化从每个子层的输出移到每个子层的输入（并在最后一个 Transformer 块之后再加一次层归一化），可以改善 Transformer 的训练稳定性 [T. Q. Nguyen et al., 2019; R. Xiong et al., 2020]——这种"pre-norm"Transformer 块的可视化见图 2。每个 Transformer 块子层的输出随后通过残差连接与该子层的输入相加（A. Vaswani et al. [8]，第 5.4 节）。对 pre-norm 的一种直觉是：从输入嵌入一直到 Transformer 的最终输出，存在一条**干净、不经过任何归一化的"残差流"**，据称这能改善梯度流动。这种 pre-norm Transformer 如今已是语言模型的标准做法（例如 GPT-3、LLaMA、PaLM 等），因此我们将实现这一变体。下面我们依次走过 pre-norm Transformer 块的各个组件，并按顺序逐一实现。

#### 3.4.1 均方根层归一化（RMSNorm）

A. Vaswani et al. [8] 的原始 Transformer 实现使用层归一化 [J. L. Ba et al., 2016] 来归一化激活值。遵循 H. Touvron et al. [12]，我们改用**均方根层归一化**（RMSNorm；B. Zhang et al. [13]，式 4）来做层归一化。给定激活值向量 $a \in \mathbb{R}^{d_{model}}$，RMSNorm 按如下方式重新缩放每个激活值 $a_i$：

$$\mathrm{RMSNorm}(a_i) = \frac{a_i}{\mathrm{RMS}(a)}\, g_i, \tag{4}$$

其中

$$\mathrm{RMS}(a) = \sqrt{\frac{1}{d_{model}}\sum_{i=1}^{d_{model}} a_i^2 + \varepsilon}.$$

这里 $g_i$ 是一个可学习的"增益（gain）"参数（总共有 `d_model` 个这样的参数），$\varepsilon$ 是一个通常固定为 `1e-5` 的超参数。

你应当先把输入**上转**到 `torch.float32`，以免对输入求平方时溢出。总体上，你的 `forward` 方法应形如：

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# 此处是你的 RMSNorm 代码
...
result = ...
# 以原始 dtype 返回结果
return result.to(in_dtype)
```

**问题（rmsnorm）：均方根层归一化（1 分）**

**交付物：** 把 RMSNorm 实现为一个 `torch.nn.Module`。我们推荐以下接口：

- `def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)` — 构造 RMSNorm 模块。该函数应接受以下参数：
  - `d_model: int` — 模型的隐藏维度
  - `eps: float = 1e-5` — 用于数值稳定性的 epsilon
  - `device: torch.device | None = None` — 存放参数的设备
  - `dtype: torch.dtype | None = None` — 参数的数据类型
- `def forward(self, x: torch.Tensor) -> torch.Tensor` — 处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量，返回同形状张量。

注意：如前述，执行归一化之前记得把输入上转到 `torch.float32`（之后再转回原始 dtype）。

要测试你的实现，请实现 `[adapters.run_rmsnorm]` 处的测试适配器，然后运行 `uv run pytest -k test_rmsnorm`。

#### 3.4.2 逐位置前馈网络

> **图 3：SiLU（又称 Swish）与 ReLU 激活函数的比较。**
> 图中曲线：SiLU: $f(x) = x\,\sigma(x)$；Identity: $f(x) = x$；ReLU: $f(x) = \max(0, x)$。

在原始 Transformer 论文中（A. Vaswani et al. [8] 第 3.3 节），Transformer 的前馈网络由两次线性变换构成，中间夹一个 ReLU 激活（$\mathrm{ReLU}(x) = \max(0, x)$）。在该原始架构中，内层前馈层的维度通常是输入维度的 4 倍。

然而，与现代语言模型相比，原始设计往往包含两处主要改动：使用另一种激活函数，并采用门控机制。具体而言，我们将实现被 Llama 3 [A. Grattafiori et al., 2024] 与 Qwen 2.5 [A. Yang et al., 2024] 等 LLM 采用的"SwiGLU"激活函数，它把 SiLU（常称 Swish）激活与一种称为门控线性单元（Gated Linear Unit, GLU）的门控机制结合起来。此外，依循 PaLM [A. Chowdhery et al., 2022] 与 LLaMA [H. Touvron et al., 2023] 以来大多数现代 LLM 的做法，我们也会省略线性层中有时会用的偏置项。

SiLU 或 Swish 激活函数 [D. Hendrycks et al., 2016; S. Elfwing et al., 2017] 定义如下：

$$\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}. \tag{5}$$

如图 3 所示，SiLU 激活函数与 ReLU 相似，但在零点处是**光滑**的。

门控线性单元（GLU）最初由 Y. N. Dauphin et al. [19] 定义为：一次经过 sigmoid 函数的线性变换与另一次线性变换的逐元素乘积：

$$\mathrm{GLU}(x, W_1, W_2) = \sigma(W_1 x) \odot W_2 x, \tag{6}$$

其中 $\odot$ 表示逐元素乘法。GLU 被认为能"通过为梯度提供一条线性通路、同时保留非线性能力，从而缓解深层架构中的梯度消失问题"。

把 SiLU/Swish 与 GLU 放在一起，就得到我们前馈网络将使用的 SwiGLU：

$$\mathrm{FFN}(x) = \mathrm{SwiGLU}(x, W_1, W_2, W_3) = W_2\left(\mathrm{SiLU}(W_1 x) \odot W_3 x\right), \tag{7}$$

其中 $x \in \mathbb{R}^{d_{model}}$，$W_1, W_3 \in \mathbb{R}^{d_{ff} \times d_{model}}$，$W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$，且按惯例 $d_{ff} = \frac{8}{3} d_{model}$。在具体实现中，为硬件效率起见，把它取整到附近的 64 的倍数即可。

N. Shazeer [20] 最先提出把 SiLU/Swish 激活与 GLU 结合起来，并通过实验表明 SwiGLU 在语言建模任务上优于 ReLU 以及不带门控的 SiLU 等基线。作业后文你会比较 SwiGLU 与 SiLU。尽管我们为这些组件给出了一些启发式论证（论文也提供了更多支持证据），保持经验视角仍是好的：Shazeer 论文中如今很有名的一句话是

> "对于这些架构为何似乎有效，我们不提供解释；我们将其成功与其他一切一样，归因于神的恩典。"

**问题（positionwise_feedforward）：实现逐位置前馈网络（2 分）**

**交付物：** 实现 SwiGLU 前馈网络，由一个 SiLU 激活函数与一个 GLU 组成。

注意：在这一特定情形下，你可以放心地在实现中使用 `torch.sigmoid`，以获得数值稳定性。

实现时你应把 $d_{ff}$ 设为约 $\frac{8}{3} \times d_{model}$，同时确保内层前馈层的维度是 64 的倍数，以充分利用硬件。要用我们提供的测试检验你的实现，你需要实现 `[adapters.run_swiglu]` 处的测试适配器，然后运行 `uv run pytest -k test_swiglu`。

#### 3.4.3 相对位置嵌入

为了把位置信息注入模型，我们将实现**旋转位置嵌入**（Rotary Position Embeddings）[J. Su et al., 2021]，通常称为 RoPE。对于位置 $i$ 处的查询 token $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$，我们会施加一个成对旋转矩阵 $R_i$，得到 $q'^{(i)} = R_i q^{(i)} = R_i W_q x^{(i)}$。这里 $R_i$ 会把成对的嵌入元素 $q^{(i)}_{2k-1:2k}$ 当作 2 维向量，按角度

$$\theta_{i,k} = i\,\Theta^{-\frac{2k-2}{d}}, \qquad k \in \{1, \dots, d/2\}$$

进行旋转，其中 $\Theta$ 是某个常数。因此，可以把 $R_i$ 视为一个 $d \times d$ 的块对角矩阵，其对角块为 $R_i^k$（$k \in \{1, \dots, d/2\}$），其中

$$R_i^k = \begin{pmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{pmatrix}. \tag{8}$$

于是得到完整的旋转矩阵

$$R_i = \begin{pmatrix} R_i^1 & 0 & 0 & \cdots & 0 \\ 0 & R_i^2 & 0 & \cdots & 0 \\ 0 & 0 & R_i^3 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & R_i^{d/2} \end{pmatrix}, \tag{9}$$

其中 0 表示 $2 \times 2$ 的零矩阵。虽然可以真的构造完整的 $d \times d$ 矩阵，但好的解法应利用该矩阵的性质，以更高效的方式实现这次变换。由于我们只关心给定序列内各 token 之间的**相对旋转**，我们可以把计算出的 $\cos(\theta_{i,k})$ 与 $\sin(\theta_{i,k})$ 跨层、跨批次复用。若想进一步优化，可以让所有层共用一个 RoPE 模块，并在其中用 `self.register_buffer(persistent=False)` 在初始化时创建一个 2 维的 sin/cos 预计算缓冲区，而不是用 `nn.Parameter`（因为我们不想学习这些固定的余弦、正弦值）。对 $k^{(j)}$ 则施加与 $q^{(i)}$ 完全相同的旋转过程，只是按对应的 $R_j$ 旋转。注意这一层**没有可学习参数**。

**问题（rope）：实现 RoPE（2 分）**

**交付物：** 实现一个 `RotaryPositionalEmbedding` 类，把 RoPE 施加到输入张量上。

推荐以下接口：

- `def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)` — 构造 RoPE 模块，并按需创建缓冲区。
  - `theta: float` — RoPE 的 $\Theta$ 值
  - `d_k: int` — 查询与键向量的维度
  - `max_seq_len: int` — 将被输入的最大序列长度
  - `device: torch.device | None = None` — 存放缓冲区的设备
- `def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor` — 处理形状为 `(..., seq_len, d_k)` 的输入张量，返回同形状张量。注意你应当容忍 $x$ 带有任意多个批处理维度。你应假定 `token_positions` 是形状为 `(..., seq_len)` 的张量，指明 $x$ 沿序列维度各位置对应的 token 位置。
- 你应当用这些 token 位置沿序列维度去切片你（可能已预计算的）cos 与 sin 张量。

要测试你的实现，请完成 `[adapters.run_rope]`，并确保它通过 `uv run pytest -k test_rope`。

#### 3.4.4 缩放点积注意力

现在我们来实现在 A. Vaswani et al. [8]（第 3.2.1 节）中描述的缩放点积注意力。作为预备，Attention 运算的定义会用到 softmax——该运算把一个未归一化的分数向量变成归一化分布：

$$\mathrm{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^{n} \exp(v_j)}. \tag{10}$$

注意当取值较大时 $\exp(v_i)$ 可能变成 `inf`（于是 $\frac{\inf}{\inf} = \text{NaN}$）。我们可以通过注意到 softmax 对"给所有输入加上任意常数 $c$"不变来避免这一点。可以利用该性质改善数值稳定性——通常做法是从 $v$ 的所有元素中减去 $v$ 的最大元素，使新的最大元素为 0。下面你将用这一技巧实现 softmax。

**问题（softmax）：实现 softmax（1 分）**

**交付物：** 写一个对张量施加 softmax 运算的函数。该函数应接受两个参数：一个张量与一个维度 $i$，并对输入张量的第 $i$ 维施加 softmax。输出张量应与输入张量形状相同，但其第 $i$ 维现在是一个归一化概率分布。请使用"从第 $i$ 维所有元素中减去该维最大值"这一技巧来避免数值稳定性问题。

要测试你的实现，请完成 `[adapters.run_softmax]`，并确保它通过 `uv run pytest -k test_softmax_matches_pytorch`。

现在我们可以用数学方式定义 Attention 运算：

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V, \tag{11}$$

其中 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，$V \in \mathbb{R}^{m \times d_v}$。这里 $Q$、$K$、$V$ 都是该运算的**输入**——注意它们**不是**可学习参数。

**掩码（Masking）**：有时对注意力运算的输出做掩码会很方便。掩码的形状应为 $M \in \{\text{True}, \text{False}\}^{n \times m}$，该布尔矩阵的第 $i$ 行指明查询 $i$ 应当关注哪些键。按惯例（也稍显迷惑）地说：位置 $(i,j)$ 处为 `True` 表示查询 $i$ **确实**关注键 $j$，为 `False` 表示不关注；换言之，在值为 `True` 的 $(i,j)$ 对上"信息可以流动"。例如，考虑一个 $1 \times 3$ 的掩码矩阵，取值为 `[[True, True, False]]`，则唯一的那个查询向量只关注前两个键。

在计算上，使用掩码比只在子序列上计算注意力高效得多；具体做法是取 softmax **之前**的值（$QK^\top/\sqrt{d_k}$），对掩码矩阵中所有为 `False` 的位置**加上 $-\infty$**。

**问题（scaled_dot_product_attention）：实现缩放点积注意力（5 分）**

**交付物：** 实现缩放点积注意力函数。你的实现应能处理形状为 `(batch_size, ..., seq_len, d_k)` 的键与查询，以及形状为 `(batch_size, ..., seq_len, d_v)` 的值，其中 `...` 表示任意多个其他类批处理维度（若提供）。实现应返回形状为 `(batch_size, ..., seq_len, d_v)` 的输出。关于类批处理维度的讨论见 §3.2。

你的实现还应支持一个可选的、形状为 `(seq_len, seq_len)` 的布尔掩码。掩码为 `True` 的位置上的注意力概率之和应为 1，掩码为 `False` 的位置上的注意力概率应为 0。

要用我们提供的测试检验你的实现，你需要实现 `[adapters.run_scaled_dot_product_attention]` 处的测试适配器。`uv run pytest -k test_scaled_dot_product_attention` 会在三阶输入张量上测试你的实现，而 `uv run pytest -k test_4d_scaled_dot_product_attention` 会在四阶输入张量上测试你的实现。

#### 3.4.5 因果多头自注意力

我们将按 A. Vaswani et al. [8] 第 3.2.2 节实现多头自注意力。回顾一下，多头注意力的运算在数学上定义如下：

$$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h) \tag{12}$$

其中

$$\mathrm{head}_i = \mathrm{Attention}(Q_i, K_i, V_i), \tag{13}$$

$Q_i, K_i, V_i$ 分别是 $Q$、$K$、$V$ 的嵌入维度中大小为 $d_k$ 或 $d_v$ 的第 $i$ 个切片（$i \in \{1, \dots, h\}$），而 Attention 即 §3.4.4 定义的缩放点积注意力。由此可以构成多头自注意力运算：

$$\mathrm{MultiHeadSelfAttention}(x) = W_O\,\mathrm{MultiHead}(W_Q x, W_K x, W_V x). \tag{14}$$

这里可学习参数为 $W_Q \in \mathbb{R}^{h d_k \times d_{model}}$、$W_K \in \mathbb{R}^{h d_k \times d_{model}}$、$W_V \in \mathbb{R}^{h d_v \times d_{model}}$、$W_O \in \mathbb{R}^{d_{model} \times h d_v}$。由于在多头注意力运算中 $Q$、$K$、$V$ 会被切片，可以认为 $W_Q$、$W_K$、$W_V$ 沿输出维度按头被分隔开了。当你把这一步做通之后，你应当总共只用**三次矩阵乘法**就算出键、值、查询的投影⁵。

> ⁵ 作为延伸目标，试着把键、查询、值的投影合并成单个权重矩阵，从而只需一次矩阵乘法。

**因果掩码（Causal masking）**

你的实现应阻止模型关注序列中未来的 token。换言之，若模型收到 token 序列 $t_1, \dots, t_n$，而我们想计算前缀 $t_1, \dots, t_i$（其中 $i < n$）的下一词预测，那么模型不应能访问（关注）位置 $t_{i+1}, \dots, t_n$ 上的 token 表示，因为推理生成文本时它根本拿不到这些 token（而且这些未来 token 泄露了真实下一词的身份信息，会使得语言建模预训练目标变得毫无意义）。对于输入 token 序列 $t_1, \dots, t_n$，朴素地阻止访问未来 token 的做法是运行 $n$ 次多头自注意力（对应序列中 $n$ 个不同前缀）。我们改用**因果注意力掩码**，允许 token $i$ 关注序列中所有位置 $j \le i$。你可以用 `torch.triu` 或广播的索引比较来构造该掩码，并应利用这一点：你在 §3.4.4 实现的缩放点积注意力已经支持注意力掩码。

**施加 RoPE**

RoPE 应施加于查询与键向量，而**不**施加于值向量。此外，头维度应当作为**批处理维度**处理，因为在多头注意力中，注意力是对每个头独立施加的。这意味着对每个头而言，查询与键向量上应施加**完全相同**的 RoPE 旋转。

**问题（multihead_self_attention）：实现因果多头自注意力（5 分）**

**交付物：** 把因果多头自注意力实现为一个 `torch.nn.Module`。你的实现应至少接受以下参数：

- `d_model: int` — Transformer 块输入的维度
- `num_heads: int` — 多头自注意力使用的头数

依循 A. Vaswani et al. [8]，取 $d_k = d_v = \dfrac{d_{model}}{h}$。要用我们提供的测试检验你的实现，请实现 `[adapters.run_multihead_self_attention]` 处的测试适配器，然后运行 `uv run pytest -k test_multihead_self_attention`。

### 3.5 完整的 Transformer LM

我们先从组装 Transformer 块开始（回顾图 2 会有帮助）。一个 Transformer 块包含两个"子层"：一个用于多头自注意力，另一个用于 SwiGLU 前馈网络。在每个子层中，我们先做 RMSNorm，再做主运算（MHA/FF），最后加上残差连接。

具体而言，Transformer 块的前一半（第一个"子层"）应实现如下更新，由输入 $x$ 得到输出 $y$：

$$y = x + \mathrm{MultiHeadSelfAttention}(\mathrm{RMSNorm}(x)). \tag{15}$$

**问题（transformer_block）：实现 Transformer 块（3 分）**

实现 §3.4 描述、图 2 所示的 pre-norm Transformer 块。你的 Transformer 块应至少接受以下参数：

- `d_model: int` — Transformer 块输入的维度
- `num_heads: int` — 多头自注意力使用的头数
- `d_ff: int` — 逐位置前馈内层的维度

要测试你的实现，请实现 `[adapters.run_transformer_block]` 处的适配器，然后运行 `uv run pytest -k test_transformer_block`。

**交付物：** 通过所提供测试的 Transformer 块代码。

现在我们把各个块组装起来，遵循图 1 中的高层示意图。按我们在 §3.1.0.1 中对嵌入的描述［原文如此］，把嵌入输出送入 `num_layers` 个 Transformer 块，然后把它送入最后的层归一化与 LM head，得到词表上的未归一化分布（即 logits）。

**问题（transformer_lm）：实现 Transformer LM（3 分）**

该把所有东西拼起来了！按 §3.1 的描述、图 1 所示实现 Transformer 语言模型。你的实现至少应接受前述 Transformer 块的全部构造参数，以及以下额外参数：

- `vocab_size: int` — 词表大小，用于确定 token 嵌入矩阵的维度
- `context_length: int` — 最大上下文长度，用于确定 RoPE 的 sin/cos 缓冲区的维度
- `num_layers: int` — 使用的 Transformer 块数量

要用我们提供的测试检验你的实现，你首先需要实现 `[adapters.run_transformer_lm]` 处的测试适配器，然后运行 `uv run pytest -k test_transformer_lm`。

**交付物：** 通过上述测试的 Transformer LM 模块。

#### 资源核算

能够理解 Transformer 各部分如何消耗计算与内存是很有用的。我们将走一遍做基础"FLOPs 核算"的步骤。Transformer 中绝大多数 FLOPs 来自矩阵乘法，因此核心思路很简单：

1. 写出 Transformer 前向传播中的所有矩阵乘法。
2. 把每个矩阵乘法换算成所需的 FLOPs。

对于第二步，下面这个事实会很有用：

> **规则**：给定 $A \in \mathbb{R}^{m \times n}$ 与 $B \in \mathbb{R}^{n \times p}$，矩阵-矩阵乘积 $AB$ 需要 $2mnp$ 个 FLOPs。

要看出这一点，注意 $(AB)[i,j] = A[i,:] \cdot B[:,j]$，而这个点积需要 $n$ 次加法与 $n$ 次乘法（共 $2n$ 个 FLOPs）。又因为矩阵乘积 $AB$ 有 $m \times p$ 个元素，故总 FLOPs 为 $(2n)(mp) = 2mnp$。

在做下一题之前，先逐一走过你 Transformer 块与 Transformer LM 的每个组件，列出所有矩阵乘法及其对应的 FLOPs 开销，可能会有帮助。

**问题（transformer_accounting）：Transformer LM 资源核算（5 分）**

(a) 考虑一个采用本作业架构、规模相当于 GPT-2 XL 的模型，其配置如下：

```
vocab_size:      50,257
context_length:  1,024
num_layers:      48
d_model:         1,600
num_heads:       25
d_ff:            4,288   （最接近 8/3 × 1,600 的 64 的倍数）
```

假设我们用该配置构造模型。我们的模型有多少可训练参数？若每个参数用单精度浮点表示，仅加载该模型需要多少内存？
**交付物：** 一句到两句话回答。

(b) 指出完成一次 GPT-2 XL 规模模型前向传播所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设输入序列有 `context_length` 个 token。
**交付物：** 一份矩阵乘法清单（含说明），以及所需 FLOPs 总数。

(c) 基于上面的分析，模型的哪些部分需要最多 FLOPs？
**交付物：** 一句到两句话回答。

(d) 对 GPT-2 small（12 层、768 d_model、12 头）、GPT-2 medium（24 层、1024 d_model、16 头）与 GPT-2 large（36 层、1280 d_model、20 头）重复上述分析。随着模型规模增大，Transformer LM 的哪些部分在总 FLOPs 中占比上升、哪些下降？
**交付物：** 对每个模型给出各组件及其 FLOPs 占比的分解（占一次前向传播所需总 FLOPs 的比例）。此外，用一两句话描述改变模型规模如何影响各组件 FLOPs 的占比。

(e) 取 GPT-2 XL，把上下文长度增大到 16,384。一次前向传播的总 FLOPs 如何变化？各组件 FLOPs 的相对贡献如何变化？
**交付物：** 一句到两句话回答。

---

*第 3 部分译文到此结束；第 4 节起进入 Transformer LM 的训练（交叉熵、AdamW、学习率调度、梯度裁剪）。*
