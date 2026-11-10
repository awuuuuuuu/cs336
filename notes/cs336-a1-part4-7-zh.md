# CS336 作业 1（basics）讲义译文 · 第 4–7 部分：训练、生成与实验

> **原文**：`cs336_assignment1_basics.pdf`，版本 26.0.3，Spring 2026，第 28–47 页（§4 至 §7 及参考文献）
> **性质**：个人学习用非官方译文，逐段对照原文翻译；术语优先保留英文原词。原文为准。
> **接续**：上接 `cs336-a1-part3-zh.md`（§3）。
> **说明**：数据集样例与模型生成样例（`tinystories_example`、`ts_generate_example`、`owt_example`）保留英文原文不译——它们的意义就在于展示英文文本本身。参考文献同样保留原样。
> **注意**：本文件不属于作业提交物。提交前请把 `notes/` 加入 `make_submission.sh` 的排除列表（如 `-x 'notes/*'`）。

---

## 4 训练 Transformer LM

至此我们已经有了预处理数据（通过分词器）与模型（Transformer）的步骤。剩下的就是构建支撑训练的全部代码，包括：

- **损失**：我们需要定义损失函数（交叉熵）。
- **优化器**：我们需要定义最小化该损失的优化器（AdamW）。
- **训练循环**：我们需要加载数据、保存检查点、管理训练的全部支撑设施。

### 4.1 交叉熵损失

回顾一下：Transformer 语言模型为每个长度为 $m+1$ 的序列 $x$ 定义分布 $p_\theta(x_{i+1} \mid x_{1:i})$，其中 $i = 1, \dots, m$。给定由长度为 $m+1$ 的序列组成的训练集 $D$，我们定义标准的交叉熵（负对数似然）损失函数：

$$\ell(\theta; D) = \frac{1}{|D|\,m} \sum_{x \in D} \sum_{i=1}^{m} -\log p_\theta(x_{i+1} \mid x_{1:i}). \tag{16}$$

（注意：Transformer 的一次前向传播就能给出所有 $i = 1, \dots, m$ 的 $p_\theta(x_{i+1} \mid x_{1:i})$。）

具体而言，Transformer 为每个位置 $i$ 计算出 logits $o_i \in \mathbb{R}^{vocab\_size}$，由此得到⁶：

$$p(x_{i+1} \mid x_{1:i}) = \mathrm{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{vocab\_size} \exp(o_i[a])}. \tag{17}$$

交叉熵损失一般是针对 logits 向量 $o_i \in \mathbb{R}^{vocab\_size}$ 与目标 $x_{i+1}$ 定义的⁷。

> ⁶ 注意 $o_i[k]$ 指数组 $o_i$ 中下标为 $k$ 的取值。
> ⁷ 这对应于 $x_{i+1}$ 上的 Dirac delta 分布与预测分布 $\mathrm{softmax}(o_i)$ 之间的交叉熵。

与 softmax 的情形一样，实现交叉熵损失时也需要小心处理数值问题。

**问题（cross_entropy）：实现交叉熵（1 分）**

**交付物：** 写一个计算交叉熵损失的函数，接受预测的 logits（$o_i$）与目标（$x_{i+1}$），计算交叉熵 $\ell_i = -\log \mathrm{softmax}(o_i)[x_{i+1}]$。你的函数应处理以下事项：

- 为数值稳定性减去最大元素。
- 尽可能把 log 与 exp 抵消。
- 处理任意额外的批处理维度，并返回批上的平均。与 §3.2 一致，我们假定类批处理维度总是排在前面、位于词表大小维度之前。

实现 `[adapters.run_cross_entropy]`，然后运行 `uv run pytest -k test_cross_entropy` 测试你的实现。

#### 困惑度（Perplexity）

交叉熵足以用于训练，但在评估模型时，我们还希望报告困惑度。对于长度为 $m$、各位置交叉熵损失为 $\ell_1, \dots, \ell_m$ 的序列：

$$\text{perplexity} = \exp\left(\frac{1}{m}\sum_{i=1}^{m} \ell_i\right). \tag{18}$$

### 4.2 SGD 优化器

有了损失函数，我们开始探索优化器。最简单的基于梯度的优化器是随机梯度下降（SGD）。我们从随机初始化的参数 $\theta_0$ 出发，然后对每一步 $t = 0, \dots, T-1$ 执行如下更新：

$$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t), \tag{19}$$

其中 $B_t$ 是从数据集 $D$ 中随机采样的一批数据，学习率 $\alpha_t$ 与批大小 $|B_t|$ 是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD

为实现我们的优化器，我们将继承 PyTorch 的 `torch.optim.Optimizer` 类。`Optimizer` 子类必须实现两个方法：

- `def __init__(self, params, ...)` 应初始化你的优化器。这里的 `params` 是待优化的参数集合（也可能是参数组，以防用户想对模型不同部分使用不同的超参数，例如不同的学习率）。务必把 `params` 传给基类的 `__init__` 方法，基类会保存这些参数以供 `step` 使用。你可以依据优化器接受额外参数（例如学习率是常见的），并把它们以字典形式传给基类构造函数，字典的键是你为这些参数选定的名称（字符串）。
- `def step(self)` 应执行一次参数更新。在训练循环中，它会在反向传播之后被调用，因此你能拿到上一批数据的梯度。该方法应遍历每个参数张量 `p` 并**原地**修改它们，即设置 `p.data`——它保存着与参数关联的张量——依据是 `p.grad`（若存在），即损失对该参数的梯度张量。

PyTorch 优化器 API 有一些细微之处，用例子来说明更容易。为了让例子更丰富，我们实现 SGD 的一个小变体：学习率随训练衰减，从初始学习率 $\alpha$ 开始，随时间的推移步子越来越小：

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla L(\theta_t; B_t). \tag{20}$$

看看这个版本的 SGD 作为 PyTorch `Optimizer` 该如何实现：

```python
from collections.abc import Callable, Iterable
from typing import Optional

import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 取出学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 取出与 p 关联的状态。
                t = state.get("t", 0)  # 从状态中取出迭代数，若没有则为 0。
                grad = p.grad.data  # 取出损失对 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad  # 原地更新权重张量。
                state["t"] = t + 1  # 迭代数加一。
        return loss
```

在 `__init__` 中，我们把参数以及默认超参数传给基类构造函数（参数可能是分组给出的，各组超参数不同）。如果参数只是一组 `torch.nn.Parameter` 对象，基类构造函数会创建一个组并为其赋予默认超参数。随后在 `step` 中，我们先遍历每个参数组，再遍历组内每个参数，施加第 20 式。这里我们把迭代数保存为与每个参数关联的状态：先读出该值、用于梯度更新，然后再更新它。该 API 规定用户可能传入一个可调用闭包，以便在优化器步进之前重算损失。我们用到的优化器不需要它，但为符合 API 我们仍然保留。

要看到它跑起来，我们可以用下面这个最小训练循环示例：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad()  # 重置所有可学习参数的梯度。
    loss = (weights**2).mean()  # 计算标量损失值。
    print(loss.cpu().item())
    loss.backward()  # 运行反向传播，计算梯度。
    opt.step()  # 执行优化器步进。
```

这就是训练循环的典型结构：在每次迭代中，我们计算损失并执行一步优化器。训练语言模型时，我们的可学习参数来自模型（在 PyTorch 中，`m.parameters()` 给出该集合）。损失将在一个采样出来的数据批上计算，但训练循环的基本结构完全相同。

**问题（learning_rate_tuning）：调学习率（1 分）**

我们将会看到，对训练影响最大的超参数之一就是学习率。让我们在上面的玩具例子中实际验证一下。用另外三个学习率——1e1、1e2、1e3——运行上面的 SGD 示例，只跑 10 次训练迭代。每个学习率下损失表现如何？它是衰减得更快、更慢，还是发散了（即在训练过程中不断增大）？

**交付物：** 用一两句话回答你观察到的行为。

### 4.3 AdamW

现代语言模型通常不用 SGD，而用更精细的优化器。近期使用的大多数优化器都是 Adam 优化器 [D. P. Kingma et al., 2015] 的衍生。我们将使用 AdamW [I. Loshchilov et al., 2019]，它在近期工作中被广泛使用。AdamW 对 Adam 提出了一项改进：以与梯度更新**解耦**的方式加入权重衰减（每次迭代把参数往 0 拉），从而改善正则化。我们将按 I. Loshchilov et al. [23] 的算法 2 实现 AdamW。

AdamW 是有状态的：对每个参数，它维护该参数一阶矩与二阶矩的滑动估计。因此 AdamW 用额外的内存换取更好的稳定性与收敛性。除学习率 $\alpha$ 外，AdamW 还有一对控制矩估计更新的超参数 $(\beta_1, \beta_2)$，以及权重衰减率 $\lambda$。典型应用把 $(\beta_1, \beta_2)$ 设为 $(0.9, 0.999)$，但 LLaMA [H. Touvron et al., 2023] 与 GPT-3 [T. B. Brown et al., 2020] 这类大语言模型常用 $(0.9, 0.95)$ 训练。该算法可写如下，其中 $\varepsilon$ 是一个很小的值（例如 $10^{-8}$），用于在 $v$ 出现极小值时改善数值稳定性：

> **算法 1：AdamW 优化器**
>
> ```
>  1  init(θ)                          ▷ 初始化可学习参数
>  2  m ← 0                            ▷ 一阶矩向量的初值；与 θ 同形状
>  3  v ← 0                            ▷ 二阶矩向量的初值；与 θ 同形状
>  4  for t = 1, ..., T do
>  5      采样一批数据 B_t
>  6      g ← ∇_θ ℓ(θ; B_t)            ▷ 计算损失的梯度
>  7      α_t ← α · √(1 − β₂^t) / (1 − β₁^t)   ▷ 计算第 t 次迭代的调整后 α
>  8      θ ← θ − α λ θ                ▷ 施加权重衰减
>  9      m ← β₁ m + (1 − β₁) g        ▷ 更新一阶矩估计
> 10      v ← β₂ v + (1 − β₂) g²       ▷ 更新二阶矩估计
> 11      θ ← θ − α_t · m / (√v + ε)   ▷ 施加经矩调整的权重更新
> 12  end for
> ```
>
> 算法 1：AdamW 优化器伪代码。

注意 $t$ 从 1 开始。现在你来实现这个优化器。

**问题（adamw）：实现 AdamW（2 分）**

**交付物：** 把 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接受学习率 $\alpha$，以及 $\beta$、$\varepsilon$、$\lambda$ 这些超参数。为帮你保存状态，基类 `Optimizer` 提供了字典 `self.state`，它把 `nn.Parameter` 对象映射到一个字典，用来存放你为该参数需要的任何信息（对 AdamW 而言就是矩估计）。实现 `[adapters.get_adamw_cls]`，并确保它通过 `uv run pytest -k test_adamw`。

**问题（adamw_accounting）：AdamW 训练的资源核算（2 分）**

我们来计算运行 AdamW 需要多少内存与计算量。假设每个张量都使用 `float32`。

(a) 运行 AdamW 需要多少峰值内存？请按参数、激活值、梯度与优化器状态的内存占用分别拆解你的答案。用 `batch_size` 与模型超参数（`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`）表示你的答案。假设 $d_{ff} = \frac{8}{3} \times d_{model}$。

为简化起见，计算激活值内存占用时只考虑以下组件：

- Transformer 块
  - RMSNorm
  - 多头自注意力子层：$QKV$ 投影、$QK^\top$ 矩阵乘、softmax、值的加权求和、输出投影
  - 逐位置前馈（SwiGLU）：$W_1$、$W_2$、门控分支上的 SiLU、逐元素乘积、$W_3$
- 最后的 RMSNorm
- 输出嵌入
- logits 上的交叉熵

**交付物：** 分别为参数、激活值、梯度与优化器状态给出代数表达式，以及总计。

(b) 把你的答案代入 GPT-2 XL 规模的模型，得到一个只依赖于 `batch_size` 的表达式。在不超过 80GB 内存的前提下，你能用的最大批大小是多少？
**交付物：** 一个形如 $a \cdot batch\_size + b$ 的表达式（$a,b$ 为具体数值），以及最大批大小这个数值。

(c) 执行一步 AdamW 需要多少 FLOPs？
**交付物：** 一个代数表达式，并附简要论证。

(d) 模型 FLOPs 利用率（Model FLOPs Utilization, MFU）定义为观测吞吐量（tokens/second）与硬件理论峰值 FLOP 吞吐量之比 [A. Chowdhery et al., 2022]。一块 NVIDIA H100 GPU 在"float32"（实际是 TensorFloat-32，真实意义上更接近 "bfloat19"）运算上的理论峰值为 495 teraFLOP/s。假设你能达到 50% MFU，在单块 H100 上以批大小 1024 训练 GPT-2 XL 共 400K 步需要多久？依循 J. Kaplan et al. [25] 与 J. Hoffmann et al. [26]，假定反向传播的 FLOPs 是前向传播的两倍。
**交付物：** 训练所需小时数，并附简要论证。

### 4.4 学习率调度

能使损失下降最快的学习率取值，往往在训练过程中会变化。训练 Transformer 时，典型做法是使用学习率调度：开始用较大的学习率、在初期做出更快的更新，然后随训练推进把它缓慢衰减到较小的值⁸。在本作业中，我们将实现 LLaMA 训练所用的余弦退火（cosine annealing）调度 [H. Touvron et al., 2023]。

调度器就是一个函数：接受当前步 $t$ 与其他相关参数（例如初始与最终学习率），返回第 $t$ 步梯度更新应使用的学习率。最简单的调度是常值函数，对任何 $t$ 都返回同一个学习率。

余弦退火学习率调度接受 (i) 当前迭代 $t$、(ii) 最大学习率 $\alpha_{max}$、(iii) 最小（最终）学习率 $\alpha_{min}$、(iv) 预热迭代数 $T_w$、(v) 余弦退火的终止迭代 $T_c$。第 $t$ 次迭代的学习率定义为：

- **预热（Warm-up）**：若 $t < T_w$，则 $\alpha_t = \dfrac{t}{T_w}\alpha_{max}$。
- **余弦退火（Cosine annealing）**：若 $T_w \le t \le T_c$，则 $\alpha_t = \alpha_{min} + \dfrac{1}{2}\left(1 + \cos\left(\dfrac{t - T_w}{T_c - T_w}\pi\right)\right)(\alpha_{max} - \alpha_{min})$。
- **退火之后（Post-annealing）**：若 $t > T_c$，则 $\alpha_t = \alpha_{min}$。

> ⁸ 有时也常见使用让学习率重新上升（restart）的调度，以帮助跳出局部极小。

**问题（learning_rate_schedule）：实现带预热的余弦学习率调度（1 分）**

写一个函数，接受 $t$、$\alpha_{max}$、$\alpha_{min}$、$T_w$ 与 $T_c$，按上述调度返回学习率 $\alpha_t$。然后实现 `[adapters.get_lr_cosine_schedule]`，并确保它通过 `uv run pytest -k test_get_lr_cosine_schedule`。

### 4.5 梯度裁剪

训练中，我们有时会遇到产生很大梯度的训练样本，这可能使训练失稳。缓解这一问题的常用技术是**梯度裁剪**：思路是在每次反向传播之后、执行优化器步进之前，对梯度的范数施加上限。

给定（全部参数的）梯度 $g$，计算其 $\ell_2$ 范数 $\|g\|_2$。若该范数小于某个最大值 $M$，则 $g$ 保持不变；否则把 $g$ 按因子 $\dfrac{M}{\|g\|_2 + \varepsilon}$ 缩小（其中加一个很小的 $\varepsilon$，例如 $10^{-6}$，以改善数值稳定性）。注意缩放后的范数会略小于 $M$。

**问题（gradient_clipping）：实现梯度裁剪（1 分）**

写一个实现梯度裁剪的函数。函数应接受一个参数列表与一个最大 $\ell_2$ 范数，并**原地**修改每个参数的梯度。使用 $\varepsilon = 10^{-6}$（PyTorch 的默认值）。然后实现 `[adapters.run_gradient_clipping]` 处的适配器，并确保它通过 `uv run pytest -k test_gradient_clipping`。

---

## 5 训练循环

现在，我们终于要把前面构建的主要组件拼在一起：分词后的数据、模型与优化器。

### 5.1 数据加载器

分词后的数据（例如你在 `tokenizer_experiments` 中准备好的数据）是一个单一的 token 序列 $x = (x_1, \dots, x_n)$。尽管源数据可能由彼此独立的文档组成（例如不同网页、或源代码文件），常见做法是把它们全部拼接成一个 token 序列，并在其间加入分隔符（例如 `<|endoftext|>` token）。

数据加载器把它变成一批批的流，每批由 $B$ 个长度为 $m$ 的序列构成，并配上相应的、长度同样为 $m$ 的下一 token 目标。例如，对 $B = 1$、$m = 3$，$([x_2, x_3, x_4], [x_3, x_4, x_5])$ 可以是一个批。

以这种方式加载数据会在若干方面简化训练。首先，任何 $1 \le i \le n - m$ 都给出一个合法的训练序列，因此采样训练序列变得轻而易举。其次，由于所有训练序列长度相同，无需对输入序列做填充，这提升了硬件利用率（也便于增大批大小 $B$）。最后，我们也不需要把整个数据集载入内存才能采样训练数据，从而轻松处理原本装不下的大数据集。

**问题（data_loading）：实现数据加载（2 分）**

**交付物：** 写一个函数，接受一个 NumPy 数组 $x$（含 token ID 的整数数组）、`batch_size`、`context_length` 与一个 PyTorch 设备字符串（例如 `'cpu'` 或 `'cuda:0'`），返回一对张量：采样出的输入序列与对应的下一 token 目标。两个张量的形状都应为 `(batch_size, context_length)`，包含 token ID，且都应放置在请求的设备上。要用我们提供的测试检验你的实现，你首先需要实现 `[adapters.run_get_batch]` 处的测试适配器，然后运行 `uv run pytest -k test_get_batch`。

> **低资源提示：在 CPU 或 Apple Silicon 上做数据加载**
> 如果你打算在 CPU 或 Apple Silicon 上训练 LM，需要把数据移到正确的设备上（同样，之后模型也应使用同一设备）。
> 若在 CPU 上，可以使用 `'cpu'` 设备字符串；若在 Apple Silicon（M* 芯片）上，可以使用 `'mps'` 设备字符串。
> 关于 MPS 的更多资料：
> - https://docs.pytorch.org/docs/stable/mps.html
> - https://docs.pytorch.org/docs/stable/notes/mps.html
> - https://developer.apple.com/documentation/metalperformanceshaders

如果数据集大到装不进内存怎么办？我们可以使用名为 `mmap` 的 Unix 系统调用：它把磁盘上的文件映射到虚拟内存，并在访问相应内存位置时惰性加载文件内容。这样你就可以"假装"整个数据集都在内存里。NumPy 通过 `np.memmap`（或者，如果你当初用 `np.save` 保存数组，则用 `np.load` 的 `mmap_mode='r'` 标志）实现这一点，返回一个类数组对象，随访问按需加载条目。训练时从数据集（即 NumPy 数组）采样，务必以内存映射模式加载数据集（用 `np.memmap`，或用 `np.load` 的 `mmap_mode='r'` 标志，取决于你当初如何保存数组）。同时确保指定的 `dtype` 与你要加载的数组一致。显式验证内存映射数据看起来是否正确（例如不包含超出预期词表大小的值）可能会有帮助。

### 5.2 检查点

除加载数据外，我们还需要在训练过程中保存模型。运行作业任务时，我们常常希望能恢复一个中途停止的训练（例如因为任务超时、机器故障等）。即便一切顺利，我们之后也可能想访问中间模型（例如事后研究训练动态、从训练不同阶段的模型采样等）。

一个检查点应包含恢复训练所需的全部状态。至少我们希望能恢复模型权重。如果使用有状态的优化器（例如 AdamW），我们还需要保存优化器的状态（例如 AdamW 的矩估计）。最后，为恢复学习率调度，我们需要知道停在哪个迭代。PyTorch 让保存这些都很容易：每个 `nn.Module` 都有 `state_dict()` 方法，返回含全部可学习权重的字典；之后可以用姊妹方法 `load_state_dict()` 恢复这些权重。任何 `torch.optim.Optimizer` 同理。最后，`torch.save(obj, dest)` 可以把一个对象（例如一个字典，其部分值张量、部分值也可以是整数这样的普通 Python 对象）转储到文件（路径）或类文件对象，随后可用 `torch.load(src)` 载回内存。

**问题（checkpointing）：实现模型检查点（1 分）**

实现以下两个函数来加载与保存检查点：

- `def save_checkpoint(model, optimizer, iteration, out)` 应把模型、优化器与迭代数中的全部状态转储到类文件对象 `out`。你可以用模型与优化器的 `state_dict` 方法取得各自的相关状态，并用 `torch.save(obj, out)` 把 `obj` 转储到 `out`（PyTorch 此处支持路径或类文件对象）。典型选择是让 `obj` 为字典，但你也可以使用任意格式，只要之后能加载回来。
  该函数接受以下参数：
  - `model: torch.nn.Module`
  - `optimizer: torch.optim.Optimizer`
  - `iteration: int`
  - `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
- `def load_checkpoint(src, model, optimizer)` 应从 `src`（路径或类文件对象）加载检查点，然后从中恢复模型与优化器状态。函数应返回保存进检查点的迭代数。你可以用 `torch.load(src)` 取回你在 `save_checkpoint` 中保存的内容，并用模型与优化器的 `load_state_dict` 方法把它们恢复到先前状态。
  该函数接受以下参数：
  - `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
  - `model: torch.nn.Module`
  - `optimizer: torch.optim.Optimizer`

实现 `[adapters.run_save_checkpoint]` 与 `[adapters.run_load_checkpoint]`，并确保它们通过 `uv run pytest -k test_checkpointing`。

### 5.3 训练循环

现在终于是把你实现的所有组件拼进主训练脚本的时候了。把"用不同超参数启动训练"变得容易（例如通过命令行参数传入）会有回报，因为后文你会反复这样做，以研究不同选择对训练的影响。

**问题（training_together）：把它拼起来（4 分）**

**交付物：** 写一个脚本，运行训练循环以在用户提供的输入上训练你的模型。具体而言，我们建议你的训练脚本至少支持以下内容：

- 能配置并控制各种模型与优化器超参数。
- 用 `np.memmap` 以内存高效的方式加载大型训练与验证数据集。
- 把检查点序列化到用户提供的路径。
- 定期记录训练与验证性能（例如输出到控制台，和／或 Weights and Biases 之类的外部服务）⁹。

> ⁹ wandb.ai

---

## 6 生成文本

现在我们已经能训练模型，最后还需要的能力是从模型生成文本。回顾一下：语言模型接受一个长度为 `sequence_length` 的（可能批处理的）整数序列，产出一个大小为 `(sequence_length, vocab_size)` 的矩阵，其中序列的每个元素都是预测该位置之后下一个 token 的概率分布。下面我们写几个函数，把它变成对新序列的采样方案。

**Softmax**

按标准约定，语言模型的输出是最后一层线性层的输出（"logits"），因此我们必须通过 softmax 运算把它变成归一化概率——我们已在第 10 式中见过。

**解码**

要从模型生成文本（解码），我们给模型提供一段前缀 token 序列（"prompt"），请它产出词表上的概率分布，用于预测序列中的下一个 token；然后我们从该词表分布中采样，确定下一个输出 token。

具体而言，解码过程的一步应接受序列 $x_{1\dots t}$，按下式返回 token $x_{t+1}$：

$$P(x_{t+1} = i \mid x_{1\dots t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)}, \tag{21}$$

$$v = \mathrm{TransformerLM}(x_{1\dots t})_t \in \mathbb{R}^{vocab\_size}, \tag{22}$$

其中 `TransformerLM` 是我们的模型：输入长度为 `sequence_length` 的序列、产出大小为 `(sequence_length, vocab_size)` 的矩阵，而我们取该矩阵的**最后一个元素**，因为我们要的是第 $t$ 个位置上的下一 token 预测。

如此重复地从这些一步条件分布中采样（把上次生成的输出 token 追加到下一次解码时间步的输入中），直到生成序列结束 token `<|endoftext|>`（或达到用户指定的最大生成 token 数），就得到了一个基本解码器。

**解码器技巧**

我们将用小型模型做实验，而小模型有时会生成质量很低的文本。两个简单的解码技巧有助于改善这些问题。第一是**温度缩放**（temperature scaling）：用温度参数 $\tau$ 修改我们的 softmax，新的 softmax 为

$$\mathrm{softmax}(v, \tau)_i = \frac{\exp(v_i/\tau)}{\sum_{j=1}^{vocab\_size} \exp(v_j/\tau)}. \tag{23}$$

注意当 $\tau \to 0$ 时，$v$ 的最大元素占据主导，softmax 的输出变成集中在该最大元素上的 one-hot 向量。

第二是**核采样（nucleus sampling）**或 **top-p 采样**：通过截断低概率 token 来修改采样分布。设 $q$ 为由（温度缩放后的）softmax 得到的、大小为 `vocab_size` 的概率分布。超参数为 $p$ 的核采样按如下公式产生下一个 token：

$$P(x_{t+1} = i \mid q) = \begin{cases} \dfrac{q_i}{\sum_{j \in V(p)} q_j} & \text{若 } i \in V(p) \\[6pt] 0 & \text{其他} \end{cases} \tag{24}$$

其中 $V(p)$ 是使 $\sum_{j \in V(p)} q_j \ge p$ 成立的**最小**下标集合。计算这个量很容易：先把概率分布 $q$ 按大小排序，然后选取最大的若干词表元素，直到达到目标水平 $p$。

**问题（decoding）：解码（3 分）**

**交付物：** 实现一个从你的语言模型解码的函数。我们建议你支持以下特性：

- 对用户提供的 prompt 生成续写（即接受若干 $x_{1\dots t}$，采样续写直到遇到 `<|endoftext|>` token）。
- 允许用户控制生成的 token 数上限。
- 给定期望的温度值，在采样前对预测的下一 token 分布施加 softmax 温度缩放。
- 给定用户指定的阈值，做 top-$p$ 采样（[A. Holtzman et al., 2020]，亦称核采样）。

---

## 7 实验

现在是时候把所有东西拼起来，在一个预训练数据集上训练（小）语言模型了。

### 7.1 如何跑实验与交付物

理解 Transformer 各架构组件背后道理的最好方式，就是亲自修改它并运行它。动手经验无可替代。

为此，能够**快速**、**一致**地做实验，并**记录**你所做的事，就非常重要。为了快速实验，我们会在一个小规模模型（总共约 17M 参数）与简单数据集（TinyStories）上做许多实验。为了一致，你要系统地消融组件、变化超参数；为了记录，我们会要求你提交实验日志以及每个实验对应的学习曲线。

为便于提交损失曲线，请确保定期评估验证损失，并同时记录**步数**与**实际耗时（wall-clock time）**。你可能会发现 Weights and Biases 之类的日志基础设施很有帮助。

**问题（experiment_log）：实验记录（3 分）**

为你的训练与评估代码建立实验跟踪基础设施，使你能针对梯度步数与实际耗时追踪实验与损失曲线。

**交付物：** 你实验所用的日志基础设施代码，以及一份实验日志（记录你尝试过的所有事情的文档），覆盖本节下列各题。

### 7.2 TinyStories

我们将从一个非常简单的数据集开始（TinyStories；R. Eldan et al. [1]），模型在其中训练很快，我们也能看到一些有趣的行为。获取该数据集的说明见第 1 节。下面是该数据集的一个示例。

> **示例（tinystories_example）：TinyStories 中的一个示例**
>
> Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed! He said, "Wow, that is a really amazing vase! Can I buy it?" The shopkeeper smiled and said, "Of course you can. You can take it home and show all your friends how amazing it is!" So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was. And that's how Ben found an amazing vase in the store!

#### 7.2.1 超参数调优

我们会告诉你一些非常基础的起始超参数，并请你自己找出其他超参数的良好取值。

- **词表大小 10000**。典型词表大小在数万到数十万之间。你应当变化它，看看词表与模型行为如何变化。
- **上下文长度 256**。像 TinyStories 这样简单的数据集可能不需要很长的序列长度，但面对后面的 OpenWebText 数据，你可能想改变它。试着变化它，观察对每次迭代运行时间与最终困惑度的影响。
- **d_model 512**。这比许多小型 Transformer 论文所用的 768 维略小，但能让速度更快。
- **d_ff 1344**。这大致是 $\frac{8}{3}d_{model}$，同时又是 64 的倍数，对 GPU 性能友好。
- **RoPE 的 theta 参数 $\Theta$ 取 10000**。
- **层数与头数**：4 层、16 头。合起来给出约 17M 非嵌入参数，是一个相当小的 Transformer。
- **处理的总 token 数 327,680,000**（你的"批大小 × 总步数 × 上下文长度"应大致等于该值）。

你应当通过一些试错，为下列其他超参数找到良好的默认值：学习率、学习率预热、其他 AdamW 超参数（$\beta_1$、$\beta_2$、$\varepsilon$）以及权重衰减。你可以在 D. P. Kingma et al. [22] 中找到这类超参数的典型选择。

#### 7.2.2 把它拼起来

现在你可以把一切拼起来：拿到训练好的 BPE 分词器、把训练数据集分词，然后放进你写的训练循环中运行。重要提醒：如果你的实现正确且高效，上述超参数在一块 B200 GPU 上的运行时间应大致为 20–30 分钟。如果你的运行时间明显更长，请检查并确认你的数据加载、检查点保存或验证损失代码没有成为瓶颈，并确认你的实现得到了恰当的批处理。

#### 7.2.3 调试模型架构的技巧

我们强烈建议你熟悉 IDE 内置的调试器（例如 VSCode/Zed），与用 print 语句调试相比它会节省你的时间。如果你使用文本编辑器，可以用 ipdb 之类工具。调试模型架构时另有几条良好实践：

- 开发任何神经网络架构时，常见的第一个步骤是**在单个 minibatch 上过拟合**。如果你的实现正确，应能迅速把训练损失压到接近零。
- 在模型各组件处设置调试断点，检查中间张量的形状是否符合你的预期。
- 监控激活值、模型权重与梯度的范数，确认它们没有爆炸或消失。

**问题（learning_rate）：调学习率（2 B200 小时）（3 分）**

学习率是最需要调优的超参数之一。以你训练好的基础模型为基础，回答下列问题：

(a) 对学习率做一次超参数扫描，报告最终损失（若优化器发散，则注明发散）。
**交付物：** 多个学习率对应的学习曲线。解释你的超参数搜索策略。
**交付物：** 一个在 TinyStories 上验证损失（按 token 计）不超过 1.45 的模型。

> **低资源提示：在 CPU 或 Apple Silicon 上只训几步**
> 若你在 cpu 或 mps 上运行，应改为把处理的总 token 数降到 40,000,000，这已足以生成相当流畅的文本。你也可以把目标验证损失从 1.45 放宽到 2.00。
> 我们的解答代码在调好学习率后，在 M4 Max 芯片（36 GB 内存）上使用 "批大小 × 总步数 × 上下文长度 = 32×5000×256 = 40,960,000" 个 token，在 cpu 上耗时 1 小时 22 分钟、在 mps 上耗时 36 分钟；在第 5000 步，我们取得 1.80 的验证损失。
>
> 一些额外提示：
> - 使用 $N$ 个训练步时，我们建议把余弦学习率衰减调度调整为**恰好在第 $N$ 步**结束衰减（即到达最小学习率）。
> - 使用 mps 时，不要启用 TF32 内核，即不要像在 cuda 设备上那样设置 `torch.set_float32_matmul_precision('high')`。我们在 mps 上尝试启用 TF32 内核（torch 版本 2.9.0），发现后端有时会静默使用有问题的内核，导致训练不稳定。
> - 你可以用 `torch.compile` 对模型做 JIT 编译来加速训练。具体来说：
>   - 在 cpu 上，用 `model = torch.compile(model)` 编译模型。
>   - 在 mps 上，可用 `model = torch.compile(model, backend="aot_eager")` 在一定程度上优化反向传播。
>   - 截至 torch 2.9.0，mps 上不支持用 Inductor 做编译。

(b) 经验之谈是：最好的学习率"位于稳定性的边缘（at the edge of stability）"。研究学习率发散的分界点与你的最优学习率之间的关系。
**交付物：** 学习率递增的学习曲线，其中至少包含一次发散运行，并分析这与收敛速度的关系。

现在让我们变化批大小，看看训练会怎样。批大小很重要——它通过做更大的矩阵乘法让我们从 GPU 获得更高效率，但我们真的总是希望批大小越大越好吗？让我们做些实验来弄清楚。

**问题（batch_size_experiment）：批大小变化（1 B200 小时）（1 分）**

把批大小从 1 一直变化到 GPU 内存上限。中间至少试几个批大小，包括 64 与 128 这类典型值。

**交付物：** 不同批大小运行的训练曲线。如有必要，学习率应重新调优。
**交付物：** 用几句话讨论你关于批大小及其对训练影响的发现。

有了手中的解码器，我们现在可以生成文本了！我们将从模型生成并看看效果如何。作为参考，你应当得到看起来至少与下例一样好的输出。

> **示例（ts_generate_example）：TinyStories 语言模型的样例输出**
>
> Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big black one. One day, Lily's mom asked her to help cook dinner. Lily was so excited! She loved to help her mom. Lily's mom made a big pot of soup for dinner. Lily was so happy and said, "Thank you, Mommy! I love you." She helped her mom pour the soup into a big bowl. After dinner, Lily's mom made some yummy soup. Lily loved it! She said, "Thank you, Mommy! This soup is so yummy!" Her mom smiled and said, "I'm glad you like it, Lily." They finished cooking and continued to cook together. The end.

> **低资源提示：在 CPU 或 Apple Silicon 上生成文本**
> 如果你改用了只处理 40M token 的低资源配置，你应看到仍然像英语、但不如上面流畅的生成结果。例如，我们用一个在 40M token 上训练的 TinyStories 语言模型得到的样例输出如下：
>
> Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much. It was his best head. One day, Sue went for a walk and met a ladybug! They became good friends and played on the path together.
> "Hey, Polly! Let's go out!" said Tim. Sue looked at the sky and saw that it was difficult to find a way to dance shining. She smiled and agreed to help the talking!"
> As Sue watched the sky moved, what it was. She

下面是精确的题目陈述与我们要求的内容：

**问题（generate）：生成文本（1 分）**

用你的解码器与你训练好的检查点，报告你的模型生成的文本。你可能需要调整解码参数（temperature、top-p 等）以获得流畅的输出。

**交付物：** 至少 256 个 token 的文本转储（或直到第一个 `<|endoftext|>` token），并简要评论该输出的流畅度，以及至少两个影响该输出好坏的因子。

### 7.3 消融与架构修改

理解 Transformer 的最好方式就是真的修改它、观察它的行为。下面我们做几个简单的消融与修改。

#### 消融 1：层归一化

人们常说层归一化对 Transformer 训练的稳定性至关重要。但也许我们想"危险地活着"。让我们把 RMSNorm 从每个 Transformer 块中移除，看看会发生什么。

**问题（layer_norm_ablation）：移除 RMSNorm 并训练（0.5 B200 小时）（1 分）**

从你的 Transformer 中移除所有 RMSNorm 并训练。在先前的最优学习率下会发生什么？你能通过使用更低的学习率获得稳定性吗？

**交付物：** 移除 RMSNorm 后训练的学习曲线，以及最优学习率下的学习曲线。
**交付物：** 几句关于 RMSNorm 影响的评论。

下面我们来考察另一个乍看很随意的层归一化选择。pre-norm Transformer 块定义为

$$z = x + \mathrm{MultiHeadSelfAttention}(\mathrm{RMSNorm}(x)) \tag{25}$$

$$y = z + \mathrm{FFN}(\mathrm{RMSNorm}(z)). \tag{26}$$

这是对原始 Transformer 架构少数几项"共识性"修改之一，原始架构采用 post-norm 写法：

$$z = \mathrm{RMSNorm}(x + \mathrm{MultiHeadSelfAttention}(x)) \tag{27}$$

$$y = \mathrm{RMSNorm}(z + \mathrm{FFN}(z)). \tag{28}$$

让我们改回 post-norm 做法，看看会发生什么。

**问题（pre_norm_ablation）：实现 post-norm 并训练（0.5 B200 小时）（1 分）**

把你的 pre-norm Transformer 实现改成 post-norm 版本。用 post-norm 模型训练，看看会发生什么。

**交付物：** post-norm Transformer 的学习曲线，与 pre-norm 的对照。

我们会看到，层归一化对 Transformer 的行为有重大影响，甚至层归一化所处的**位置**也很重要。

#### 消融 2：位置嵌入

接下来我们考察位置嵌入对模型性能的影响。具体而言，我们要把基础模型（带 RoPE）与完全不加入位置嵌入（NoPE）做对比。事实证明，decoder-only Transformer（即像我们实现的这样带因果掩码的模型）在理论上无需显式提供位置嵌入就能推断相对或绝对位置信息 [Y.-H. H. Tsai et al., 2019; A. Kazemnejad et al., 2023]。现在我们将用实验检验 NoPE 相对于 RoPE 的表现。

**问题（no_pos_emb）：实现 NoPE（0.5 B200 小时）（1 分）**

修改你带 RoPE 的 Transformer 实现，完全移除位置嵌入信息，看看会发生什么。

**交付物：** 比较 RoPE 与 NoPE 性能的学习曲线。

#### 消融 3：SwiGLU 与 SiLU

接下来，我们依循 N. Shazeer [20]，通过比较 SwiGLU 前馈网络与使用 SiLU 激活但没有门控线性单元（GLU）的前馈网络，来检验前馈网络中门控的重要性：

$$\mathrm{FFN}_{\mathrm{SiLU}}(x) = W_2\,\mathrm{SiLU}(W_1 x). \tag{29}$$

回顾一下，在我们的 SwiGLU 实现中，内层前馈层的维度设为约 $d_{ff} = \frac{8}{3}d_{model}$（同时确保 $d_{ff} \bmod 64 = 0$，以便利用 GPU 张量核心）。在这个消融基线中，你的 $\mathrm{FFN}_{\mathrm{SiLU}}$ 实现应改为取 $d_{ff} = 4 \times d_{model}$，以大致匹配默认 SwiGLU 前馈网络的参数量（后者有三个而不是两个权重矩阵）。

**问题（swiglu_ablation）：SwiGLU 与 SiLU（0.5 B200 小时）（1 分）**

**交付物：** 在参数量大致匹配的前提下，比较 SwiGLU 与 SiLU 前馈网络性能的学习曲线。
**交付物：** 几句讨论你的发现。

> **低资源提示：GPU 资源有限的线上学生应在 TinyStories 上测试修改**
> 在作业余下部分，我们将转向一个规模更大、噪声更多的网络数据集（OpenWebText），做架构修改实验，并（可选地）向课程排行榜提交结果。
> 在 OpenWebText 上把 LM 训练到流畅耗时很长，因此我们建议 GPU 访问受限的线上学生继续在 TinyStories 上测试修改（用验证损失作为评估性能的指标）。

### 7.4 在 OpenWebText 上运行

现在我们将转向一个由网页抓取构建的、更标准的预训练数据集。OpenWebText [A. Gokaslan et al., 2019] 的一小份样本同样以单个文本文件提供：如何获取见第 1 节。

下面是 OpenWebText 的一个示例。注意其文本要真实、复杂、多样得多。你也许想浏览训练数据集，感受一下网页抓取语料的训练数据是什么样子。

> **示例（owt_example）：OWT 中的一个示例**
>
> Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.
>
> Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, "no corner of American culture is more precisely counted, more passionately quantified, than performances of baseball players." With a few clicks here and there, you can find out that Noah Syndergaard's fastball revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the game's highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that seem ripped from a video game or science fiction novel. The rising ocean of data has empowered an increasingly important actor in baseball's culture: the analytical hobbyist.
>
> That empowerment comes with added scrutiny – on the measurements, but also on the people and publications behind them. With Baseball Prospectus, Pavlidis knew all about the backlash that accompanies quantitative imperfection. He also knew the site's catching metrics needed to be reworked, and that it would take a learned mind – someone who could tackle complex statistical modeling problems – to complete the job.
>
> "He freaks us out." Harry Pavlidis
>
> Pavlidis had a hunch that Judge "got it" based on the latter's writing and their interaction at a site-sponsored ballpark event. [...]

注意：这个实验可能需要你重新调优学习率、批大小等超参数。

**问题（main_experiment）：在 OWT 上做实验（2 B200 小时）（2 分）**

用与 TinyStories 相同的模型架构与总训练迭代数，在 OpenWebText 上训练你的语言模型。该模型表现如何？

**交付物：** 你的语言模型在 OpenWebText 上的学习曲线。描述与 TinyStories 相比损失的差异——这些损失应如何解读？
**交付物：** OpenWebText LM 的生成文本，格式与 TinyStories 输出相同。这段文本的流畅度如何？在模型与计算预算都与 TinyStories 相同的情况下，为什么输出质量更差？

### 7.5 你自己的改动 + 排行榜

恭喜你走到这一步。你马上就要完成了！现在你要尝试改进 Transformer 架构，并看看你的超参数与架构在全班同学中处于什么水平。

**排行榜规则**

除以下几条外没有其他限制：

- **运行时间**：你的提交在 B200 上最多可运行 45 分钟。如果你使用 SLURM 或 Modal，可能想在提交脚本里强制这一限制。
- **数据**：你只能使用我们提供的 OpenWebText 训练数据集。

除此之外，你想做什么都可以。

如果你在寻找实现思路，可以看看这些资源：

- 最先进的开源 LLM 系列，例如 Llama 3 [A. Grattafiori et al., 2024] 或 Qwen 2.5 [A. Yang et al., 2024]。
- NanoGPT speedrun 仓库（`github.com/KellerJordan/modded-nanogpt`），社区成员在其中发布了许多用于"速通"小规模语言模型预训练的改动。例如，一个可追溯到原始 Transformer 论文的常见改动是把输入嵌入与输出嵌入的权重绑定在一起（见 A. Vaswani et al. [8]（第 3.4 节）与 A. Chowdhery et al. [16]（第 2 节））。如果你确实尝试权重绑定，可能需要调低嵌入／LM head 初始化的标准差。

在尝试完整的 45 分钟运行之前，你会想先在 OpenWebText 的一小部分或 TinyStories 上测试这些改动。

作为提示，我们确实注意到：你在本排行榜上可能发现有效的一些改动，未必能推广到更大规模的预训练。我们会在课程的缩放定律（scaling laws）单元进一步探讨这一想法。

**问题（leaderboard）：排行榜（10 B200 小时）（6 分）**

你将在上述排行榜规则下训练一个模型，目标是在 0.75 B200 小时内最小化语言模型的验证损失。

**交付物：** 最终记录的验证损失、一条对应的学习曲线（x 轴明确为实际耗时，且小于 45 分钟），以及对你所做工作的描述。我们期望一次排行榜提交至少能超过 5.0 损失这一朴素基线。在此提交到排行榜：`github.com/stanford-cs336/assignment1-basics-leaderboard`。

---

## 参考文献

（保留原文，未作翻译。）

[1] R. Eldan and Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?." 2023.

[2] A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, "OpenWebText corpus." 2019.

[3] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in Proc. of ACL, 2016.

[4] C. Wang, K. Cho, and J. Gu, "Neural Machine Translation with Byte-Level Subwords." 2019.

[5] P. Gage, "A new algorithm for data compression," C Users Journal, vol. 12, no. 2, pp. 23–38, Feb. 1994.

[6] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language Models are Unsupervised Multitask Learners." 2019.

[7] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, "Improving Language Understanding by Generative Pre-Training." 2018.

[8] A. Vaswani et al., "Attention is All you Need," in Proc. of NeurIPS, 2017.

[9] T. Q. Nguyen and J. Salazar, "Transformers without Tears: Improving the Normalization of Self-Attention," in Proc. of IWSWLT, 2019.

[10] R. Xiong et al., "On Layer Normalization in the Transformer Architecture," in Proc. of ICML, 2020.

[11] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization." 2016.

[12] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models." 2023.

[13] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in Proc. of NeurIPS, 2019.

[14] A. Grattafiori et al., "The Llama 3 Herd of Models." https://arxiv.org/abs/2407.21783

[15] A. Yang et al., "Qwen2.5 Technical Report," arXiv preprint arXiv:2412.15115, 2024.

[16] A. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways." 2022.

[17] D. Hendrycks and K. Gimpel, "Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units." 2016.

[18] S. Elfwing, U. Uchibe, and K. Doya, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning." https://arxiv.org/abs/1702.03118

[19] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, "Language Modeling with Gated Convolutional Networks." https://arxiv.org/abs/1612.08083

[20] N. Shazeer, "GLU Variants Improve Transformer." 2020.

[21] J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.

[22] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in Proc. of ICLR, 2015.

[23] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in Proc. of ICLR, 2019.

[24] T. B. Brown et al., "Language Models are Few-Shot Learners," in Proc. of NeurIPS, 2020.

[25] J. Kaplan et al., "Scaling Laws for Neural Language Models." 2020.

[26] J. Hoffmann et al., "Training Compute-Optimal Large Language Models." 2022.

[27] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The Curious Case of Neural Text Degeneration," in Proc. of ICLR, 2020.

[28] Y.-H. H. Tsai, S. Bai, M. Yamada, L.-P. Morency, and R. Salakhutdinov, "Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China, 2019, pp. 4344–4353. doi: 10.18653/v1/D19-1443.

[29] A. Kazemnejad, I. Padhi, K. Natesan, P. Das, and S. Reddy, "The Impact of Positional Encoding on Length Generalization in Transformers," in Thirty-seventh Conference on Neural Information Processing Systems, 2023. https://openreview.net/forum?id=Drrl2gcjzl

---

*译文到此结束。全篇对应讲义 §1–§7 与参考文献，共三个文件：*
*- `cs336-a1-part1-2-zh.md`（§1–§2）*
*- `cs336-a1-part3-zh.md`（§3）*
*- `cs336-a1-part4-7-zh.md`（§4–§7）*
