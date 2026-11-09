---
title: "CS336 Assignment 1 (basics): Writeup"
author: "awu"
date: "2026-11-04"
---

<!--
编译方式（任选其一，在 notes/writeup/ 目录下执行）：

  1) pandoc + typst（讲义 26.0.0 起官方用 Typst）：
     pandoc writeup.md -o writeup.pdf --pdf-engine=typst -V mainfont="Songti SC"

  2) pandoc + xelatex：
     pandoc writeup.md -o writeup.pdf --pdf-engine=xelatex \
       -V mainfont="Times New Roman" -V CJKmainfont="PingFang SC" -V geometry:margin=1in

  3) 直接写 Typst / LaTeX 也可以，本文件只是结构参考。

图片放在 ./figs/ 下，引用写 ![](./figs/xxx.png){width=80%}
注意：本目录已在 .gitignore 之外，提交前把 notes/ 加入 make_submission.sh 的 -x 列表。
-->

# 使用说明

- 每题后面括号内是分值。
- **Deliverable** 行是评分依据，逐条对照，别多写也别漏写。
- 标 `[代码]` 的题只需通过 `uv run pytest -k <name>`，writeup 里不必贴实现代码；但建议在对应小节留一行命令与结果，便于自查。
- 标 `[书面]` 的题必须出现在本 PDF 中。
- 标 `[实验]` 的题需要学习曲线图 + 文字分析。
- 讲义要求 "one-sentence response" / "one-to-two sentence response" 的，就真的只写一两句。

---

# 1 Assignment Overview

本节无书面题。

---

# 2 Byte-Pair Encoding (BPE) Tokenizer

## 2.1 The Unicode Standard

### unicode1 — Understanding Unicode（1 分）[书面]

**(a)** `chr(0)` 返回什么字符？
> 答：Unicode 字符 U+0000（NULL，空字符），通常表示为 \x00

**(b)** 其 `__repr__()` 与打印表示有何不同？
> 答：`__repr__()` 返回的是对应的 可见的转义序列  \x00

**(c)** 该字符出现在文本中会发生什么？
> 答：按理说这个符号是文档结束的标记，在C语言中字符串输出到这里就会停止，但是在py中并没有停止
>

## 2.2 Unicode Encodings

### unicode2 — Unicode Encodings（3 分）[书面]

**(a)** 为何优先用 UTF-8 而非 UTF-16/UTF-32 训练分词器？
> 答：英文和代码的存储效率更高,避免大量零字节,与现有文本生态兼容

**(b)** `decode_utf8_bytes_to_str_wrong` 错在哪里？给出一个错误输入。
> 错误输入：`b'...'`（必须写成可复现的字面量）
> 原因：因为uft-8编码不是每个词对应一个token，这个函数对逐字节解码会遇到错误；比如说 “你好”

**(c)** 给出一个无法解码为任何 Unicode 字符的两字节序列。
> 示例：\xff\xff

## 2.3 Subword Tokenization

无书面题。

## 2.4 BPE Tokenizer Training

无书面题（`bpe_example` 为讲义示例，不需要作答）。

## 2.5 Experimenting with BPE Tokenizer Training

### train_bpe — BPE Tokenizer Training（15 分）[代码]

测试：`uv run pytest tests/test_train_bpe.py`
> 结果：（通过 / 未通过，耗时）

### train_bpe_tinystories — BPE on TinyStories（2 分）[实验 + 书面]

**(a)** 训练耗时与内存？词表中最长的 token？是否合理？
> 答：（一到两句话 + 关键数字：训练时间、峰值内存、最长 token 及其字节内容）

**(b)** Profiling：训练中最耗时的部分？
> 答：（一到两句话，附 profiler 结论）
>
> 图（可选）：![](./figs/bpe_profile.png){width=70%}

### train_bpe_expts_owt — BPE on OpenWebText（2 分）[实验 + 书面]

**(a)** 词表中最长的 token？
> 答：（一到两句话）

**(b)** 对比 TinyStories 与 OpenWebText 的分词器。
> 答：（一到两句话）

## 2.6 BPE Tokenizer: Encoding and Decoding

### tokenizer — Implementing the Tokenizer（15 分）[代码]

测试：`uv run pytest tests/test_tokenizer.py`
> 结果：

## 2.7 Experiments

### tokenizer_experiments（4 分）[实验 + 书面]

**(a)** 各分词器的压缩率（bytes/token）？
> 答：（TS-10K: ___ ；OWT-32K: ___，一到两句话）

**(b)** 用 TinyStories 分词器编码 OWT 样本会怎样？
> 答：（一到两句话，含压缩率对比）

**(c)** 吞吐量（bytes/s）？分词 Pile（825GB）需多久？
> 答：（一到两句话，含实测吞吐量与推算时长）

**(d)** 为什么用 `uint16`？
> 答：（一到两句话）

---

# 3 Transformer Language Model Architecture

## 3.1 / 3.2 说明性章节

无书面题（`einstein_example1-3` 为示例，不需作答）。

## 3.3 Basic Building Blocks

### linear — Implementing the Linear Module（1 分）[代码]
> 测试：`uv run pytest -k test_linear` → 

### embedding — Implementing the Embedding Module（1 分）[代码]
> 测试：`uv run pytest -k test_embedding` → 

## 3.4 Pre-Norm Transformer Block

### rmsnorm（1 分）[代码]
> 测试：`uv run pytest -k test_rmsnorm` → 

### positionwise_feedforward（2 分）[代码]
> 测试：`uv run pytest -k test_swiglu` → 

### rope（2 分）[代码]
> 测试：`uv run pytest -k test_rope` → 

### softmax（1 分）[代码]
> 测试：`uv run pytest -k test_softmax_matches_pytorch` → 

### scaled_dot_product_attention（5 分）[代码]
> 测试：`uv run pytest -k test_scaled_dot_product_attention` / `-k test_4d_scaled_dot_product_attention` → 

### multihead_self_attention（5 分）[代码]
> 测试：`uv run pytest -k test_multihead_self_attention` → 

## 3.5 The Full Transformer LM

### transformer_block（3 分）[代码]
> 测试：`uv run pytest -k test_transformer_block` → 

### transformer_lm（3 分）[代码]
> 测试：`uv run pytest -k test_transformer_lm` → 

### transformer_accounting（5 分）[书面]

**(a)** GPT-2 XL 配置下的参数量？fp32 加载需多少内存？
> 答：（一到两句话 + 参数量与 GiB）

**(b)** 前向所需的全部矩阵乘（含描述）与总 FLOPs。
> 矩阵乘清单：
>
> | 组件 | 矩阵乘 | 形状 | FLOPs |
> |---|---|---|---|
> | Token embedding | — | — | — |
> | QKV 投影 | | | |
> | QKᵀ | | | |
> | softmax·V | | | |
> | 输出投影 | | | |
> | SwiGLU W1/W3 | | | |
> | SwiGLU W2 | | | |
> | LM head | | | |
> | **合计** | | | |
>
> 总计：___ FLOPs

**(c)** 哪部分 FLOPs 最多？
> 答：（一到两句话）

**(d)** GPT-2 small / medium / large 的分解，及随规模变化的趋势。
> 表：
>
> | 模型 | layers | d_model | heads | 注意力占比 | FFN 占比 | LM head 占比 | 其他 |
> |---|---|---|---|---|---|---|---|
> | small | 12 | 768 | 12 | | | | |
> | medium | 24 | 1024 | 16 | | | | |
> | large | 36 | 1280 | 20 | | | | |
> | XL | 48 | 1600 | 25 | | | | |
>
> 分析：（一到两句话）

**(e)** 上下文长度增至 16384 时，总 FLOPs 与各组件占比如何变化？
> 答：（一到两句话 + 关键数字）

---

# 4 Training a Transformer LM

## 4.1 Cross-entropy Loss

### cross_entropy（1 分）[代码]
> 测试：`uv run pytest -k test_cross_entropy` → 

## 4.2 The SGD Optimizer

### learning_rate_tuning（1 分）[实验 + 书面]

lr = 1e1 / 1e2 / 1e3，跑 10 步的 loss 行为？
> 答：（一到两句话，列出三者的 loss 变化趋势：衰减更快 / 更慢 / 发散）
>
> 图（可选）：![](./figs/sgd_lr.png){width=70%}

## 4.3 AdamW

### adamw（2 分）[代码]
> 测试：`uv run pytest -k test_adamw` → 

### adamw_accounting（2 分）[书面]

**(a)** 峰值内存分解（parameters / activations / gradients / optimizer state）。
> 参数：
> 激活：
> 梯度：
> 优化器状态：
> 总计：
>
> 推导说明：（必要时 1–2 段）

**(b)** GPT-2 XL 下化为 `a·batch_size + b`，80GB 内最大 batch size？
> 表达式：___ · batch_size + ___
> 最大 batch size：___

**(c)** AdamW 单步 FLOPs？
> 答：（表达式 + 一句理由）

**(d)** 50% MFU、单卡 H100、400K steps、batch 1024 下 GPT-2 XL 训练时长。
> 答：（小时数 + 简要推导）

## 4.4 Learning Rate Scheduling

### learning_rate_schedule（1 分）[代码]
> 测试：`uv run pytest -k test_get_lr_cosine_schedule` → 

## 4.5 Gradient Clipping

### gradient_clipping（1 分）[代码]
> 测试：`uv run pytest -k test_gradient_clipping` → 

---

# 5 Training Loop

## 5.1 Data Loader

### data_loading（2 分）[代码]
> 测试：`uv run pytest -k test_get_batch` → 

## 5.2 Checkpointing

### checkpointing（1 分）[代码]
> 测试：`uv run pytest -k test_checkpointing` → 

## 5.3 Training Loop

### training_together（4 分）[代码]

脚本位置：`cs336_basics/___`
> 支持的 CLI 参数：
> 结果：能正常收敛到 ___

---

# 6 Generating Text

### decoding（3 分）[代码]
> 支持：prompt 续写 / max_tokens / temperature / top-p

---

# 7 Experiments

## 7.1 How to Run Experiments and Deliverables

### experiment_log（3 分）[实验 + 书面]

> 日志存放位置：
> 实验记录表：
>
> | # | 日期 | 改动 | 关键超参 | steps | 最终 val loss | 备注 |
> |---|---|---|---|---|---|---|
> | 1 | | | | | | |

## 7.2 TinyStories

### 基准配置（自填，用于所有实验对照）

| 项 | 值 |
|---|---|
| vocab_size | 10000 |
| context_length | 256 |
| d_model | 512 |
| d_ff | 1344 |
| RoPE theta | 10000 |
| layers / heads | 4 / 16 |
| 总 token 数 | 327,680,000 |
| batch size × steps | |
| 学习率 | |
| warmup steps | |
| AdamW (β1, β2, ε) | |
| weight decay | |

### learning_rate（3 分）[实验 + 书面]

**(a)** 学习率扫描，最终 loss / 发散情况。
> 图：![](./figs/lr_sweep.png){width=80%}
> 搜索策略：（一两句话）
> 最佳 val loss：___（要求 ≤ 1.45）

**(b)** 最优学习率与发散边界的"稳定性边缘"关系。
> 图：![](./figs/lr_divergence.png){width=80%}
> 分析：（数句话）

### batch_size_experiment（1 分）[实验 + 书面]

> 图：![](./figs/batch_size.png){width=80%}
> 讨论：（数句话）

### generate（1 分）[实验 + 书面]

> 解码参数：（temperature / top-p / max_tokens）
> 生成文本（≥256 token 或至第一个 `<|endoftext|>`）：
> ```
> ...
> ```
> 评论：（流畅度评价 + 至少两个影响因素）

## 7.3 Ablations and Architecture Modification

### layer_norm_ablation（1 分）[实验 + 书面]
> 图：![](./figs/no_rmsnorm.png){width=80%}
> 分析：（数句话；含"原最优 lr 下表现"与"降 lr 是否稳定"）

### pre_norm_ablation（1 分）[实验 + 书面]
> 图：![](./figs/post_norm.png){width=80%}
> 分析：（数句话）

### no_pos_emb（1 分）[实验 + 书面]
> 图：![](./figs/rope_vs_nope.png){width=80%}
> 分析：（数句话）

### swiglu_ablation（1 分）[实验 + 书面]
> 图：![](./figs/swiglu_vs_silu.png){width=80%}
> 分析：（数句话；注意对照组 d_ff = 4×d_model 以对齐参数量）

## 7.4 Running on OpenWebText

### main_experiment（2 分）[实验 + 书面]

> 学习曲线：![](./figs/owt_loss.png){width=80%}
> 与 TinyStories 的 loss 差异及解释：（数句话）
> 生成文本：
> ```
> ...
> ```
> 输出质量更差的原因：（数句话）

## 7.5 Your own modification + leaderboard

### leaderboard（6 分）[实验 + 书面]

> 最终 val loss：___（要求 < 5.0）
> 学习曲线（x 轴为 wall-clock 时间，需明显 < 45 分钟）：![](./figs/leaderboard.png){width=80%}
> 所做改动：（描述）
> 消融/验证：说明该改动在小规模上的验证结果

---

# 附：结果汇总

| 实验 | val loss | 与基线差 | 备注 |
|---|---|---|---|
| baseline | | | |
| lr sweep 最优 | | | |
| batch size 最优 | | | |
| no RMSNorm | | | |
| post-norm | | | |
| NoPE | | | |
| SiLU (no GLU) | | | |
| OWT | | | |
| leaderboard | | | |
