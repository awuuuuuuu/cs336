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
> 答：`__repr__()` 显示为可见的转义序列 `'\x00'`；`print` 出来则是空的（NULL 是不可见的控制字符，没有字形）。

**(c)** 该字符出现在文本中会发生什么？
> 答：NUL 打印时不可见，所以 `"this is a test" + chr(0) + "string"` 的 repr 是 `'this is a test\x00string'`，但 print 出来是 `this is a teststring`——两个词看起来像被拼接了，可字节 0 实际仍在字符串里。
>

## 2.2 Unicode Encodings

### unicode2 — Unicode Encodings（3 分）[书面]

**(a)** 为何优先用 UTF-8 而非 UTF-16/UTF-32 训练分词器？
> 答：英文和代码的存储效率更高,避免大量零字节,与现有文本生态兼容

**(b)** `decode_utf8_bytes_to_str_wrong` 错在哪里？给出一个错误输入。
> 错误输入：`"你好".encode("utf-8")` = `b'\xe4\xbd\xa0\xe5\xa5\xbd'`
> 原因：UTF-8 是变长编码，一个字符可能占 2–4 个字节。该函数逐个字节 `decode("utf-8")`，会把多字节字符拆开；单个字节 `\xe4`（前导字节）无法单独解码成合法字符，从而报错。

**(c)** 给出一个无法解码为任何 Unicode 字符的两字节序列。
> 示例：`b'\xff\xff'`（0xFF 不是合法的 UTF-8 字节，无法解码）

## 2.3 Subword Tokenization

无书面题。

## 2.4 BPE Tokenizer Training

无书面题（`bpe_example` 为讲义示例，不需要作答）。

## 2.5 Experimenting with BPE Tokenizer Training

### train_bpe — BPE Tokenizer Training（15 分）[代码]

测试：`uv run pytest tests/test_train_bpe.py`
> 结果：3 passed（test_train_bpe / test_train_bpe_speed / test_train_bpe_special_tokens）

### train_bpe_tinystories — BPE on TinyStories（2 分）[实验 + 书面]

**(a)** 训练耗时与内存？词表中最长的 token？是否合理？
> 答：训练耗时约 79.8s，峰值内存约 385 MB（并行预分词 + 增量合并 + 堆优化）。词表中最长 token 是 `' accomplishment'`（15 字节，带前导空格），合理——BPE 会把高频完整词收进词表。

**(b)** Profiling：训练中最耗时的部分？
> 答：优化前瓶颈是 `max(pair_counts)` 全表扫描（约 1.96 亿次 key 调用）；改用堆后瓶颈变为预分词；并行预分词后，剩余瓶颈是单线程的合并循环（BPE 合并无法并行化）。
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
> 结果：23 passed, 2 skipped（2 个内存测试因 macOS 不支持 rlimit 而跳过）

## 2.7 Experiments

### tokenizer_experiments（4 分）[实验 + 书面]

**(a)** 各分词器的压缩率（bytes/token）？
> 答：（TS-10K: 4.11 ；OWT-32K: ___，一到两句话）

**(b)** 用 TinyStories 分词器编码 OWT 样本会怎样？
> 答：压缩率会下降（bytes/token 变小）。因为 TinyStories 词表是在儿童故事上训练的，词汇覆盖窄；而 OpenWebText 是成人网页文本（词汇更丰富，含技术词、代码、专有名词等），大量词不在 TinyStories 词表里，分词器只能把它们拆成更小的子词甚至单个字节，导致 token 数变多、压缩率变低。

**(c)** 吞吐量（bytes/s）？分词 Pile（825GB）需多久？
> 答：吞吐约 1.6 MB/s。Pile 825GB = 8.25×10¹¹ 字节，需 8.25×10¹¹ / 1.6×10⁶ ≈ 147 小时。

**(d)** 为什么用 `uint16`？
> 答：因为词表大小 10000 < 65536 = 2¹⁶，token ID 能装进 uint16（范围 0–65535）；uint8（0–255）太小装不下，uint32（4 字节）则浪费一倍内存。所以 uint16（2 字节）是「刚好装得下、又最省内存」的选择

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
> 答：总参数量约 1.64B（1,640,452,800）：embedding 与 lm_head 各 80,411,200，每个 block 30,825,600（其中 FFN 20,582,400 占大头）。fp32（4 字节/参数）加载需约 6.56 GB（≈ 6.11 GiB）。

**(b)** 前向所需的全部矩阵乘（含描述）与总 FLOPs。
> 矩阵乘清单（记 batch=B, seq_len=S, d_model=d, d_ff=f, vocab=v, heads=h, d_k=d_v=d/h）：

```
1. embedding: 查表，非矩阵乘，0 FLOPs
2. 每个 TransformerBlock:
   MHA:
     - Q/K/V 投影: (B,S,d) × (d,d) ×3    → 3·2BSd²  = 6BSd²
     - Q @ K^T:    (B,h,S,d_k) × (d_k,S) → 2BhS²d_k = 2BS²d
     - Attn @ V:   (B,h,S,S) × (S,d_v)   → 2BhS²d_v = 2BS²d
     - 输出投影:   (B,S,d) × (d,d)       → 2BSd²
   FFN (SwiGLU):
     - W1/W2/W3:   ×3                    → 3·2BSdf  = 6BSdf
3. lm_head:       (B,S,d) × (d,v)        → 2BSdv
```
> 每 block = 8BSd² + 4BS²d + 6BSdf；总计 = (8BSd² + 4BS²d + 6BSdf)·num_layers + 2BSdv。
>
> GPT-2 XL（B=1, S=1024, d=1600, f=4288, v=50257, num_layers=48）：
> ① 投影 = 8×1×1024×1600² = 20,971,520,000；② 注意力 = 4×1×1024²×1600 = 6,710,886,400；③ FFN = 6×1×1024×1600×4288 = 42,152,755,200。
> 每 block = 69,835,161,600；48 块 = 3,352,087,756,800；LM head = 2×1024×1600×50257 = 164,682,137,600。
> 总计 = 3,516,769,894,400 ≈ 3.52 TFLOPs。

**(c)** 哪部分 FLOPs 最多？
> 答：FFN（SwiGLU 的 W1/W2/W3）最多（约 42.2 GFLOPs/block），其次是 QKV/output 投影（约 21.0 GFLOPs），注意力 QK^T+AV 最少（约 6.7 GFLOPs，因 S=1024 尚小）。

**(d)** GPT-2 small / medium / large 的分解，及随规模变化的趋势。
> 各组件 FLOPs 占比（B=1, S=1024, vocab=50257, d_ff=8/3·d_model 取 64 倍数）：
>
> | 模型 | layers | d_model | d_ff | 投影(QKV+out) | 注意力 | FFN | LM head |
> |---|---|---|---|---|---|---|---|
> | small | 12 | 768 | 2048 | 19.9% | 13.3% | 39.8% | 27.1% |
> | medium | 24 | 1024 | 2752 | 24.8% | 12.4% | 50.1% | 12.7% |
> | large | 36 | 1280 | 3392 | 27.3% | 10.9% | 54.3% | 7.5% |
> | XL | 48 | 1600 | 4288 | 28.6% | 9.2% | 57.5% | 4.7% |
>
> 分析：随 d_model 增大，∝d² 的项（FFN、投影）占比单调上升（FFN 39.8%→57.5%、投影 19.9%→28.6%），而 ∝d 的项（注意力、LM head）占比下降（注意力 13.3%→9.2%、LM head 27.1%→4.7%），因为 d² 增长快于 d。

**(e)** 上下文长度增至 16384 时，总 FLOPs 与各组件占比如何变化？
> 答：S 从 1024 增至 16384（16 倍）：投影/FFN/LM head 都 ∝ S 只增 16 倍，注意力 ∝ S² 增 256 倍，于是注意力从 9.2% 反超为 61.7%，FFN 从 57.5% 降至 24.2%。总 FLOPs 从 3.52 T 增至约 133.6 T（约 38 倍）。

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
