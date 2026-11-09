# 在 Linux 服务器上训练 OpenWebText BPE 分词器

本文档指导在一台 Linux 服务器上：拉取代码 → 下载数据 → 训练 OpenWebText（OWT）字节级 BPE 分词器（vocab 32000）。

## 资源要求

- **磁盘**：约 15 GB（OWT train 11GB + valid 277MB + 输出文件）
- **内存**：约 10 GB（训练中峰值约 9 GB）
- **时间**：讲义参考 ≤ 12 小时；本实现的合并部分是单线程，实际可能几小时到 1 天
- **GPU**：不需要（纯 CPU + 多进程预分词）

## 步骤 1：克隆代码

```sh
git clone https://github.com/awuuuuuuu/cs336.git
cd cs336
```

## 步骤 2：安装 uv（如果还没有）

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或：pip install uv
```

`uv` 会根据 `pyproject.toml` 自动创建环境并安装依赖（重点是 `regex` 包，BPE 预分词的正则 `\p{L}` 需要它）。

## 步骤 3：下载 OWT 数据

```sh
mkdir -p data
cd data
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
cd ..
```

## 步骤 4：后台启动训练

```sh
nohup uv run python scripts/train_owt.py > data/owt_train.log 2>&1 &
```

## 步骤 5：随时查看进度

```sh
tail -f data/owt_train.log
```

进度每 1000 次合并打印一行，形如：

```
merge 1000/31744 (3.2%) elapsed 120s eta 3688s
merge 2000/31744 (6.3%) elapsed 245s eta 3644s
```

## 输出

训练完成后会生成：

- `data/owt_vocab.json` — 词表，格式 `{"ID": [字节值, ...]}`
- `data/owt_merges.json` — 合并列表，格式 `[[[字节,...], [字节,...]], ...]`

日志末尾会打印：耗时、峰值内存、词表大小、合并数、最长 token。

## 关键说明

- OWT 数据**不含** `<|endoftext|>`，训练脚本里 `SPECIAL_TOKENS = []`（空列表）。
- 词表 32000 → 需要 **31744 次合并**（32000 − 256 字节）。
- 合并循环是**单线程**（BPE 合并无法并行化），是主要耗时部分；预分词用 8 进程并行。
- `scripts/train_owt.py` 里的关键参数：`VOCAB_SIZE=32000`、`NUM_PROCESSES=8`、`PROGRESS_INTERVAL=1000`。

## 训练完成后（可选）

用训练好的 OWT 分词器把数据编码成 uint16 数组（供 LM 训练用）：

```sh
uv run python scripts/encode_datasets.py
```

该脚本会自动检测 `data/owt_vocab.json` 是否存在，存在则编码 `owt_train.txt` / `owt_valid.txt`。
