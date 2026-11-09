"""在 OpenWebText 训练集上训练字节级 BPE 分词器，并序列化 vocab/merges 到磁盘。

OpenWebText 数据不含 <|endoftext|>，所以 special_tokens 传空列表。
合并循环会定期打印进度（verbose=True），方便重定向到日志后随时查看。

用法：
    uv run python scripts/train_owt.py
    nohup uv run python scripts/train_owt.py > data/owt_train.log 2>&1 &
"""
import resource
import sys
import time

from cs336_basics.bpe import train_bpe
from serialize import save_vocab, save_merges

INPUT_PATH = "data/owt_train.txt"
VOCAB_SIZE = 32_000
SPECIAL_TOKENS: list[str] = []
NUM_PROCESSES = 8
PROGRESS_INTERVAL = 1000   # 每合并 1000 次打印一次进度
VOCAB_OUT = "data/owt_vocab.json"
MERGES_OUT = "data/owt_merges.json"


def peak_memory_mb() -> float:
    """峰值 RSS（MB）。ru_maxrss 单位：macOS 是字节，Linux 是 KB。"""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def main() -> None:
    print(f"开始训练 OWT：vocab={VOCAB_SIZE}, processes={NUM_PROCESSES}", flush=True)
    print("（先并行预分词，再单线程合并；合并阶段每 1000 次打印一次进度）", flush=True)

    start = time.time()
    vocab, merges = train_bpe(
        INPUT_PATH,
        VOCAB_SIZE,
        SPECIAL_TOKENS,
        num_processes=NUM_PROCESSES,
        verbose=True,
        progress_interval=PROGRESS_INTERVAL,
    )
    elapsed = time.time() - start

    longest = max(vocab.values(), key=len)
    print(f"耗时: {elapsed:.1f}s", flush=True)
    print(f"峰值内存: {peak_memory_mb():.0f} MB", flush=True)
    print(f"词表大小: {len(vocab)}  合并数: {len(merges)}", flush=True)
    print(f"最长 token: {longest!r}  ({len(longest)} 字节)", flush=True)
    print(f"最长 token 解码: {longest.decode('utf-8', errors='replace')!r}", flush=True)

    save_vocab(vocab, VOCAB_OUT)
    save_merges(merges, MERGES_OUT)
    print(f"已保存: {VOCAB_OUT}, {MERGES_OUT}", flush=True)


if __name__ == "__main__":
    main()
