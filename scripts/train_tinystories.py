"""在 TinyStories 训练集上训练字节级 BPE 分词器，并序列化 vocab/merges 到磁盘。

用法：uv run python train_tinystories.py
"""
import resource
import time

from cs336_basics.bpe import train_bpe
from serialize import save_vocab, save_merges

INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]
NUM_PROCESSES = 8
VOCAB_OUT = "data/tinystories_vocab.json"
MERGES_OUT = "data/tinystories_merges.json"


def main() -> None:
    start = time.time()
    vocab, merges = train_bpe(
        INPUT_PATH, VOCAB_SIZE, SPECIAL_TOKENS, num_processes=NUM_PROCESSES
    )
    elapsed = time.time() - start

    longest = max(vocab.values(), key=len)
    # macOS 上 ru_maxrss 单位是字节（Linux 上是 KB）
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    print(f"耗时: {elapsed:.1f}s")
    print(f"峰值内存: {peak_mb:.0f} MB")
    print(f"词表大小: {len(vocab)}  合并数: {len(merges)}")
    print(f"最长 token: {longest!r}  ({len(longest)} 字节)")
    print(f"最长 token 解码: {longest.decode('utf-8', errors='replace')!r}")

    save_vocab(vocab, VOCAB_OUT)
    save_merges(merges, MERGES_OUT)
    print(f"已保存: {VOCAB_OUT}, {MERGES_OUT}")


if __name__ == "__main__":
    main()
