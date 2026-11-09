"""§2.7 分词器实验：压缩率 (a) + 吞吐量 (c)。

OpenWebText 部分（b 及 a 的 OWT 一半）需等 OWT 分词器训练完（data/owt_vocab.json 生成后）再补。
"""
import time

from cs336_basics.tokenizer import Tokenizer


def load_docs(path: str, sep: str, n: int = 10) -> list[str]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    docs = [d for d in text.split(sep) if d.strip()]
    return docs[:n]


def compression_rate(tokenizer: Tokenizer, docs: list[str]) -> float:
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tokenizer.encode(d)) for d in docs)
    return total_bytes / total_tokens


def main() -> None:
    ts = Tokenizer.from_files(
        "data/tinystories_vocab.json",
        "data/tinystories_merges.json",
        special_tokens=["<|endoftext|>"],
    )

    # (a) TinyStories 压缩率
    docs = load_docs("data/TinyStoriesV2-GPT4-valid.txt", "<|endoftext|>")
    rate = compression_rate(ts, docs)
    print(f"TinyStories 分词器 on TinyStories 样本: {rate:.2f} bytes/token")

    # (c) 吞吐量：编码一段文本测速
    sample = "<|endoftext|>".join(docs)
    nbytes = len(sample.encode("utf-8"))
    start = time.time()
    ts.encode(sample)
    elapsed = time.time() - start
    tput = nbytes / elapsed
    print(f"吞吐: {tput / 1e6:.1f} MB/s")
    print(f"Pile (825GB) 预计: {825e9 / tput / 3600:.1f} 小时")


if __name__ == "__main__":
    main()
