"""把数据集编码成 uint16 token ID 数组（供 LM 训练用）。

编码 TinyStories（train + valid）和 OpenWebText（train + valid）。
OpenWebText 需等 data/owt_vocab.json 训练完成后才能编码，脚本会自动跳过。
"""
import os

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def encode_to_npy(tokenizer: Tokenizer, input_path: str, output_path: str) -> None:
    # 用 encode_iterable 流式编码，避免把整个文件读进内存（train 集有 2.1G / 11G）
    with open(input_path, encoding="utf-8") as f:
        arr = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
    np.save(output_path, arr)
    print(f"{output_path}: {len(arr)} tokens")


def main() -> None:
    # TinyStories 分词器
    ts = Tokenizer.from_files(
        "data/tinystories_vocab.json",
        "data/tinystories_merges.json",
        special_tokens=["<|endoftext|>"],
    )
    encode_to_npy(ts, "data/TinyStoriesV2-GPT4-valid.txt", "data/tinystories_valid.npy")
    encode_to_npy(ts, "data/TinyStoriesV2-GPT4-train.txt", "data/tinystories_train.npy")

    # OpenWebText 分词器（训练完成后 data/owt_vocab.json 才存在）
    owt_vocab = "data/owt_vocab.json"
    owt_merges = "data/owt_merges.json"
    if os.path.exists(owt_vocab) and os.path.exists(owt_merges):
        owt = Tokenizer.from_files(owt_vocab, owt_merges)
        encode_to_npy(owt, "data/owt_valid.txt", "data/owt_valid.npy")
        encode_to_npy(owt, "data/owt_train.txt", "data/owt_train.npy")
    else:
        print("OpenWebText 分词器未就绪，跳过（等 data/owt_vocab.json 生成后再跑）")


if __name__ == "__main__":
    main()
