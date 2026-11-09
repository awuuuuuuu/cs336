"""训练脚本共享的 vocab/merges 序列化辅助函数。"""
import json


def save_vocab(vocab: dict[int, bytes], path: str) -> None:
    """dict[int, bytes] -> {str(id): [int, ...]}"""
    with open(path, "w") as f:
        json.dump({str(i): list(b) for i, b in vocab.items()}, f)


def save_merges(merges: list[tuple[bytes, bytes]], path: str) -> None:
    """list[tuple[bytes, bytes]] -> [[[int, ...], [int, ...]], ...]"""
    with open(path, "w") as f:
        json.dump([[list(a), list(b)] for a, b in merges], f)
