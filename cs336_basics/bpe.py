import heapq
import multiprocessing
import time

import regex as re
from collections import Counter, defaultdict

from cs336_basics.pretokenization_example import find_chunk_boundaries


def _pretokenize_file_chunk(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return pretokenize(chunk, special_tokens)


def pretokenize_file(
    input_path: str,
    special_tokens: list[str],
    num_processes: int = 8
) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "rb") as f:
        if special_tokens:
            token_bytes = special_tokens[0].encode("utf-8")
            boundaries = find_chunk_boundaries(f, num_processes, token_bytes)
        else:
            # 没有特殊 token 时按固定字节切块（允许在词中间切断，作业允许此边界情况）
            f.seek(0, 2)  # SEEK_END
            file_size = f.tell()
            chunk_size = file_size // num_processes
            boundaries = [i * chunk_size for i in range(num_processes)] + [file_size]

    chunks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    if num_processes == 1:
        results = [_pretokenize_file_chunk(*c) for c in chunks]
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(_pretokenize_file_chunk, chunks)

    counts = Counter()
    for result in results:
        for token, cnt in result.items():
            counts[token] += cnt
    return dict(counts)


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(
    text: str,
    special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    """
    预分词，返回词频字典
    """
    counter = Counter()

    if special_tokens:
        # 将特殊token从文本中分离出来
        sep= "|".join(
            re.escape(t) 
            for t in sorted(special_tokens, key=len, reverse=True)
        )
        parts = re.split(sep, text)
    else:
        parts = [text]

    for part in parts:
        for chunk in re.finditer(PAT, part):
            bytes1 =  chunk.group().encode("utf-8")
            token = tuple(bytes([b]) for b in bytes1)
            counter[token] += 1
    return dict(counter)

def merge_pair(
    word: tuple[bytes, ...],
    pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    result = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            result.append(word[i] + word[i + 1])
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)

def get_pairs(
    word: list[bytes]
):
    c = Counter()
    for a, b in zip(word, word[1:]):
        c[(a, b)] += 1
    return c

def negate(bs):
    return tuple(255 - b for b in bs) + (256,)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
    verbose: bool = False,
    progress_interval: int = 1000,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. 读文件 + 2. 预分词
    words = pretokenize_file(input_path, special_tokens, num_processes)
    
    word_list = [list(w) for w in words]
    word_counts = list(words.values())

    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for idx, word in enumerate(word_list):
        for a, b in zip(word, word[1:]):
            pair_counts[(a, b)] += word_counts[idx]
            pair_to_words.setdefault((a, b), set()).add(idx)

    heap = []
    for (a, b), cnt in pair_counts.items():
        heapq.heappush(heap, (-cnt, negate(a), negate(b), a, b))

    # 3. 初始化词表
    vocab = {i : bytes([i]) for i in range(256)}

    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        vocab[len(vocab)] = token_bytes

    #4. 合并
    merges = []
    num_merges = vocab_size - len(vocab)
    start_time = time.time()
    for merge_idx in range(num_merges):
        while True:
            neg_cnt, _, _, a, b = heapq.heappop(heap)
            pair = (a, b)
            if pair_counts[pair] == -neg_cnt and pair_counts[pair] > 0:
                break

        for idx in list(pair_to_words[pair]):
            word = word_list[idx]
            cnt = word_counts[idx]

            old = get_pairs(word)
            new_word = merge_pair(word, pair)
            new = get_pairs(new_word)
            
            for p, c in old.items():                    # 旧 pair 消失
                pair_counts[p] -= c * cnt
                pair_to_words[p].discard(idx)
                heapq.heappush(heap, (-pair_counts[p], negate(p[0]), negate(p[1]), p[0], p[1]))
            for p, c in new.items():                    # 新 pair 出现
                pair_counts[p] += c * cnt
                pair_to_words[p].add(idx)
                heapq.heappush(heap, (-pair_counts[p], negate(p[0]), negate(p[1]), p[0], p[1]))
            
            word_list[idx] = list(new_word)
        merges.append(pair)
        vocab[len(vocab)] = pair[0] + pair[1]
        if verbose and (merge_idx + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (merge_idx + 1) * (num_merges - merge_idx - 1)
            print(
                f"merge {merge_idx + 1}/{num_merges} "
                f"({(merge_idx + 1) / num_merges:.1%}) elapsed {elapsed:.0f}s eta {eta:.0f}s",
                flush=True,
            )

    return vocab, merges


    



