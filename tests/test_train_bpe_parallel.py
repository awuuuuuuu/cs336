import time
from cs336_basics.bpe import pretokenize_file

if __name__ == "__main__":
    start = time.time()
    words = pretokenize_file("data/TinyStoriesV2-GPT4-valid.txt", ["<|endoftext|>"], num_processes=8)
    print(f"time: {time.time()-start:.1f}s  unique words: {len(words)}")
