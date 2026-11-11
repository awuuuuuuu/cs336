"""从语言模型生成文本的解码函数。"""

import torch
from torch import Tensor, nn

from .nn_utils import softmax


def top_p_filter(probs: Tensor, top_p: float = 1.0) -> Tensor:
    """Top-p（核）过滤：保留累积概率刚 >= top_p 的最小集合，其余置 0 并重新归一化。"""
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1].")
    if top_p == 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # 累积到前一个元素已 >= top_p 的位置（含自身及之后）都移除
    sorted_to_remove = cumulative_probs - sorted_probs >= top_p
    to_remove = torch.zeros_like(sorted_to_remove).scatter_(
        dim=-1, index=sorted_indices, src=sorted_to_remove
    )
    filtered = probs.masked_fill(to_remove, 0.0)
    return filtered / filtered.sum(dim=-1, keepdim=True)


def generate(
    model: nn.Module,
    token_ids: Tensor | list[int],
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token: int | None = None,
) -> list[int]:
    """从语言模型自回归生成 token 序列。

    Args:
        model: 语言模型，输入 (batch, seq) token id、输出 (batch, seq, vocab) logits。
        token_ids: prompt 的 token id（tensor 或 list）。
        max_tokens: 最多生成的 token 数。
        temperature: 温度缩放；0 表示 greedy（argmax）。
        top_p: 核采样阈值；1 表示不做 top-p 过滤。
        end_token: 若生成该 token 则提前停止（如 <|endoftext|> 的 id）。
    """
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    device = next(model.parameters()).device
    tokens = token_ids.tolist() if torch.is_tensor(token_ids) else list(token_ids)

    for _ in range(max_tokens):
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)[0, -1]  # 最后一个位置的下一词 logits (vocab,)

        if temperature == 0.0:
            token = torch.argmax(logits).item()  # greedy 解码
        else:
            probs = softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                probs = top_p_filter(probs, top_p)
            token = torch.multinomial(probs, num_samples=1).item()

        tokens.append(token)
        if end_token is not None and token == end_token:
            break

    return tokens
