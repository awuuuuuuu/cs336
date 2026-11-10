import math
import torch
from torch import Tensor
from jaxtyping import Bool, Float

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

    
def softmax(
    x: torch.Tensor,
    dim: int = -1
):
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max
    exp_x = x.exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1)/ math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return attn @ V