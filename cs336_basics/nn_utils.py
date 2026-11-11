import math
import torch

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
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1)/ math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return attn @ V


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) :
    logits_max = logits.max(dim=-1, keepdim=True).values
    logits_shifted = logits - logits_max

    log_sum_exp = torch.log(
        torch.exp(logits_shifted).sum(dim=-1, keepdim=True)
    )
    log_probs = logits_shifted.gather(-1, targets.unsqueeze(-1))

    loss = (log_sum_exp - log_probs).mean()

    return loss