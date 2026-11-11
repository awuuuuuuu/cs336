import math
import torch
from typing import Any, Iterable, Optional, Callable


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ) -> None:
        default = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(params, default)

    def step(
        self,
        closure: Optional[Callable] = None
    ) :
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr: Any = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                
                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                denom = (
                    exp_avg_sq.sqrt()
                    / math.sqrt(bias_correction2)
                    + eps
                )

                step_size = lr / bias_correction1

                # AdamW：解耦的权重衰减
                p.data.mul_(1 - lr * weight_decay)

                # Adam 的梯度更新
                p.data.addcdiv_(
                    exp_avg,
                    denom,
                    value=-step_size
                )

        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    cosine_cycle_iters: int
) -> float:
    if it < warmup_steps:
        return max_learning_rate * it / warmup_steps
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_steps) / (cosine_cycle_iters - warmup_steps)
        return min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(
    parameters: Iterable[torch.Tensor],
    max_l2_norm: float
) :
    parameters = [p for p in parameters if p.grad is not None]

    total_norm = math.sqrt(sum(p.grad.data.norm() ** 2 for p in parameters))

    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    
