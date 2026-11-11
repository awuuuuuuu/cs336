"""训练 Transformer 语言模型的主脚本。

把模型、优化器、数据加载、损失、学习率调度、梯度裁剪、检查点、评估全部拼起来。

用法示例（TinyStories，低资源配置 40M tokens = 32×5000×256）：
    uv run python scripts/run_training.py \
        --train_data data/tinystories_train.npy \
        --valid_data data/tinystories_valid.npy \
        --vocab_size 10000 --context_length 256 \
        --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
        --batch_size 32 --total_steps 5000 --warmup_steps 100 \
        --checkpoint_dir checkpoints
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer LM")
    # 数据
    p.add_argument("--train_data", type=str, required=True, help="训练集 .npy（token ID 数组）")
    p.add_argument("--valid_data", type=str, default=None, help="验证集 .npy（可选）")
    # 模型
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # 优化器
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=None, help="cosine 结束时的学习率，默认 lr/10")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    # 训练
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=10)
    p.add_argument("--checkpoint_interval", type=int, default=500)
    # 设备与输出
    p.add_argument("--device", type=str, default="cpu", help="cpu / mps / cuda:0")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None, help="从该检查点恢复")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def evaluate(model, dataset, batch_size, context_length, device, num_batches) -> tuple[float, float]:
    """在验证集上采样若干 batch，返回 (平均 loss, perplexity)。"""
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
    model.train()
    avg_loss = total_loss / num_batches
    return avg_loss, math.exp(avg_loss)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10

    # 1. 内存映射加载数据（大文件不进内存）
    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r") if args.valid_data else None
    assert train_data.max() < args.vocab_size, "训练数据含超出词表大小的 token ID"
    print(f"训练集: {len(train_data)} tokens, dtype={train_data.dtype}")

    # 2. 构建模型与优化器
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # 3. 可选：从检查点恢复
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"从检查点恢复，起始 step = {start_step}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 4. 训练循环
    train_loss_sum = 0.0
    t_start = time.time()
    for step in range(start_step, args.total_steps):
        # 学习率调度
        lr = get_lr_cosine_schedule(step, args.lr, min_lr, args.warmup_steps, args.total_steps)
        set_lr(optimizer, lr)

        # 前向 + 反向 + 裁剪 + 更新
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()
        train_loss_sum += loss.item()

        # 定期记录训练/验证性能
        if (step + 1) % args.eval_interval == 0:
            train_avg = train_loss_sum / args.eval_interval
            train_loss_sum = 0.0
            elapsed = time.time() - t_start
            msg = f"step {step+1}/{args.total_steps} | lr {lr:.2e} | train loss {train_avg:.4f} | train ppl {math.exp(train_avg):.1f}"
            if valid_data is not None:
                val_loss, val_ppl = evaluate(
                    model, valid_data, args.batch_size, args.context_length, args.device, args.eval_batches
                )
                msg += f" | val loss {val_loss:.4f} | val ppl {val_ppl:.1f}"
            msg += f" | {elapsed:.0f}s"
            print(msg, flush=True)

        # 定期保存检查点
        if (step + 1) % args.checkpoint_interval == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_{step+1}.pt"
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            print(f"已保存检查点: {ckpt_path}", flush=True)

    # 5. 最终检查点
    final_path = checkpoint_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, args.total_steps, final_path)
    print(f"训练完成，最终检查点: {final_path}")


if __name__ == "__main__":
    main()
