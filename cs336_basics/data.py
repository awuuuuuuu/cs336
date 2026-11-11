import torch
import numpy as np
from torch.utils.data import Dataset

def get_batch(
    dataset: Dataset,
    batch_size: int,
    context_length: int,
    device: torch.device | None = None,
):
    n = len(dataset)
    starting_indices = np.random.randint(0, n - context_length, batch_size)
    x = np.stack([dataset[i: i + context_length] for i in starting_indices])
    y = np.stack([dataset[i + 1: i + context_length + 1] for i in starting_indices])
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x, y

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]