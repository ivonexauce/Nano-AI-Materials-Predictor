"""
pooling.py — Global Graph Pooling Strategies
Implements various pooling methods for aggregating node features
into a single graph-level representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def global_mean_pool(x: Tensor, batch: Tensor) -> Tensor:
    sizes = batch.unique(return_counts=True)[1]
    return torch.stack([x[batch == i].mean(0) for i in range(len(sizes))])


def global_add_pool(x: Tensor, batch: Tensor) -> Tensor:
    sizes = batch.unique(return_counts=True)[1]
    return torch.stack([x[batch == i].sum(0) for i in range(len(sizes))])


def global_max_pool(x: Tensor, batch: Tensor) -> Tensor:
    sizes = batch.unique(return_counts=True)[1]
    return torch.stack([x[batch == i].max(0)[0] for i in range(len(sizes))])


class AttentionPooling(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.attn = nn.Linear(in_channels, 1)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        scores = torch.softmax(self.attn(x), dim=0)
        return torch.stack([(scores[batch == i] * x[batch == i]).sum(0) for i in range(len(batch.unique()))])


class Set2SetPooling(nn.Module):
    def __init__(self, in_channels: int, processing_steps: int = 3):
        super().__init__()
        self.LSTM = nn.LSTMCell(in_channels * 2, in_channels)
        self.processing_steps = processing_steps
        self.linear = nn.Linear(in_channels * 2, in_channels)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        unique_batches = batch.unique()
        pooled = []

        for b in unique_batches:
            mask = batch == b
            nodes = x[mask]
            n_nodes = nodes.size(0)

            q_star = torch.zeros(1, self.LSTM.hidden_size, device=x.device)
            h = torch.zeros(1, self.LSTM.hidden_size, device=x.device)
            c = torch.zeros(1, self.LSTM.hidden_size, device=x.device)

            for _ in range(self.processing_steps):
                q_expanded = q_star.expand(n_nodes, -1)
                attn_scores = torch.softmax(
                    torch.mm(torch.cat([nodes, q_expanded], dim=1), self.linear.weight.t()),
                    dim=0,
                )
                r = (attn_scores * nodes).sum(0, keepdim=True)
                h, c = self.LSTM(torch.cat([q_star, r], dim=1), (h, c))
                q_star = h

            pooled.append(q_star.squeeze(0))

        return torch.stack(pooled)


def pool_by_crystal_idx(atom_fea: Tensor, crystal_atom_idx: list, mode: str = "mean") -> Tensor:
    if mode == "mean":
        return torch.stack([atom_fea[idx].mean(0) for idx in crystal_atom_idx])
    elif mode == "sum":
        return torch.stack([atom_fea[idx].sum(0) for idx in crystal_atom_idx])
    elif mode == "max":
        return torch.stack([atom_fea[idx].max(0)[0] for idx in crystal_atom_idx])
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")


if __name__ == "__main__":
    N, C, B = 20, 64, 4
    x = torch.randn(N, C)
    batch = torch.repeat_interleave(torch.arange(B), N // B)

    for name, fn in [("mean", global_mean_pool), ("sum", global_add_pool), ("max", global_max_pool)]:
        out = fn(x, batch)
        print(f"Global {name} pool: {out.shape}")

    attn = AttentionPooling(C)
    out = attn(x, batch)
    print(f"Attention pool: {out.shape}")
