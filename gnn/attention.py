"""
attention.py — Graph Attention Mechanism
Implements GAT-style attention and edge-aware attention for crystal graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge_channels: int = 0, heads: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.head_dim = out_channels // heads
        assert out_channels % heads == 0, "out_channels must be divisible by heads"

        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a = nn.Parameter(torch.randn(heads, self.head_dim * 2 + edge_channels))
        self.leaky_relu = nn.LeakyReLU(0.2)

        if edge_channels > 0:
            self.edge_proj = nn.Linear(edge_channels, heads * self.head_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        H, D = self.heads, self.head_dim
        N = x.size(0)

        x_transformed = self.W(x).view(N, H, D)
        src_idx, dst_idx = edge_index[0], edge_index[1]

        src_features = x_transformed[src_idx]
        dst_features = x_transformed[dst_idx]
        attention_input = torch.cat([src_features, dst_features], dim=2)

        if edge_attr is not None and self.a.size(-1) > 2 * self.head_dim:
            attention_input = torch.cat([attention_input, edge_attr.unsqueeze(1).expand(-1, H, -1)], dim=2)

        e = self.leaky_relu(torch.einsum("ehd, hd -> eh", attention_input, self.a))

        attention = F.softmax(e, dim=0)
        weighted_features = (attention.unsqueeze(-1) * x_transformed[dst_idx]).sum(dim=1)

        if edge_attr is not None and hasattr(self, "edge_proj"):
            weighted_features = weighted_features + self.edge_proj(edge_attr.mean(dim=0, keepdim=True)).view(-1, H, D).sum(dim=1)

        return weighted_features


if __name__ == "__main__":
    N, E, C, EC = 10, 30, 64, 41
    x = torch.randn(N, C)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, EC)

    layer = GraphAttentionLayer(C, 64, EC, heads=4)
    out = layer(x, edge_index, edge_attr)
    print(f"GraphAttentionLayer: {x.shape} -> {out.shape}")
