"""
message_passing.py — Custom Message Passing Layer
Generic message-passing module for graph neural networks.
Provides a reusable MessagePassing base for GNN experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MessagePassingLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        self.message_net = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, out_channels),
            nn.Softplus(),
        )
        self.update_net = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Softplus(),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        src_features = x[src_idx]
        dst_features = x[dst_idx]
        messages = self.message_net(torch.cat([src_features, dst_features, edge_attr], dim=1))

        aggregated = torch.zeros(x.size(0), self.out_channels, device=x.device)
        aggregated.index_add_(0, dst_idx, messages)

        return self.update_net(torch.cat([x, aggregated], dim=1))


class EdgeConditionedMessagePassing(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge_channels: int):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(edge_channels, in_channels * out_channels),
        )
        self.gru = nn.GRUCell(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        weights = self.edge_net(edge_attr).view(-1, self.gru.input_size, self.gru.hidden_size)
        messages = (x[src].unsqueeze(1) @ weights).squeeze(1)
        aggregated = torch.zeros(x.size(0), self.gru.hidden_size, device=x.device)
        aggregated.index_add_(0, dst, messages)
        return self.gru(aggregated, x)


if __name__ == "__main__":
    N, E, C_in, C_out, EC = 10, 30, 64, 64, 41
    x = torch.randn(N, C_in)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, EC)

    layer = MessagePassingLayer(C_in, C_out, EC)
    out = layer(x, edge_index, edge_attr)
    print(f"MessagePassingLayer: {x.shape} -> {out.shape}")
