"""
schnet.py — SchNet: Distance-based Continuous-filter Convolution
Reference: Schütt et al., NeurIPS (2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cgcnn import ATOMIC_FEATURE_DIM, BOND_FEATURE_DIM


class CFConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, nbr_fea_len: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(nbr_fea_len, 64),
            nn.Softplus(),
            nn.Linear(64, in_channels * out_channels),
        )
        self.dense = nn.Linear(in_channels, out_channels)

    def forward(self, atom_fea: Tensor, nbr_fea: Tensor, nbr_idx: Tensor) -> Tensor:
        N, M = nbr_idx.shape
        W = self.filter_net(nbr_fea)
        W = W.view(N, M, -1, self.dense.out_features)
        atom_nbr = atom_fea[nbr_idx, :]
        conv = torch.einsum("nmd, nmdf -> nmf", atom_nbr, W)
        conv = conv.sum(dim=1)
        out = self.dense(atom_fea) + conv
        return out


class SchNet(nn.Module):
    def __init__(
        self,
        orig_atom_fea_len: int = ATOMIC_FEATURE_DIM,
        nbr_fea_len: int = BOND_FEATURE_DIM,
        atom_fea_len: int = 64,
        n_interactions: int = 3,
        h_fea_len: int = 128,
        n_targets: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        self.cfconvs = nn.ModuleList([
            CFConv(atom_fea_len, atom_fea_len, nbr_fea_len) for _ in range(n_interactions)
        ])
        self.interaction_linear = nn.ModuleList([
            nn.Linear(atom_fea_len, atom_fea_len) for _ in range(n_interactions)
        ])

        self.fc_out = nn.Sequential(
            nn.Linear(atom_fea_len, h_fea_len),
            nn.Softplus(),
            nn.Dropout(0.1),
            nn.Linear(h_fea_len, n_targets),
        )

    def forward(self, atom_fea: Tensor, nbr_fea: Tensor, nbr_idx: Tensor, crystal_atom_idx: list) -> Tensor:
        atom_fea = self.embedding(atom_fea)

        for conv, linear in zip(self.cfconvs, self.interaction_linear):
            atom_fea = conv(atom_fea, nbr_fea, nbr_idx)
            atom_fea = F.softplus(atom_fea)
            atom_fea = linear(atom_fea)

        crys_fea = torch.stack([
            atom_fea[idx_map].mean(0) for idx_map in crystal_atom_idx
        ])
        return self.fc_out(crys_fea)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SchNet()
    print(f"[✓] SchNet initialized | Parameters: {model.count_parameters():,}")
    N, M = 20, 12
    B = 4
    atom_fea = torch.randn(N * B, ATOMIC_FEATURE_DIM)
    nbr_fea = torch.randn(N * B, M, BOND_FEATURE_DIM)
    nbr_idx = torch.randint(0, N * B, (N * B, M))
    crys_idx = [list(range(i * N, (i + 1) * N)) for i in range(B)]
    out = model(atom_fea, nbr_fea, nbr_idx, crys_idx)
    print(f"    Output shape: {out.shape}")
