"""
megnet.py — Multi-layer Edge Graph Network
Extends CGCNN by updating edge features during message passing.
Reference: Chen et al., Nat. Comput. Sci. (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cgcnn import ATOMIC_FEATURE_DIM, BOND_FEATURE_DIM


class EdgeUpdateBlock(nn.Module):
    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        self.edge_net = nn.Sequential(
            nn.Linear(2 * atom_fea_len + nbr_fea_len, 128),
            nn.Softplus(),
            nn.Linear(128, nbr_fea_len),
        )
        self.node_net = nn.Linear(atom_fea_len + nbr_fea_len, atom_fea_len)

    def forward(self, atom_fea: Tensor, nbr_fea: Tensor, nbr_idx: Tensor) -> tuple:
        N, M = nbr_idx.shape
        atom_nbr_fea = atom_fea[nbr_idx, :]

        edge_input = torch.cat([
            atom_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
            atom_nbr_fea,
            nbr_fea,
        ], dim=2)

        nbr_fea_update = self.edge_net(edge_input)
        nbr_fea = nbr_fea + nbr_fea_update

        nbr_summed = nbr_fea.sum(dim=1)
        node_input = torch.cat([atom_fea, nbr_summed], dim=1)
        atom_fea = atom_fea + self.node_net(node_input)

        return atom_fea, nbr_fea


class MEGNet(nn.Module):
    def __init__(
        self,
        orig_atom_fea_len: int = ATOMIC_FEATURE_DIM,
        nbr_fea_len: int = BOND_FEATURE_DIM,
        atom_fea_len: int = 64,
        n_blocks: int = 4,
        h_fea_len: int = 128,
        n_h: int = 1,
        n_targets: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.edge_embedding = nn.Linear(nbr_fea_len, nbr_fea_len)

        self.blocks = nn.ModuleList([
            EdgeUpdateBlock(atom_fea_len, nbr_fea_len) for _ in range(n_blocks)
        ])

        self.fc = nn.Sequential(
            nn.Linear(atom_fea_len, h_fea_len),
            nn.Softplus(),
            *[layer for _ in range(n_h - 1)
              for layer in [nn.Linear(h_fea_len, h_fea_len), nn.Softplus()]],
            nn.Dropout(0.1),
            nn.Linear(h_fea_len, n_targets),
        )

    def forward(self, atom_fea: Tensor, nbr_fea: Tensor, nbr_idx: Tensor, crystal_atom_idx: list) -> Tensor:
        atom_fea = self.embedding(atom_fea)
        nbr_fea = self.edge_embedding(nbr_fea)

        for block in self.blocks:
            atom_fea, nbr_fea = block(atom_fea, nbr_fea, nbr_idx)

        crys_fea = torch.stack([
            atom_fea[idx_map].mean(0) for idx_map in crystal_atom_idx
        ])
        return self.fc(crys_fea)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MEGNet()
    print(f"[✓] MEGNet initialized | Parameters: {model.count_parameters():,}")
    N, M = 20, 12
    B = 4
    atom_fea = torch.randn(N * B, ATOMIC_FEATURE_DIM)
    nbr_fea = torch.randn(N * B, M, BOND_FEATURE_DIM)
    nbr_idx = torch.randint(0, N * B, (N * B, M))
    crys_idx = [list(range(i * N, (i + 1) * N)) for i in range(B)]
    out = model(atom_fea, nbr_fea, nbr_idx, crys_idx)
    print(f"    Output shape: {out.shape}")
