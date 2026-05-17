"""
cgcnn.py — Crystal Graph Convolutional Neural Network
Predicts material properties from crystal structure graphs.

Reference: Xie & Grossman, PRL 2018 (https://arxiv.org/abs/1710.10324)
Adapted and extended by UMBA YANGA IVON EXAUCE — UMBA Consulting Engineers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

try:
    from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("[WARN] torch-geometric not installed. Run: pip install torch-geometric")


ATOMIC_FEATURE_DIM = 92   # One-hot encoded atomic number + properties
BOND_FEATURE_DIM = 41     # Bond distance encoded as Gaussian basis


class ConvLayer(nn.Module):
    """
    Single crystal graph convolutional layer.
    Aggregates neighbor atom + bond features to update atom representations.
    """

    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        self.fc_full = nn.Linear(
            2 * atom_fea_len + nbr_fea_len,
            2 * atom_fea_len
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea: Tensor, nbr_fea: Tensor, nbr_fea_idx: Tensor) -> Tensor:
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # (N, M, atom_fea_len)
        total_nbr_fea = torch.cat([
            atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
            atom_nbr_fea,
            nbr_fea,
        ], dim=2)  # (N, M, 2*atom_fea_len + nbr_fea_len)

        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network.
    Maps crystal structure graph → material property prediction.
    """

    def __init__(
        self,
        orig_atom_fea_len: int = ATOMIC_FEATURE_DIM,
        nbr_fea_len: int = BOND_FEATURE_DIM,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        n_targets: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)
        ])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        hidden_layers = []
        for _ in range(n_h - 1):
            hidden_layers += [nn.Linear(h_fea_len, h_fea_len), nn.Softplus()]
        self.fcs = nn.Sequential(*hidden_layers)
        self.fc_out = nn.Linear(h_fea_len, n_targets)
        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        atom_fea: Tensor,
        nbr_fea: Tensor,
        nbr_fea_idx: Tensor,
        crystal_atom_idx: list,
    ) -> Tensor:
        atom_fea = self.embedding(atom_fea)

        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        # Global pooling: mean over atoms in each crystal
        crys_fea = self._pool(atom_fea, crystal_atom_idx)

        crys_fea = self.conv_to_fc_softplus(self.conv_to_fc(crys_fea))
        crys_fea = self.dropout(crys_fea)
        crys_fea = self.fcs(crys_fea)
        out = self.fc_out(crys_fea)
        return out

    def _pool(self, atom_fea: Tensor, crystal_atom_idx: list) -> Tensor:
        """Mean pooling over atoms within each crystal."""
        return torch.stack([
            atom_fea[idx_map].mean(0) for idx_map in crystal_atom_idx
        ])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiTargetCGCNN(CGCNN):
    """
    Extends CGCNN to predict multiple properties simultaneously.
    Useful for joint bandgap + formation energy prediction.
    """

    TARGETS = ["bandgap_ev", "formation_energy_ev_atom", "bulk_modulus_gpa"]

    def __init__(self, **kwargs):
        kwargs["n_targets"] = len(self.TARGETS)
        super().__init__(**kwargs)

    def forward_with_names(self, *args) -> dict:
        out = self.forward(*args)
        return {name: out[:, i] for i, name in enumerate(self.TARGETS)}


def build_model(config: dict = None) -> CGCNN:
    """Factory function for CGCNN with sensible defaults."""
    cfg = config or {}
    model = CGCNN(
        orig_atom_fea_len=cfg.get("atom_fea_len_input", ATOMIC_FEATURE_DIM),
        nbr_fea_len=cfg.get("nbr_fea_len", BOND_FEATURE_DIM),
        atom_fea_len=cfg.get("atom_fea_len", 64),
        n_conv=cfg.get("n_conv", 3),
        h_fea_len=cfg.get("h_fea_len", 128),
        n_h=cfg.get("n_h", 1),
        n_targets=cfg.get("n_targets", 1),
    )
    print(f"[✓] CGCNN initialized | Parameters: {model.count_parameters():,}")
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("  CGCNN — Crystal Graph Convolutional Neural Network")
    print("  UMBA Nano-AI Materials Predictor")
    print("=" * 60)

    model = build_model()
    print(model)

    # Quick sanity check with random tensors
    N_atoms = 20
    N_neighbors = 12
    batch_size = 4

    atom_fea = torch.randn(N_atoms * batch_size, ATOMIC_FEATURE_DIM)
    nbr_fea = torch.randn(N_atoms * batch_size, N_neighbors, BOND_FEATURE_DIM)
    nbr_fea_idx = torch.randint(0, N_atoms * batch_size, (N_atoms * batch_size, N_neighbors))
    crystal_atom_idx = [
        list(range(i * N_atoms, (i + 1) * N_atoms)) for i in range(batch_size)
    ]

    out = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
    print(f"\n[✓] Forward pass OK | Output shape: {out.shape}")
    print(f"    Sample predictions: {out.detach().numpy().flatten().tolist()}")
