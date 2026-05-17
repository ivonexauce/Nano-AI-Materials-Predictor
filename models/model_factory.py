"""
model_factory.py — Model Selection and Initialization
Factory for creating CGCNN, MEGNet, or SchNet models with shared config.
"""

from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_REGISTRY = {}


def register_model(name: str):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(
    model_name: str = "cgcnn",
    n_targets: int = 1,
    atom_fea_len: int = 64,
    n_conv: int = 3,
    h_fea_len: int = 128,
    **kwargs,
):
    from models.cgcnn import CGCNN, ATOMIC_FEATURE_DIM, BOND_FEATURE_DIM

    if model_name == "cgcnn":
        model = CGCNN(
            orig_atom_fea_len=ATOMIC_FEATURE_DIM,
            nbr_fea_len=BOND_FEATURE_DIM,
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_targets=n_targets,
            **kwargs,
        )
    elif model_name == "megnet":
        from models.megnet import MEGNet
        model = MEGNet(
            orig_atom_fea_len=ATOMIC_FEATURE_DIM,
            nbr_fea_len=BOND_FEATURE_DIM,
            atom_fea_len=atom_fea_len,
            n_blocks=n_conv,
            h_fea_len=h_fea_len,
            n_targets=n_targets,
            **kwargs,
        )
    elif model_name == "schnet":
        from models.schnet import SchNet
        model = SchNet(
            orig_atom_fea_len=ATOMIC_FEATURE_DIM,
            nbr_fea_len=BOND_FEATURE_DIM,
            atom_fea_len=atom_fea_len,
            n_interactions=n_conv,
            h_fea_len=h_fea_len,
            n_targets=n_targets,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: cgcnn, megnet, schnet")

    print(f"[✓] {model_name.upper()} initialized | Parameters: {model.count_parameters():,}")
    return model


if __name__ == "__main__":
    for name in ["cgcnn", "megnet", "schnet"]:
        model = build_model(name)
        print(f"  {name}: {model.count_parameters():,} params")
