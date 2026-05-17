"""
test_model.py — Tests for GNN model forward passes and parameter counts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.cgcnn import CGCNN, ConvLayer, build_model, ATOMIC_FEATURE_DIM, BOND_FEATURE_DIM


def test_conv_layer():
    N, M = 10, 12
    atom_fea = torch.randn(N, 64)
    nbr_fea = torch.randn(N, M, 41)
    nbr_idx = torch.randint(0, N, (N, M))

    layer = ConvLayer(64, 41)
    out = layer(atom_fea, nbr_fea, nbr_idx)
    assert out.shape == (N, 64), f"Expected (10, 64), got {out.shape}"
    print(f"  [✓] ConvLayer: {atom_fea.shape} -> {out.shape}")


def test_cgcnn_forward():
    N, M, B = 20, 12, 4
    atom_fea = torch.randn(N * B, ATOMIC_FEATURE_DIM)
    nbr_fea = torch.randn(N * B, M, BOND_FEATURE_DIM)
    nbr_idx = torch.randint(0, N * B, (N * B, M))
    crystal_idx = [list(range(i * N, (i + 1) * N)) for i in range(B)]

    model = CGCNN()
    out = model(atom_fea, nbr_fea, nbr_idx, crystal_idx)
    assert out.shape == (B, 1), f"Expected (4, 1), got {out.shape}"
    print(f"  [✓] CGCNN forward: {out.shape}")
    print(f"  [✓] Parameters: {model.count_parameters():,}")


def test_build_model():
    model = build_model()
    assert isinstance(model, CGCNN)
    assert model.count_parameters() > 0
    print(f"  [✓] build_model() creates CGCNN with {model.count_parameters():,} params")


def test_multi_target():
    from models.cgcnn import MultiTargetCGCNN
    model = MultiTargetCGCNN()
    N, M, B = 20, 12, 4
    atom_fea = torch.randn(N * B, ATOMIC_FEATURE_DIM)
    nbr_fea = torch.randn(N * B, M, BOND_FEATURE_DIM)
    nbr_idx = torch.randint(0, N * B, (N * B, M))
    crystal_idx = [list(range(i * N, (i + 1) * N)) for i in range(B)]

    out = model(atom_fea, nbr_fea, nbr_idx, crystal_idx)
    assert out.shape == (B, 3), f"Expected (4, 3), got {out.shape}"

    named = model.forward_with_names(atom_fea, nbr_fea, nbr_idx, crystal_idx)
    assert isinstance(named, dict)
    assert "bandgap_ev" in named
    print(f"  [✓] MultiTargetCGCNN: {out.shape}, keys: {list(named.keys())}")


if __name__ == "__main__":
    test_conv_layer()
    test_cgcnn_forward()
    test_build_model()
    test_multi_target()
    print("\n[✓] All model tests passed")
