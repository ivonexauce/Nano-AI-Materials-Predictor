"""
test_predictor.py — Tests for material property prediction pipeline.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from prediction.trainer import encode_atom, encode_bond_distance, build_dataset_from_json


def test_encode_atom():
    vec = encode_atom("Si")
    assert len(vec) == 92
    assert sum(vec) == 1.0
    assert vec[13] == 1.0

    vec_o = encode_atom("O")
    assert vec_o[7] == 1.0

    vec_unknown = encode_atom("Xx")
    assert sum(vec_unknown) == 1.0
    print("  [✓] encode_atom: correct one-hot for known/unknown elements")


def test_encode_bond_distance():
    vec = encode_bond_distance(2.5)
    assert len(vec) == 41
    assert all(0 <= v <= 1 for v in vec)
    assert max(vec) > 0.9
    print("  [✓] encode_bond_distance: Gaussian basis correct")


def test_build_dataset():
    dataset, labels = build_dataset_from_json("data/sample_materials.json", "bandgap_ev")
    assert len(dataset) > 0
    assert len(dataset) == len(labels)

    atom_fea, nbr_fea, nbr_idx = dataset[0]
    assert isinstance(atom_fea, torch.Tensor)
    assert isinstance(nbr_fea, torch.Tensor)
    assert isinstance(nbr_idx, torch.Tensor)
    assert atom_fea.shape[1] == 92
    assert nbr_fea.shape[2] == 41
    print(f"  [✓] build_dataset: {len(dataset)} samples, "
          f"sample 0: {atom_fea.shape[0]} atoms, {nbr_fea.shape[1]} neighbors")


def test_training_history():
    history_path = Path("checkpoints/history_bandgap_ev.json")
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        assert "train_loss" in history
        assert "val_mae" in history
        assert len(history["train_loss"]) > 0
        print(f"  [✓] Training history: {len(history['train_loss'])} epochs")
    else:
        print("  [!] No training history found (train first)")


if __name__ == "__main__":
    test_encode_atom()
    test_encode_bond_distance()
    test_build_dataset()
    test_training_history()
    print("\n[✓] All predictor tests passed")
