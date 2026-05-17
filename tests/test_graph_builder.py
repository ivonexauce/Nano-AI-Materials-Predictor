"""
test_graph_builder.py — Tests for graph construction from crystal structures.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from gnn.graph_builder import build_graph


def test_build_graph_from_synthetic():
    with open("data/sample_materials.json") as f:
        data = json.load(f)
    materials = data.get("materials", data) if isinstance(data, dict) else data

    assert len(materials) > 0, "No materials found"

    for mat in materials[:5]:
        graph = build_graph(mat)
        assert "atom_features" in graph
        assert "edge_index" in graph
        assert "edge_attr" in graph
        assert graph["n_atoms"] == len(mat["sites"])
        print(f"  [✓] {mat['formula']}: {graph['n_atoms']} atoms, "
              f"{graph['edge_index'].shape[1]} edges")


def test_graph_tensor_shapes():
    import torch
    with open("data/sample_materials.json") as f:
        data = json.load(f)
    materials = data.get("materials", data) if isinstance(data, dict) else data

    for mat in materials[:3]:
        graph = build_graph(mat)
        n_atoms = graph["n_atoms"]
        assert isinstance(graph["atom_features"], torch.Tensor)
        assert graph["atom_features"].shape[0] == n_atoms
        assert graph["atom_features"].shape[1] == 92


if __name__ == "__main__":
    test_build_graph_from_synthetic()
    test_graph_tensor_shapes()
    print("\n[✓] All graph builder tests passed")
