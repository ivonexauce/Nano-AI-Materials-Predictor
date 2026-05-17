"""
graph_builder.py — Atom/Bond Graph Construction
Constructs graph representations from raw crystal structure data.
Converts atom coordinates and lattice parameters into node/edge features.
"""

import math
import numpy as np
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.periodic_table import encode_atom, ATOMIC_FEATURE_DIM
from prediction.trainer import encode_bond_distance

CUTOFF_RADIUS = 5.0
MAX_NEIGHBORS = 12


def build_graph(material: dict, cutoff: float = CUTOFF_RADIUS, max_neighbors: int = MAX_NEIGHBORS):
    sites = material["sites"]
    n_sites = len(sites)

    node_features = [encode_atom(s["species"]) for s in sites]

    edges = []
    edge_features = []
    for i in range(n_sites):
        neighbors = []
        for j in range(n_sites):
            if i == j:
                continue
            dx = [sites[i]["xyz"][k] - sites[j]["xyz"][k] for k in range(3)]
            dist = math.sqrt(sum(d ** 2 for d in dx))
            if dist <= cutoff:
                neighbors.append((dist, j))
        neighbors.sort(key=lambda x: x[0])
        neighbors = neighbors[:max_neighbors]
        for dist, j in neighbors:
            edges.append([i, j])
            edge_features.append(encode_bond_distance(dist))

    if TORCH_AVAILABLE:
        return {
            "atom_features": torch.tensor(node_features, dtype=torch.float32),
            "edge_index": torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 1), dtype=torch.long),
            "edge_attr": torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.zeros((1, 41), dtype=torch.float32),
            "n_atoms": n_sites,
        }
    return {
        "atom_features": np.array(node_features, dtype=np.float32),
        "edge_index": np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 1), dtype=np.int64),
        "edge_attr": np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((1, 41), dtype=np.float32),
        "n_atoms": n_sites,
    }


def build_batch_graph(materials: list, cutoff: float = CUTOFF_RADIUS, max_neighbors: int = MAX_NEIGHBORS):
    graphs = [build_graph(m, cutoff, max_neighbors) for m in materials]
    if not TORCH_AVAILABLE:
        return graphs

    from torch_geometric.data import Batch, Data
    data_list = []
    for g in graphs:
        d = Data(x=g["atom_features"], edge_index=g["edge_index"], edge_attr=g["edge_attr"])
        data_list.append(d)
    return Batch.from_data_list(data_list)


if __name__ == "__main__":
    import json
    with open("data/sample_materials.json") as f:
        data = json.load(f)
    mats = data.get("materials", data) if isinstance(data, dict) else data
    if mats:
        g = build_graph(mats[0])
        print(f"Material: {mats[0]['formula']} ({mats[0]['nsites']} atoms)")
        print(f"  Node features: {g['atom_features'].shape}")
        print(f"  Edge index: {g['edge_index'].shape}")
        print(f"  Edge features: {g['edge_attr'].shape}")
