"""
dataset.py — PyTorch Geometric Dataset for Crystal Structures
Wraps material data into a torch-geometric InMemoryDataset for
standardized data loading during GNN training.
"""

import json
import math
from pathlib import Path
from typing import Optional, Callable

import numpy as np

try:
    import torch
    from torch_geometric.data import Data, InMemoryDataset, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    torch = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.periodic_table import ELEMENT_TO_IDX, ATOMIC_FEATURE_DIM, encode_atom

CUTOFF_RADIUS = 5.0
MAX_NEIGHBORS = 12

BOND_GAUSSIAN_CENTERS = np.linspace(0.5, 6.0, 41)
BOND_GAUSSIAN_WIDTH = 0.5


def encode_bond_distance(dist: float) -> list:
    return [
        math.exp(-((dist - mu) ** 2) / (2 * BOND_GAUSSIAN_WIDTH ** 2))
        for mu in BOND_GAUSSIAN_CENTERS
    ]


def material_to_data(material: dict) -> "Data":
    sites = material.get("sites", [])
    n_sites = len(sites)

    x = torch.tensor([encode_atom(s["species"]) for s in sites], dtype=torch.float32)

    edge_list, edge_attr = [], []
    for i in range(n_sites):
        for j in range(n_sites):
            if i == j:
                continue
            dx = [sites[i]["xyz"][k] - sites[j]["xyz"][k] for k in range(3)]
            dist = math.sqrt(sum(d ** 2 for d in dx))
            if dist <= CUTOFF_RADIUS:
                edge_list.append([i, j])
                edge_attr.append(encode_bond_distance(dist))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 1), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.zeros((1, len(BOND_GAUSSIAN_CENTERS)), dtype=torch.float32)

    y = {}
    for key in ["bandgap_ev", "formation_energy_ev_atom", "bulk_modulus"]:
        if key in material and material[key] is not None:
            y[key] = torch.tensor([[float(material[key])]], dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    for k, v in y.items():
        data[k] = v
    data.material_id = material.get("material_id", "")
    data.formula = material.get("formula", "")
    data.n_atoms = n_sites
    return data


class CrystalGraphDataset(InMemoryDataset):
    def __init__(
        self,
        json_path: str,
        target: str = "bandgap_ev",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.json_path = json_path
        self.target = target
        super().__init__(Path(json_path).parent, transform, pre_transform)
        if PYG_AVAILABLE:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            self.data, self.slices = None, None

    @property
    def raw_file_names(self):
        return [Path(self.json_path).name]

    @property
    def processed_file_names(self):
        return [f"dataset_{Path(self.json_path).stem}.pt"]

    def download(self):
        pass

    def process(self):
        if not PYG_AVAILABLE:
            print("[WARN] torch-geometric not installed. Skipping dataset processing.")
            return

        with open(self.json_path) as f:
            raw = json.load(f)
        materials = raw.get("materials", raw) if isinstance(raw, dict) else raw

        data_list = []
        for mat in materials:
            if mat.get(self.target) is None or not mat.get("sites"):
                continue
            data = material_to_data(mat)
            if data is not None:
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"[✓] Processed {len(data_list)} graphs -> {self.processed_paths[0]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_materials.json")
    parser.add_argument("--target", default="bandgap_ev")
    args = parser.parse_args()

    if not PYG_AVAILABLE:
        print("[WARN] torch-geometric required. pip install torch-geometric")
    else:
        dataset = CrystalGraphDataset(json_path=args.data, target=args.target)
        print(f"[✓] Dataset: {len(dataset)} samples")
        if len(dataset) > 0:
            print(f"    Sample: {dataset[0]}")
