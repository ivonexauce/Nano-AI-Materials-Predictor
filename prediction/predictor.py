"""
predictor.py — Inference on New Materials
Loads a trained model and predicts properties for new crystal structures.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.periodic_table import encode_atom
from prediction.trainer import encode_bond_distance


class MaterialPredictor:
    def __init__(self, checkpoint_path: str):
        from models.cgcnn import build_model

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for prediction")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = build_model()
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.target_mean = ckpt.get("target_mean", 0.0)
        self.target_std = ckpt.get("target_std", 1.0)
        print(f"[✓] Loaded model from {checkpoint_path}")

    def predict(self, material: dict) -> dict:
        sites = material["sites"]
        n_sites = len(sites)

        atom_fea = torch.tensor(
            [encode_atom(s["species"]) for s in sites], dtype=torch.float32
        )

        n_neighbors = 12
        nbr_fea_list, nbr_idx_list = [], []
        for i, si in enumerate(sites):
            neighbors = []
            for j, sj in enumerate(sites):
                if i == j:
                    continue
                dx = [si["xyz"][k] - sj["xyz"][k] for k in range(3)]
                dist = math.sqrt(sum(d ** 2 for d in dx))
                neighbors.append((dist, j))
            neighbors.sort(key=lambda x: x[0])
            neighbors = neighbors[:n_neighbors]
            while len(neighbors) < n_neighbors:
                neighbors.append((3.0, 0))
            nbr_fea_list.append([encode_bond_distance(d) for d, _ in neighbors])
            nbr_idx_list.append([j for _, j in neighbors])

        nbr_fea = torch.tensor(nbr_fea_list, dtype=torch.float32)
        nbr_idx = torch.tensor(nbr_idx_list, dtype=torch.long)

        with torch.no_grad():
            pred = self.model(
                atom_fea,
                nbr_fea,
                nbr_idx,
                [list(range(n_sites))],
            )

        pred_val = pred.item() * self.target_std + self.target_mean
        return {"prediction": round(pred_val, 4), "formula": material.get("formula", "unknown")}

    def predict_batch(self, materials: list) -> list:
        return [self.predict(m) for m in materials]


def predict_from_formula(formula: str) -> dict:
    from data.fetcher import generate_synthetic_samples
    samples = generate_synthetic_samples(1, save_to_disk=False)
    samples[0]["formula"] = formula
    ckpt_path = "checkpoints/best_bandgap_ev.pt"
    if not Path(ckpt_path).exists():
        return {"error": "No trained model found. Run trainer.py first."}
    predictor = MaterialPredictor(ckpt_path)
    return predictor.predict(samples[0])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--formula", type=str, help="Material formula (e.g., TiO2)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_bandgap_ev.pt")
    parser.add_argument("--input", type=str, help="JSON file with materials to predict")
    args = parser.parse_args()

    if args.formula:
        result = predict_from_formula(args.formula)
        print(json.dumps(result, indent=2))
    elif args.input:
        with open(args.input) as f:
            data = json.load(f)
        materials = data.get("materials", data) if isinstance(data, dict) else data
        predictor = MaterialPredictor(args.checkpoint)
        results = predictor.predict_batch(materials)
        print(json.dumps(results, indent=2))
    else:
        print("Provide --formula or --input")
