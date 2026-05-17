"""
evaluator.py — MAE, RMSE, R² Metrics
Evaluates model predictions against ground truth DFT values.
"""

import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metrics import compute_all_metrics, mean_absolute_error, root_mean_squared_error, r2_score


class ModelEvaluator:
    def __init__(self, model=None, target_mean: float = 0.0, target_std: float = 1.0):
        self.model = model
        self.target_mean = target_mean
        self.target_std = target_std

    def evaluate(self, dataset, labels):
        import torch
        self.model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for i in range(len(dataset)):
                atom_fea, nbr_fea, nbr_idx = dataset[i]
                pred = self.model(
                    atom_fea, nbr_fea, nbr_idx,
                    [list(range(len(atom_fea)))],
                )
                pred_denorm = pred.item() * self.target_std + self.target_mean
                preds.append(pred_denorm)
                targets.append(labels[i])

        y_true = np.array(targets)
        y_pred = np.array(preds)
        return compute_all_metrics(y_true, y_pred)

    def print_report(self, metrics: dict):
        print("=" * 50)
        print("  Model Evaluation Report")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k:20s}: {v:.4f}")
        print("=" * 50)


def evaluate_checkpoint(checkpoint_path: str, dataset, labels):
    import torch
    import sys
    sys.path.insert(0, str(Path(checkpoint_path).parent.parent))
    from models.cgcnn import build_model

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    target_mean = ckpt.get("target_mean", 0.0)
    target_std = ckpt.get("target_std", 1.0)

    evaluator = ModelEvaluator(model, target_mean, target_std)
    metrics = evaluator.evaluate(dataset, labels)
    evaluator.print_report(metrics)
    return metrics


if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from prediction.trainer import build_dataset_from_json

    dataset, labels = build_dataset_from_json("data/sample_materials.json")
    if dataset and labels:
        ckpt_path = "checkpoints/best_bandgap_ev.pt"
        import os
        if os.path.exists(ckpt_path):
            evaluate_checkpoint(ckpt_path, dataset, labels)
        else:
            print(f"No checkpoint found at {ckpt_path}. Train first.")
