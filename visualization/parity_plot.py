"""
parity_plot.py — Predicted vs Actual Plots
Generates parity plots comparing GNN predictions against DFT-calculated values.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metrics import compute_all_metrics


def set_style():
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150})


def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predictions vs DFT", save_path: Optional[str] = None):
    set_style()
    metrics = compute_all_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, s=30, c="steelblue", edgecolors="gray", linewidth=0.3)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, alpha=0.7, label="Ideal")

    text_str = f"MAE:  {metrics['mae']:.3f}\nRMSE: {metrics['rmse']:.3f}\nR²:   {metrics['r2']:.3f}"
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("DFT Values")
    ax.set_ylabel("GNN Predictions")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[✓] Saved: {save_path}")
    plt.show()

    return metrics


def parity_from_checkpoint(
    checkpoint_path: str,
    dataset,
    labels_true,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch required.")
        return None

    from models.cgcnn import build_model

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    target_mean = ckpt.get("target_mean", 0.0)
    target_std = ckpt.get("target_std", 1.0)

    preds = []
    with torch.no_grad():
        for i in range(len(dataset)):
            atom_fea, nbr_fea, nbr_idx = dataset[i]
            pred = model(atom_fea, nbr_fea, nbr_idx, [list(range(len(atom_fea)))])
            pred_denorm = pred.item() * target_std + target_mean
            preds.append(pred_denorm)

    y_true = np.array(labels_true)
    y_pred = np.array(preds)
    title = title or f"Parity Plot — {Path(checkpoint_path).stem}"

    return parity_plot(y_true, y_pred, title, save_path)


def parity_from_json(pred_json_path: str, actual_key: str = "bandgap_ev", pred_key: str = "prediction", save_path: Optional[str] = None):
    with open(pred_json_path) as f:
        data = json.load(f)

    y_true = np.array([m[actual_key] for m in data if actual_key in m and pred_key in m])
    y_pred = np.array([m[pred_key] for m in data if actual_key in m and pred_key in m])

    return parity_plot(y_true, y_pred, f"Parity Plot — {Path(pred_json_path).stem}", save_path)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_bandgap_ev.pt")
    parser.add_argument("--data", type=str, default="data/sample_materials.json")
    parser.add_argument("--target", type=str, default="bandgap_ev")
    parser.add_argument("--output", type=str, help="Save path")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from prediction.trainer import build_dataset_from_json

    dataset, labels = build_dataset_from_json(args.data, args.target)
    if dataset and len(dataset) > 0:
        metrics = parity_from_checkpoint(args.checkpoint, dataset, labels, save_path=args.output)
        if metrics:
            print(f"\nR² Score: {metrics['r2']:.4f}")
