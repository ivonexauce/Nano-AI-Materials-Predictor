"""
trainer.py — GNN Training Pipeline for Materials Property Prediction
Handles data loading, training loop, early stopping, and checkpointing.
"""

import json
import math
import time
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not installed.")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cgcnn import build_model, ATOMIC_FEATURE_DIM, BOND_FEATURE_DIM

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ─── Atomic feature encoding ─────────────────────────────────────────────────

ELEMENT_TO_IDX = {
    "H": 0, "He": 1, "Li": 2, "Be": 3, "B": 4, "C": 5, "N": 6, "O": 7,
    "F": 8, "Ne": 9, "Na": 10, "Mg": 11, "Al": 12, "Si": 13, "P": 14,
    "S": 15, "Cl": 16, "Ar": 17, "K": 18, "Ca": 19, "Ti": 20, "V": 21,
    "Cr": 22, "Mn": 23, "Fe": 24, "Co": 25, "Ni": 26, "Cu": 27, "Zn": 28,
    "Ga": 29, "Ge": 30, "As": 31, "Se": 32, "Br": 33, "Kr": 34, "Mo": 35,
    "Pd": 36, "Ag": 37, "In": 38, "Sn": 39, "Sb": 40, "Te": 41, "I": 42,
    "W": 43, "Pt": 44, "Au": 45, "Pb": 46, "Bi": 47, "Nb": 48, "Zr": 49,
}

def encode_atom(species: str) -> list:
    """One-hot encode atom species into a fixed-length feature vector."""
    vec = [0.0] * ATOMIC_FEATURE_DIM
    idx = ELEMENT_TO_IDX.get(species, len(ELEMENT_TO_IDX) % ATOMIC_FEATURE_DIM)
    vec[idx % ATOMIC_FEATURE_DIM] = 1.0
    return vec


def encode_bond_distance(distance: float, n_bins: int = BOND_FEATURE_DIM) -> list:
    """Gaussian basis expansion of interatomic distance."""
    centers = np.linspace(0.5, 6.0, n_bins)
    width = 0.5
    return [float(math.exp(-((distance - c) ** 2) / (2 * width ** 2))) for c in centers]


# ─── Dataset builder ──────────────────────────────────────────────────────────

def build_dataset_from_json(json_path: str, target: str = "bandgap_ev"):
    """Convert raw materials JSON into tensors for CGCNN training."""
    if not TORCH_AVAILABLE:
        return [], []

    with open(json_path) as f:
        raw = json.load(f)

    materials = raw.get("materials", raw) if isinstance(raw, dict) else raw
    valid_materials = [m for m in materials if m.get(target) is not None and m.get("sites")]

    print(f"[*] Building dataset | Target: {target} | Valid materials: {len(valid_materials)}")

    dataset = []
    labels = []

    for mat in valid_materials:
        sites = mat["sites"]
        n_sites = len(sites)

        # Atom features
        atom_fea = torch.tensor(
            [encode_atom(s["species"]) for s in sites], dtype=torch.float32
        )

        # Build neighbor list (simplified: all pairs within cutoff, max 12 neighbors)
        n_neighbors = 12
        nbr_fea_list = []
        nbr_idx_list = []

        for i, site_i in enumerate(sites):
            neighbors = []
            for j, site_j in enumerate(sites):
                if i == j:
                    continue
                dx = [site_i["xyz"][k] - site_j["xyz"][k] for k in range(3)]
                dist = math.sqrt(sum(d ** 2 for d in dx))
                neighbors.append((dist, j))
            neighbors.sort(key=lambda x: x[0])
            neighbors = neighbors[:n_neighbors]

            # Pad if fewer neighbors than expected
            while len(neighbors) < n_neighbors:
                neighbors.append((3.0, 0))  # dummy

            nbr_fea_list.append([encode_bond_distance(d) for d, _ in neighbors])
            nbr_idx_list.append([j for _, j in neighbors])

        nbr_fea = torch.tensor(nbr_fea_list, dtype=torch.float32)
        nbr_idx = torch.tensor(nbr_idx_list, dtype=torch.long)

        dataset.append((atom_fea, nbr_fea, nbr_idx))
        labels.append(float(mat[target]))

    print(f"[✓] Dataset built: {len(dataset)} samples")
    return dataset, labels


# ─── Training loop ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train(
    data_path: str,
    target: str = "bandgap_ev",
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_split: float = 0.1,
    patience: int = 25,
):
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch required for training.")
        return

    print("=" * 60)
    print("  CGCNN Training — UMBA Nano-AI Materials Predictor")
    print(f"  Target: {target} | Epochs: {epochs} | LR: {lr}")
    print("=" * 60)

    # Load data
    dataset, labels = build_dataset_from_json(data_path, target)
    if not dataset:
        print("[ERROR] No valid data found.")
        return

    # Train/val split
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * (1 - val_split))
    train_idx = indices[:split]
    val_idx = indices[split:]

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    target_mean = labels_tensor.mean().item()
    target_std = labels_tensor.std().item()
    print(f"[*] Target stats | Mean: {target_mean:.4f} | Std: {target_std:.4f}")

    # Model, optimizer, loss
    model = build_model()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=patience)

    best_val_mae = float("inf")
    history = {"train_loss": [], "val_mae": [], "val_rmse": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        random.shuffle(train_idx)

        # Mini-batch training
        for start in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[start:start + batch_size]
            batch_data = [dataset[i] for i in batch_idx]
            batch_labels = labels_tensor[batch_idx]

            # Normalize targets
            batch_labels_norm = (batch_labels - target_mean) / (target_std + 1e-8)

            # Build batch tensors (simple concatenation)
            atom_feas, nbr_feas, nbr_idxs = zip(*batch_data)
            crystal_atom_idx = []
            offset = 0
            for af in atom_feas:
                crystal_atom_idx.append(list(range(offset, offset + len(af))))
                offset += len(af)

            # Adjust neighbor indices for batch offset
            adjusted_nbr_idxs = []
            offset = 0
            for i, (af, ni) in enumerate(zip(atom_feas, nbr_idxs)):
                adjusted_nbr_idxs.append(ni + offset)
                offset += len(af)

            atom_fea_batch = torch.cat(atom_feas, dim=0)
            nbr_fea_batch = torch.cat(nbr_feas, dim=0)
            nbr_idx_batch = torch.cat(adjusted_nbr_idxs, dim=0)

            optimizer.zero_grad()
            pred = model(atom_fea_batch, nbr_fea_batch, nbr_idx_batch, crystal_atom_idx)
            loss = criterion(pred.squeeze(), batch_labels_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_idx)

        train_loss /= len(train_idx)
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for i in val_idx:
                af, nf, ni = dataset[i]
                pred = model(af, nf, ni, [list(range(len(af)))])
                pred_denorm = pred.item() * target_std + target_mean
                val_preds.append(pred_denorm)
                val_targets.append(labels[i])

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_mae = np.mean(np.abs(val_preds - val_targets))
        val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val MAE: {val_mae:.4f} eV | "
                f"Val RMSE: {val_rmse:.4f} eV"
            )

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_mae": val_mae,
                "target": target,
                "target_mean": target_mean,
                "target_std": target_std,
            }
            torch.save(checkpoint, CHECKPOINT_DIR / f"best_{target}.pt")

        if early_stopper(val_mae):
            print(f"[!] Early stopping at epoch {epoch}")
            break

    print(f"\n[✓] Training complete | Best Val MAE: {best_val_mae:.4f} eV")
    with open(CHECKPOINT_DIR / f"history_{target}.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CGCNN on materials data")
    parser.add_argument("--data", default="data/sample_materials.json")
    parser.add_argument("--target", default="bandgap_ev",
                        choices=["bandgap_ev", "formation_energy_ev_atom"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # Generate sample data if not present
    data_path = Path(args.data)
    if not data_path.exists():
        print("[*] Sample data not found. Generating synthetic dataset...")
        import sys
        sys.path.insert(0, "data")
        from fetcher import generate_synthetic_samples
        generate_synthetic_samples(200)

    train(
        data_path=str(data_path),
        target=args.target,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
