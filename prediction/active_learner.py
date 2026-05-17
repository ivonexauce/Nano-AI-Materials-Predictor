"""
active_learner.py — Uncertainty-Based Active Learning for Materials Discovery
Identifies which unmeasured materials the model is most uncertain about,
prioritizing them for expensive DFT simulations or lab experiments.
"""

import json
import math
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cgcnn import build_model


class UncertaintyEstimator:
    """
    Monte Carlo Dropout for uncertainty quantification in GNN predictions.
    Runs multiple stochastic forward passes to estimate prediction variance.
    """

    def __init__(self, model, n_passes: int = 20):
        self.model = model
        self.n_passes = n_passes

    def enable_mc_dropout(self):
        """Keep dropout active during inference (Monte Carlo Dropout)."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(
        self,
        atom_fea: "torch.Tensor",
        nbr_fea: "torch.Tensor",
        nbr_idx: "torch.Tensor",
        crystal_atom_idx: list,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ) -> dict:
        """
        Returns mean prediction and uncertainty estimate.
        Higher uncertainty = model less confident = better candidate for labeling.
        """
        if not TORCH_AVAILABLE:
            return {"mean": 0.0, "std": 0.0, "uncertainty": 0.0}

        self.enable_mc_dropout()
        predictions = []

        with torch.no_grad():
            for _ in range(self.n_passes):
                pred = self.model(atom_fea, nbr_fea, nbr_idx, crystal_atom_idx)
                pred_denorm = pred.item() * target_std + target_mean
                predictions.append(pred_denorm)

        mean_pred = float(np.mean(predictions))
        std_pred = float(np.std(predictions))

        return {
            "mean": round(mean_pred, 4),
            "std": round(std_pred, 4),
            "uncertainty": round(std_pred, 4),  # Higher = more uncertain
            "predictions": predictions,
        }


class ActiveLearner:
    """
    Active learning loop for materials discovery.
    Manages labeled/unlabeled pool and selects next materials to measure.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        strategy: str = "uncertainty",  # uncertainty | random | hybrid
    ):
        self.strategy = strategy
        self.labeled_pool = []
        self.unlabeled_pool = []
        self.query_history = []
        self.model = None
        self.estimator = None

        if checkpoint_path and Path(checkpoint_path).exists() and TORCH_AVAILABLE:
            self._load_model(checkpoint_path)

    def _load_model(self, path: str):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model = build_model()
        self.model.load_state_dict(checkpoint["model_state"])
        self.target_mean = checkpoint.get("target_mean", 0.0)
        self.target_std = checkpoint.get("target_std", 1.0)
        self.estimator = UncertaintyEstimator(self.model)
        print(f"[✓] Model loaded from {path}")

    def load_unlabeled_pool(self, materials: list):
        """Load candidate materials that have no DFT labels yet."""
        self.unlabeled_pool = materials
        print(f"[*] Unlabeled pool: {len(self.unlabeled_pool)} materials")

    def load_labeled_pool(self, materials: list):
        """Load materials that already have measured/computed properties."""
        self.labeled_pool = materials
        print(f"[*] Labeled pool: {len(self.labeled_pool)} materials")

    def score_uncertainty(self, material: dict) -> float:
        """Compute model uncertainty for a single material."""
        if not self.estimator or not TORCH_AVAILABLE:
            return random.uniform(0, 1)  # Random fallback

        sites = material.get("sites", [])
        if not sites:
            return 0.0

        import sys
        sys.path.insert(0, "prediction")
        from trainer import encode_atom, encode_bond_distance

        n_atoms = len(sites)
        atom_fea = torch.tensor(
            [encode_atom(s["species"]) for s in sites], dtype=torch.float32
        )

        n_neighbors = min(4, n_atoms - 1) if n_atoms > 1 else 1
        nbr_fea_list, nbr_idx_list = [], []
        for i, si in enumerate(sites):
            neighbors = []
            for j, sj in enumerate(sites):
                if i == j:
                    continue
                dx = [si["xyz"][k] - sj["xyz"][k] for k in range(3)]
                dist = math.sqrt(sum(d**2 for d in dx))
                neighbors.append((dist, j))
            neighbors.sort()
            neighbors = neighbors[:n_neighbors]
            while len(neighbors) < n_neighbors:
                neighbors.append((3.0, 0))
            nbr_fea_list.append([encode_bond_distance(d) for d, _ in neighbors])
            nbr_idx_list.append([j for _, j in neighbors])

        nbr_fea = torch.tensor(nbr_fea_list, dtype=torch.float32)
        nbr_idx = torch.tensor(nbr_idx_list, dtype=torch.long)

        result = self.estimator.predict_with_uncertainty(
            atom_fea, nbr_fea, nbr_idx,
            [list(range(n_atoms))],
            self.target_mean, self.target_std,
        )
        return result["uncertainty"]

    def select_query_batch(self, n: int = 10) -> list:
        """
        Select next n materials for labeling based on strategy.

        Returns:
            List of selected material dicts with uncertainty scores.
        """
        if not self.unlabeled_pool:
            print("[WARN] Unlabeled pool is empty.")
            return []

        print(f"[*] Scoring {len(self.unlabeled_pool)} candidates ({self.strategy} strategy)...")

        if self.strategy == "random":
            selected = random.sample(self.unlabeled_pool, min(n, len(self.unlabeled_pool)))
            for m in selected:
                m["query_score"] = random.uniform(0, 1)
            return selected

        # Score all unlabeled materials
        scored = []
        for i, material in enumerate(self.unlabeled_pool):
            score = self.score_uncertainty(material)
            scored.append((score, material))
            if (i + 1) % 50 == 0:
                print(f"  Scored {i+1}/{len(self.unlabeled_pool)}...")

        # Sort by uncertainty (descending = most uncertain first)
        scored.sort(key=lambda x: -x[0])
        selected = []
        for score, mat in scored[:n]:
            mat["query_score"] = round(score, 4)
            mat["selected_by"] = self.strategy
            mat["selected_at"] = datetime.utcnow().isoformat()
            selected.append(mat)

        self.query_history.append({
            "round": len(self.query_history) + 1,
            "n_selected": len(selected),
            "strategy": self.strategy,
            "timestamp": datetime.utcnow().isoformat(),
            "top_scores": [s["query_score"] for s in selected[:5]],
        })

        print(f"[✓] Selected {len(selected)} materials | Top uncertainty: {selected[0]['query_score']:.4f}")
        return selected

    def simulate_labeling(self, selected: list, target: str = "bandgap_ev") -> list:
        """
        Simulate DFT labeling with synthetic values.
        In production: replace with actual DFT run or API call.
        """
        labeled = []
        for mat in selected:
            mat[target] = round(random.uniform(0.1, 6.5), 3)  # Synthetic DFT result
            mat["label_source"] = "synthetic_DFT"
            self.labeled_pool.append(mat)
            labeled.append(mat)
        print(f"[✓] Labeled {len(labeled)} materials (synthetic DFT simulation)")
        return labeled

    def run_discovery_loop(
        self,
        n_rounds: int = 5,
        query_size: int = 10,
        target: str = "bandgap_ev",
    ):
        """Run full active learning discovery loop."""
        print("\n" + "=" * 60)
        print("  Active Learning Discovery Loop")
        print(f"  Rounds: {n_rounds} | Query size: {query_size} | Strategy: {self.strategy}")
        print("=" * 60)

        for round_num in range(1, n_rounds + 1):
            print(f"\n── Round {round_num}/{n_rounds} ──")
            print(f"   Labeled: {len(self.labeled_pool)} | Unlabeled: {len(self.unlabeled_pool)}")

            # Select candidates
            selected = self.select_query_batch(n=query_size)
            if not selected:
                print("  No more candidates. Discovery complete.")
                break

            # Label them (simulate)
            self.simulate_labeling(selected, target)

            # Remove from unlabeled pool
            selected_ids = {m.get("material_id") for m in selected}
            self.unlabeled_pool = [
                m for m in self.unlabeled_pool
                if m.get("material_id") not in selected_ids
            ]

        print(f"\n[✓] Discovery loop complete | Total labeled: {len(self.labeled_pool)}")
        return self.query_history

    def save_results(self, output_path: str = "reports/active_learning_results.json"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results = {
            "strategy": self.strategy,
            "labeled_count": len(self.labeled_pool),
            "unlabeled_remaining": len(self.unlabeled_pool),
            "query_history": self.query_history,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[✓] Results saved: {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "data")
    from fetcher import generate_synthetic_samples

    print("=" * 60)
    print("  Active Learning Demo — UMBA Nano-AI Materials Predictor")
    print("=" * 60)

    # Generate sample pool
    all_materials = generate_synthetic_samples(300)

    # Split: 50 labeled (known), 250 unlabeled (candidates for DFT)
    labeled = all_materials[:50]
    unlabeled = all_materials[50:]

    learner = ActiveLearner(strategy="uncertainty")
    learner.load_labeled_pool(labeled)
    learner.load_unlabeled_pool(unlabeled)

    history = learner.run_discovery_loop(n_rounds=3, query_size=10)
    learner.save_results()

    print("\n── Query History ──")
    for h in history:
        print(f"  Round {h['round']}: {h['n_selected']} selected | "
              f"Top scores: {h['top_scores'][:3]}")
