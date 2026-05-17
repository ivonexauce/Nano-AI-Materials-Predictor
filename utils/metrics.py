"""
metrics.py — Custom Evaluation Metrics
MAE, RMSE, R², and other regression metrics for materials property prediction.
"""

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def within_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.1) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= threshold))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "within_0.1": within_threshold(y_true, y_pred, 0.1),
        "within_0.5": within_threshold(y_true, y_pred, 0.5),
    }


if __name__ == "__main__":
    y_true = np.array([3.0, 2.5, 4.0, 1.2, 5.5])
    y_pred = np.array([2.9, 2.6, 4.1, 1.3, 5.3])
    metrics = compute_all_metrics(y_true, y_pred)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
