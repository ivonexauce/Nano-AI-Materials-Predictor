"""
crystal_utils.py — Symmetry and Lattice Utilities
Helper functions for crystal structure manipulation, symmetry analysis,
and lattice parameter conversions used across the project.
"""

import math
import numpy as np


def calculate_distance(coord1: list, coord2: list) -> float:
    dx = [coord1[k] - coord2[k] for k in range(3)]
    return math.sqrt(sum(d ** 2 for d in dx))


def calculate_angle(p1: list, p2: list, p3: list) -> float:
    v1 = [p1[k] - p2[k] for k in range(3)]
    v2 = [p3[k] - p2[k] for k in range(3)]
    dot = sum(v1[k] * v2[k] for k in range(3))
    n1 = math.sqrt(sum(v ** 2 for v in v1))
    n2 = math.sqrt(sum(v ** 2 for v in v2))
    if n1 * n2 == 0:
        return 0.0
    return math.degrees(math.acos(max(-1, min(1, dot / (n1 * n2)))))


def frac_to_cartesian(frac_coords: list, lattice: dict) -> list:
    a, b, c = lattice["a"], lattice["b"], lattice["c"]
    alpha = math.radians(lattice.get("alpha", 90))
    beta = math.radians(lattice.get("beta", 90))
    gamma = math.radians(lattice.get("gamma", 90))

    volume = a * b * c * math.sqrt(
        1 - math.cos(alpha) ** 2 - math.cos(beta) ** 2 - math.cos(gamma) ** 2
        + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
    )

    matrix = np.array([
        [a, b * math.cos(gamma), c * math.cos(beta)],
        [0, b * math.sin(gamma), c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)],
        [0, 0, volume / (a * b * math.sin(gamma))],
    ])

    cart = matrix @ np.array(frac_coords)
    return cart.tolist()


def cartesian_to_frac(cart_coords: list, lattice: dict) -> list:
    a, b, c = lattice["a"], lattice["b"], lattice["c"]
    alpha = math.radians(lattice.get("alpha", 90))
    beta = math.radians(lattice.get("beta", 90))
    gamma = math.radians(lattice.get("gamma", 90))

    volume = a * b * c * math.sqrt(
        1 - math.cos(alpha) ** 2 - math.cos(beta) ** 2 - math.cos(gamma) ** 2
        + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
    )

    inv_a = 1.0 / a
    inv_b = 1.0 / b
    inv_c = 1.0 / c
    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_gamma = math.cos(gamma)
    sin_gamma = math.sin(gamma)

    frac = np.array([
        [inv_a, -cos_gamma / (a * sin_gamma),
         (cos_alpha * cos_gamma - cos_beta) / (a * sin_gamma * math.sin(beta))],
        [0, 1.0 / (b * sin_gamma),
         (cos_beta * cos_gamma - cos_alpha) / (b * sin_gamma * math.sin(beta))],
        [0, 0, 1.0 / (c * math.sin(beta))],
    ])

    return (frac @ np.array(cart_coords)).tolist()


def build_neighbor_list(sites: list, cutoff: float = 5.0) -> list:
    neighbors = []
    for i, si in enumerate(sites):
        for j, sj in enumerate(sites):
            if i == j:
                continue
            dist = calculate_distance(si["xyz"], sj["xyz"])
            if dist <= cutoff:
                neighbors.append((i, j, dist))
    return neighbors


if __name__ == "__main__":
    lattice = {"a": 5.0, "b": 5.0, "c": 5.0, "alpha": 90, "beta": 90, "gamma": 90}
    cart = frac_to_cartesian([0.5, 0.5, 0.5], lattice)
    frac = cartesian_to_frac(cart, lattice)
    print(f"Frac (0.5,0.5,0.5) -> Cart {cart}")
    print(f"Cart {cart} -> Frac {frac}")
