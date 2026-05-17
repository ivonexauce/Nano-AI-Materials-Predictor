"""
structure_viewer.py — 3D Crystal Structure Renderer
Visualizes crystal structures using matplotlib with atomic spheres and bonds.
"""

import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json


ELEMENT_COLORS = {
    "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30",
    "Ti": "#BFC2C7", "Fe": "#E06633", "Cu": "#C88033", "Zn": "#BD80B3",
    "Ga": "#C28F8F", "Ge": "#668F8F", "As": "#BD80B3", "Se": "#FFA100",
    "Ag": "#C0C0C0", "In": "#A67573", "Sn": "#668080", "Sb": "#9E63B5",
    "Te": "#D47A00", "W": "#42878D", "Pt": "#A0A0A0", "Au": "#D0A040",
    "Al": "#BFA6A6", "Mo": "#54B5B5", "Nb": "#73C2C9", "Pd": "#006985",
    "Mn": "#9C7AC7", "Co": "#BD80B3", "Ni": "#50C878", "Cr": "#8A99C7",
    "V": "#A0A0FF", "Mg": "#8AFF8A", "Ca": "#8AFF8A", "Na": "#AB5CF2",
    "K": "#8F40D4", "Cl": "#1FF01F", "Br": "#A62929", "I": "#940094",
}


ATOMIC_RADII = {
    "H": 0.25, "C": 0.7, "N": 0.65, "O": 0.6, "F": 0.5,
    "Si": 1.1, "P": 1.0, "S": 1.0, "Ti": 1.4, "Fe": 1.25,
    "Cu": 1.35, "Zn": 1.35, "Ga": 1.3, "Ge": 1.25, "As": 1.15,
    "Se": 1.15, "Ag": 1.6, "In": 1.55, "Sn": 1.45, "Sb": 1.4,
    "Te": 1.4, "W": 1.35, "Pt": 1.35, "Au": 1.35, "Al": 1.25,
    "Mo": 1.4, "Nb": 1.45, "Pd": 1.4, "Mn": 1.4, "Co": 1.35,
    "Ni": 1.35, "Cr": 1.4, "V": 1.35, "Mg": 1.5, "Ca": 1.8,
    "Na": 1.8, "K": 2.2, "Cl": 1.0, "Br": 1.15, "I": 1.4,
    "O": 0.6, "N": 0.65,
}


def visualize_structure(material: dict, show_bonds: bool = True, save_path: Optional[str] = None):
    sites = material["sites"]
    lattice = material.get("lattice", {})

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for site in sites:
        species = site["species"]
        x, y, z = site["xyz"]
        color = ELEMENT_COLORS.get(species, "#CCCCCC")
        radius = ATOMIC_RADII.get(species, 0.5) * 0.4
        ax.scatter(x, y, z, c=color, s=radius * 500, alpha=0.8, edgecolors="black", linewidth=0.5)

    if show_bonds:
        drawn = set()
        for i, si in enumerate(sites):
            for j, sj in enumerate(sites):
                if i >= j:
                    continue
                dx = [si["xyz"][k] - sj["xyz"][k] for k in range(3)]
                dist = math.sqrt(sum(d ** 2 for d in dx))
                if dist < 3.0:
                    key = (min(i, j), max(i, j))
                    if key not in drawn:
                        drawn.add(key)
                        ax.plot(
                            [si["xyz"][0], sj["xyz"][0]],
                            [si["xyz"][1], sj["xyz"][1]],
                            [si["xyz"][2], sj["xyz"][2]],
                            color="gray", alpha=0.3, linewidth=0.5,
                        )

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title(f"{material.get('formula', 'Unknown')} — {material.get('material_id', '')}")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Saved: {save_path}")
    plt.show()


def visualize_from_json(json_path: str, index: int = 0, show_bonds: bool = True, save: Optional[str] = None):
    with open(json_path) as f:
        data = json.load(f)
    materials = data.get("materials", data) if isinstance(data, dict) else data

    if index < len(materials):
        visualize_structure(materials[index], show_bonds, save)
    else:
        print(f"Index {index} out of range ({len(materials)} materials)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", type=str, default="mp-149", help="Material ID")
    parser.add_argument("--file", type=str, default="data/sample_materials.json")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--save", type=str, help="Save path for image")
    args = parser.parse_args()

    visualize_from_json(args.file, args.index, save=args.save)
