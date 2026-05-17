"""
property_plotter.py — Property Distribution Plots
Generates histograms, scatter plots, and heatmaps for materials property data.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_style():
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150})


def plot_property_distribution(
    materials: list,
    property_name: str = "bandgap_ev",
    bins: int = 30,
    save_path: Optional[str] = None,
):
    set_style()
    values = [m[property_name] for m in materials if property_name in m and m[property_name] is not None]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=bins, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(np.mean(values), color="red", linestyle="--", label=f"Mean: {np.mean(values):.2f}")
    ax.axvline(np.median(values), color="green", linestyle="--", label=f"Median: {np.median(values):.2f}")

    ax.set_xlabel(property_name.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {property_name.replace('_', ' ').title()}")
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[✓] Saved: {save_path}")
    plt.show()


def plot_scatter(
    materials: list,
    x_prop: str = "bandgap_ev",
    y_prop: str = "formation_energy_ev_atom",
    color_by: Optional[str] = "is_stable",
    save_path: Optional[str] = None,
):
    set_style()
    x_vals = [m[x_prop] for m in materials if x_prop in m and y_prop in m]
    y_vals = [m[y_prop] for m in materials if x_prop in m and y_prop in m]

    fig, ax = plt.subplots(figsize=(10, 7))

    if color_by and color_by in materials[0]:
        colors = ["green" if m.get(color_by) else "red" for m in materials if x_prop in m and y_prop in m]
        scatter = ax.scatter(x_vals, y_vals, c=colors, alpha=0.6, edgecolors="gray", linewidth=0.3)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="green", alpha=0.6, label="Stable"),
            Patch(facecolor="red", alpha=0.6, label="Unstable"),
        ]
        ax.legend(handles=legend_elements)
    else:
        ax.scatter(x_vals, y_vals, alpha=0.6, c="steelblue", edgecolors="gray", linewidth=0.3)

    ax.set_xlabel(x_prop.replace("_", " ").title())
    ax.set_ylabel(y_prop.replace("_", " ").title())
    ax.set_title(f"{y_prop.replace('_', ' ').title()} vs {x_prop.replace('_', ' ').title()}")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[✓] Saved: {save_path}")
    plt.show()


def plot_element_frequency(
    materials: list,
    top_n: int = 15,
    save_path: Optional[str] = None,
):
    set_style()
    from collections import Counter
    element_counter = Counter()
    for m in materials:
        for el in m.get("elements", []):
            element_counter[el] += 1

    elements, counts = zip(*element_counter.most_common(top_n))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(elements)), counts, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(elements)))
    ax.set_xticklabels(elements, rotation=45, ha="right")
    ax.set_xlabel("Element")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Top {top_n} Elements in Dataset")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", va="bottom", fontsize=9)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[✓] Saved: {save_path}")
    plt.show()


def generate_all_plots(materials: list, output_dir: str = "reports/figures"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_property_distribution(materials, "bandgap_ev", save_path=f"{output_dir}/bandgap_dist.png")
    plot_property_distribution(materials, "formation_energy_ev_atom", save_path=f"{output_dir}/formation_energy_dist.png")
    plot_scatter(materials, save_path=f"{output_dir}/bandgap_vs_formation.png")
    plot_element_frequency(materials, save_path=f"{output_dir}/element_frequency.png")
    print(f"[✓] All plots saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_materials.json")
    parser.add_argument("--property", default="bandgap_ev", help="Property to plot")
    parser.add_argument("--output", type=str, help="Save figure to path")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    materials = data.get("materials", data) if isinstance(data, dict) else data

    plot_property_distribution(materials, args.property, save_path=args.output)
