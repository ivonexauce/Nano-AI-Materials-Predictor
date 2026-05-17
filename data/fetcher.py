"""
fetcher.py — Materials Project API Client
Fetches crystal structure data and DFT-computed properties for GNN training.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False
    print("[WARN] mp-api not installed. Run: pip install mp-api")

MP_API_KEY = os.environ.get("MP_API_KEY", "")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


PROPERTIES = [
    "material_id",
    "formula_pretty",
    "structure",
    "bandgap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "bulk_modulus",
    "shear_modulus",
    "density",
    "volume",
    "nsites",
    "elements",
    "nelements",
    "is_stable",
    "theoretical",
]


def fetch_materials(
    n_materials: int = 1000,
    properties: list = None,
    output_file: str = "data/materials_dataset.json",
    bandgap_range: tuple = (0, 10),
) -> list:
    """
    Fetch materials from the Materials Project database.

    Args:
        n_materials: Number of materials to fetch
        properties: List of properties to retrieve
        output_file: Path to save JSON output
        bandgap_range: (min, max) bandgap filter in eV

    Returns:
        List of material dictionaries
    """
    if not MP_API_AVAILABLE:
        print("[ERROR] mp-api not available. Using sample data instead.")
        return load_sample_data()

    if not MP_API_KEY:
        print("[WARN] MP_API_KEY not set. Register at materialsproject.org")
        return load_sample_data()

    props = properties or PROPERTIES
    materials = []

    print(f"[*] Fetching {n_materials} materials from Materials Project...")
    print(f"[*] Bandgap range: {bandgap_range[0]} – {bandgap_range[1]} eV")

    try:
        with MPRester(MP_API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                bandgap=(bandgap_range[0], bandgap_range[1]),
                num_sites=(1, 50),
                fields=props,
            )

            for i, doc in enumerate(docs[:n_materials]):
                material = {
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "bandgap_ev": doc.bandgap,
                    "formation_energy_ev_atom": doc.formation_energy_per_atom,
                    "energy_above_hull": doc.energy_above_hull,
                    "nsites": doc.nsites,
                    "nelements": doc.nelements,
                    "is_stable": doc.is_stable,
                    "density": doc.density,
                    "volume": doc.volume,
                }

                # Serialize structure to dict
                if doc.structure:
                    material["lattice"] = {
                        "a": doc.structure.lattice.a,
                        "b": doc.structure.lattice.b,
                        "c": doc.structure.lattice.c,
                        "alpha": doc.structure.lattice.alpha,
                        "beta": doc.structure.lattice.beta,
                        "gamma": doc.structure.lattice.gamma,
                        "volume": doc.structure.lattice.volume,
                    }
                    material["sites"] = [
                        {
                            "species": str(site.specie),
                            "coords": list(site.frac_coords),
                            "xyz": list(site.coords),
                        }
                        for site in doc.structure.sites
                    ]
                    material["elements"] = [str(el) for el in doc.structure.elements]

                materials.append(material)

                if (i + 1) % 100 == 0:
                    print(f"  [{i+1}/{n_materials}] fetched...")

    except Exception as e:
        print(f"[ERROR] API fetch failed: {e}")
        return load_sample_data()

    with open(output_file, "w") as f:
        json.dump({"fetched_at": datetime.utcnow().isoformat(), "count": len(materials), "materials": materials}, f, indent=2)

    print(f"[✓] Saved {len(materials)} materials to {output_file}")
    return materials


def load_sample_data() -> list:
    """Load bundled sample materials for offline use."""
    sample_path = Path("data/sample_materials.json")
    if sample_path.exists():
        with open(sample_path) as f:
            data = json.load(f)
        materials = data.get("materials", data) if isinstance(data, dict) else data
        print(f"[✓] Loaded {len(materials)} sample materials from local file")
        return materials
    else:
        print("[INFO] Generating synthetic sample data...")
        return generate_synthetic_samples(100)


def generate_synthetic_samples(n: int = 100) -> list:
    """Generate synthetic material data for testing without API access."""
    import random
    import math

    elements = ["Si", "Ti", "Al", "Fe", "Cu", "Zn", "Ga", "Ge", "As", "Se",
                "Nb", "Mo", "Pd", "Ag", "In", "Sn", "Sb", "Te", "W", "Pt"]
    compounds = [
        ("TiO2", "Ti", "O"), ("SiO2", "Si", "O"), ("GaN", "Ga", "N"),
        ("ZnO", "Zn", "O"), ("AlN", "Al", "N"), ("MoS2", "Mo", "S"),
        ("WS2", "W", "S"), ("InP", "In", "P"), ("GaAs", "Ga", "As"),
    ]

    samples = []
    for i in range(n):
        comp = random.choice(compounds)
        bandgap = round(random.uniform(0.1, 6.5), 3)
        formation_e = round(random.uniform(-4.0, 0.5), 3)
        nsites = random.randint(2, 20)

        sample = {
            "material_id": f"mp-{10000 + i}",
            "formula": comp[0],
            "bandgap_ev": bandgap,
            "formation_energy_ev_atom": formation_e,
            "energy_above_hull": round(max(0, formation_e + random.uniform(0, 0.5)), 3),
            "nsites": nsites,
            "nelements": 2,
            "is_stable": formation_e < -0.5,
            "density": round(random.uniform(2.0, 12.0), 2),
            "volume": round(nsites * random.uniform(8, 25), 2),
            "elements": [comp[1], comp[2]],
            "lattice": {
                "a": round(random.uniform(3.0, 8.0), 4),
                "b": round(random.uniform(3.0, 8.0), 4),
                "c": round(random.uniform(3.0, 12.0), 4),
                "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
                "volume": round(nsites * random.uniform(8, 25), 2),
            },
            "sites": [
                {
                    "species": random.choice([comp[1], comp[2]]),
                    "coords": [round(random.uniform(0, 1), 4) for _ in range(3)],
                    "xyz": [round(random.uniform(0, 6), 4) for _ in range(3)],
                }
                for _ in range(nsites)
            ],
        }
        samples.append(sample)

    output = {"generated_at": datetime.utcnow().isoformat(), "count": n, "materials": samples}
    with open("data/sample_materials.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"[✓] Generated {n} synthetic material samples")
    return samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-materials", type=int, default=1000)
    parser.add_argument("--output", default="data/materials_dataset.json")
    args = parser.parse_args()

    materials = fetch_materials(n_materials=args.n_materials, output_file=args.output)
    print(f"\n[Summary] Total materials: {len(materials)}")
    if materials:
        print(f"  Example: {materials[0]['formula']} | Bandgap: {materials[0].get('bandgap_ev')} eV")
