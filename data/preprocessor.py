"""
preprocessor.py — CIF to Graph Conversion
Parses crystal structure files (CIF, POSCAR) into graph representations
for GNN training: atoms as nodes, bonds as edges with distance features.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from pymatgen.io.cif import CifParser
    from pymatgen.core.structure import Structure
    PMG_AVAILABLE = True
except ImportError:
    PMG_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.periodic_table import ELEMENT_TO_IDX, ATOMIC_FEATURE_DIM, encode_atom
from prediction.trainer import encode_bond_distance

CUTOFF_RADIUS = 5.0
MAX_NEIGHBORS = 12


def parse_cif(cif_path: str) -> dict:
    if not PMG_AVAILABLE:
        raise ImportError("pymatgen required for CIF parsing. pip install pymatgen")

    parser = CifParser(cif_path)
    structure = parser.get_structures()[0]

    material = {
        "material_id": Path(cif_path).stem,
        "formula": structure.composition.reduced_formula,
        "nsites": len(structure.sites),
        "nelements": len(structure.elements),
        "elements": [str(el) for el in structure.elements],
        "lattice": {
            "a": structure.lattice.a,
            "b": structure.lattice.b,
            "c": structure.lattice.c,
            "alpha": structure.lattice.alpha,
            "beta": structure.lattice.beta,
            "gamma": structure.lattice.gamma,
            "volume": structure.lattice.volume,
        },
        "sites": [
            {
                "species": str(site.specie),
                "coords": list(site.frac_coords),
                "xyz": list(site.coords),
            }
            for site in structure.sites
        ],
    }
    return material


def structure_to_graph(material: dict) -> dict:
    sites = material["sites"]
    n_sites = len(sites)

    atom_features = [encode_atom(s["species"]) for s in sites]

    edge_list = []
    edge_features = []
    for i in range(n_sites):
        for j in range(n_sites):
            if i == j:
                continue
            dx = [sites[i]["xyz"][k] - sites[j]["xyz"][k] for k in range(3)]
            dist = math.sqrt(sum(d ** 2 for d in dx))
            if dist <= CUTOFF_RADIUS:
                edge_list.append((i, j))
                edge_features.append(encode_bond_distance(dist))

    edge_index = np.array(edge_list, dtype=np.int64).T if edge_list else np.zeros((2, 1), dtype=np.int64)
    edge_attr = np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((1, 41), dtype=np.float32)

    if TORCH_AVAILABLE:
        return {
            "atom_features": torch.tensor(atom_features, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        }
    return {"atom_features": atom_features, "edge_index": edge_index, "edge_attr": edge_attr}


def process_cif_directory(cif_dir: str, output_file: str = "data/processed_materials.json"):
    cif_dir = Path(cif_dir)
    materials = []
    for cif_path in cif_dir.glob("*.cif"):
        try:
            mat = parse_cif(str(cif_path))
            materials.append(mat)
        except Exception as e:
            print(f"[WARN] Failed to parse {cif_path.name}: {e}")

    output = {"count": len(materials), "materials": materials}
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[✓] Processed {len(materials)} structures -> {output_file}")
    return materials


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert CIF files to graph data")
    parser.add_argument("--cif-dir", type=str, help="Directory containing .cif files")
    parser.add_argument("--cif-file", type=str, help="Single CIF file")
    parser.add_argument("--output", type=str, default="data/processed_materials.json")
    args = parser.parse_args()

    if args.cif_file:
        mat = parse_cif(args.cif_file)
        graph = structure_to_graph(mat)
        print(f"[✓] Parsed {args.cif_file}: {mat['formula']} ({mat['nsites']} atoms)")
        print(f"    Edges: {graph['edge_index'].shape[1]}")
    elif args.cif_dir:
        process_cif_directory(args.cif_dir, args.output)
    else:
        print("Provide --cif-dir or --cif-file")
