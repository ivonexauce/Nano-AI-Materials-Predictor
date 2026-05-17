"""
periodic_table.py — Atomic Feature Vectors
Provides one-hot encoding and property features for chemical elements.
"""

ATOMIC_FEATURE_DIM = 92

ELEMENT_TO_IDX = {
    "H": 0, "He": 1, "Li": 2, "Be": 3, "B": 4, "C": 5, "N": 6, "O": 7, "F": 8, "Ne": 9,
    "Na": 10, "Mg": 11, "Al": 12, "Si": 13, "P": 14, "S": 15, "Cl": 16, "Ar": 17,
    "K": 18, "Ca": 19, "Sc": 20, "Ti": 21, "V": 22, "Cr": 23, "Mn": 24, "Fe": 25,
    "Co": 26, "Ni": 27, "Cu": 28, "Zn": 29, "Ga": 30, "Ge": 31, "As": 32, "Se": 33,
    "Br": 34, "Kr": 35, "Rb": 36, "Sr": 37, "Y": 38, "Zr": 39, "Nb": 40, "Mo": 41,
    "Tc": 42, "Ru": 43, "Rh": 44, "Pd": 45, "Ag": 46, "Cd": 47, "In": 48, "Sn": 49,
    "Sb": 50, "Te": 51, "I": 52, "Xe": 53, "Cs": 54, "Ba": 55,
    "La": 56, "Ce": 57, "Pr": 58, "Nd": 59, "Pm": 60, "Sm": 61, "Eu": 62, "Gd": 63,
    "Tb": 64, "Dy": 65, "Ho": 66, "Er": 67, "Tm": 68, "Yb": 69, "Lu": 70,
    "Hf": 71, "Ta": 72, "W": 73, "Re": 74, "Os": 75, "Ir": 76, "Pt": 77, "Au": 78,
    "Hg": 79, "Tl": 80, "Pb": 81, "Bi": 82, "Po": 83, "At": 84, "Rn": 85,
    "Fr": 86, "Ra": 87, "Ac": 88, "Th": 89, "Pa": 90, "U": 91,
}


def encode_atom(species: str) -> list:
    idx = ELEMENT_TO_IDX.get(species, 0)
    vec = [0.0] * ATOMIC_FEATURE_DIM
    vec[idx % ATOMIC_FEATURE_DIM] = 1.0
    return vec


if __name__ == "__main__":
    print(f"Periodic Table: {len(ELEMENT_TO_IDX)} elements indexed")
    print(f"Feature dimension: {ATOMIC_FEATURE_DIM}")
    for el in ["H", "C", "O", "Fe", "Si", "Ti", "Ga", "As"]:
        vec = encode_atom(el)
        print(f"  {el:4s} -> index {ELEMENT_TO_IDX[el]:2d}, one-hot sum: {sum(vec):.0f}")
