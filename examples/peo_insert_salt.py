"""Insert Li+ / TFSI- salt into the equilibrated PEO matrix — for an illustration.

This is the "just give me a figure" path. Rather than parameterizing the salt
for MD, it takes the GAFF-equilibrated PEO matrix
(``step10_lammps_equilibrated.data`` from ``peo_lammps_relax.py``) and **randomly
inserts** the ions into the voids with simple overlap rejection: each Li+ as a
point and each TFSI- as a rigid molecule (3D conformer) at a random position and
orientation. The result is a complete polymer-electrolyte box — the
GAFF/LAMMPS-relaxed PEO melt plus salt — written as a LAMMPS data file and a PDB
for rendering.

(The equilibrated matrix is clean because the GAFF force field has real torsions;
the ion positions are random and clash-free, but not themselves energy-relaxed,
so this is an illustration, not an equilibrated electrolyte.)

Run (after peo_lammps_relax.py)::

    python examples/peo_insert_salt.py
"""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np

from molpy.conformer import Conformer
from molpy.core.box import Box
from molpy.core.frame import Block, Frame
from molpy.io import read_lammps_data, write_lammps_data, write_pdb
from molpy.parser import parse_molecule

_WF_PATH = Path(__file__).resolve().parent / "peo_electrolyte_workflow.py"
_spec = importlib.util.spec_from_file_location("peo_workflow", _WF_PATH)
wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wf)

OUT_DIR = Path(__file__).resolve().parent / "peo_workflow_output"
PEO_FILE = OUT_DIR / "step10_lammps_equilibrated.data"
# Recover element symbols from atomic mass (the GAFF data file stores GAFF atom
# types like c3/os/hc in the 'type' column, not element symbols).
_MASS_TO_ELEMENT = {1: "H", 7: "Li", 12: "C", 14: "N", 16: "O", 19: "F", 32: "S"}
N_LI = 8
N_TFSI = 8
SEED = 42
LI_MIN_DIST = 2.3  # A, Li+ clearance to any atom
TFSI_MIN_DIST = 2.2  # A, TFSI atom clearance to any atom


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random 3x3 rotation matrix (QR of a Gaussian matrix)."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _tfsi_template() -> tuple[list[str], np.ndarray, np.ndarray]:
    """TFSI- anion: element symbols, centered coords, and bonds (0-based)."""
    anion = parse_molecule("[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F")
    anion, _ = Conformer(seed=SEED).generate(anion)
    frame = anion.to_frame()
    atoms = frame["atoms"]
    elements = list(np.asarray(atoms["element"]).astype(str))
    xyz = np.column_stack([np.asarray(atoms[k]).astype(float) for k in ("x", "y", "z")])
    xyz = xyz - xyz.mean(axis=0)  # center at origin for rigid placement
    bonds = frame["bonds"]
    bond_idx = np.column_stack(
        [np.asarray(bonds[k]).astype(int) for k in ("atomi", "atomj")]
    )
    return elements, xyz, bond_idx


def _min_distance(candidate: np.ndarray, existing: np.ndarray) -> float:
    """Smallest distance between any candidate point and any existing point."""
    diff = candidate[:, None, :] - existing[None, :, :]
    return float(np.sqrt((diff**2).sum(axis=-1)).min())


def main() -> None:
    if not PEO_FILE.exists():
        print(f"Missing {PEO_FILE.name}. Run examples/peo_lammps_relax.py first.")
        return

    peo = read_lammps_data(str(PEO_FILE), atom_style="full")
    pa = peo["atoms"]
    # GAFF data stores GAFF atom types (c3/os/hc) in 'type'; recover the element
    # symbol from atomic mass for clean element-colored rendering.
    elements = [_MASS_TO_ELEMENT.get(int(round(float(m))), "C") for m in pa["mass"]]
    xyz = np.column_stack([np.asarray(pa[k]).astype(float) for k in ("x", "y", "z")])
    charge = list(np.asarray(pa["charge"]).astype(float))
    mol_id = list(np.asarray(pa["mol_id"]).astype(int))
    box = peo.box
    lengths, origin = np.asarray(box.lengths), np.asarray(box.origin)

    # Connectivity from PEO (0-based indices), with element-pattern type labels.
    conn = {}
    for name, keys in (
        ("bonds", ["atomi", "atomj"]),
        ("angles", ["atomi", "atomj", "atomk"]),
        ("dihedrals", ["atomi", "atomj", "atomk", "atoml"]),
    ):
        if name in peo:
            block = peo[name]
            conn[name] = np.column_stack(
                [np.asarray(block[k]).astype(int) for k in keys]
            )

    rng = np.random.default_rng(SEED)
    placed = xyz.copy()  # all occupied positions (grows as we insert)
    next_mol = max(mol_id) + 1
    tfsi_elems, tfsi_xyz, tfsi_bonds = _tfsi_template()

    def random_point() -> np.ndarray:
        return origin + rng.random(3) * lengths

    # --- insert Li+ (points) ---
    n_li = 0
    for _ in range(N_LI * 500):
        if n_li >= N_LI:
            break
        pt = random_point()[None, :]
        if _min_distance(pt, placed) >= LI_MIN_DIST:
            elements.append("Li")
            xyz = np.vstack([xyz, pt])
            placed = np.vstack([placed, pt])
            charge.append(1.0)
            mol_id.append(next_mol)
            next_mol += 1
            n_li += 1

    # --- insert TFSI- (rigid molecules) ---
    n_tfsi = 0
    for _ in range(N_TFSI * 2000):
        if n_tfsi >= N_TFSI:
            break
        center = random_point()
        coords = (tfsi_xyz @ _random_rotation(rng).T) + center
        if _min_distance(coords, placed) < TFSI_MIN_DIST:
            continue
        base = len(elements)
        elements.extend(tfsi_elems)
        xyz = np.vstack([xyz, coords])
        placed = np.vstack([placed, coords])
        # formal -1 on the central N, 0 elsewhere (net TFSI = -1)
        charge.extend([-1.0 if e == "N" else 0.0 for e in tfsi_elems])
        mol_id.extend([next_mol] * len(tfsi_elems))
        next_mol += 1
        conn.setdefault("bonds", np.empty((0, 2), int))
        conn["bonds"] = np.vstack([conn["bonds"], tfsi_bonds + base])
        n_tfsi += 1

    print(f"inserted Li+ : {n_li}/{N_LI}")
    print(f"inserted TFSI-: {n_tfsi}/{N_TFSI}")

    # --- assemble combined frame ---
    elements_arr = np.array(elements)
    n = len(elements)
    frame = Frame(
        {
            "atoms": Block(
                {
                    "id": np.arange(1, n + 1, dtype=int),
                    "mol_id": np.array(mol_id, dtype=int),
                    "type": elements_arr,
                    "charge": np.array(charge, dtype=float),
                    "element": elements_arr,
                    "x": xyz[:, 0].copy(),
                    "y": xyz[:, 1].copy(),
                    "z": xyz[:, 2].copy(),
                }
            )
        }
    )
    block_keys = {
        2: ["atomi", "atomj"],
        3: ["atomi", "atomj", "atomk"],
        4: ["atomi", "atomj", "atomk", "atoml"],
    }
    for name, idx in conn.items():
        keys = block_keys[idx.shape[1]]
        labels = np.array(["-".join(elements_arr[idx[r]]) for r in range(idx.shape[0])])
        data = {k: idx[:, j] for j, k in enumerate(keys)}
        data["type"] = labels
        frame[name] = Block(data)
    frame.box = Box(lengths, origin=origin)

    counts = Counter(elements)
    net_q = sum(charge)
    print(
        f"electrolyte box: {n} atoms  "
        + ", ".join(f"{el}:{counts[el]}" for el in sorted(counts))
        + f"  net charge {net_q:+.1f} e  box {lengths[0]:.0f}^3 A"
    )

    data_path = OUT_DIR / "step11_electrolyte_box.data"
    pdb_path = OUT_DIR / "step11_electrolyte_box.pdb"
    write_lammps_data(data_path, frame, atom_style="full")
    write_pdb(pdb_path, frame)
    print(f"  -> {data_path.name}")
    print(f"  -> {pdb_path.name}")
    print("\n[OK] PEO + Li+/TFSI- electrolyte box ready for rendering.")


if __name__ == "__main__":
    main()
