"""Physical sanity of molrs-backed conformer generation.

``molpy.conformer.Conformer`` subclasses :class:`molrs.Conformer` (distance
geometry + minimization), operating on a :class:`molpy.Atomistic` graph and
returning a fresh structure. The RDKit adapter (``molpy.adapter.rdkit``) remains
available as a separate external backend, but is not the trunk.
"""

from __future__ import annotations

import numpy as np
import pytest

from molpy.conformer import Conformer
from molpy.parser import parse_molecule

# Equilibrium bond lengths (Angstrom) from standard references
# (CRC Handbook / Allen et al., J. Chem. Soc. Perkin Trans. 2, 1987).
_LIT_BOND_LENGTH = {
    frozenset(("O", "H")): 0.96,
    frozenset(("C", "H")): 1.09,
    frozenset(("C", "C")): 1.54,
    frozenset(("C", "O")): 1.43,
}


def _bond_lengths_by_pair(mol):
    atoms = list(mol.atoms)
    index = {id(a): i for i, a in enumerate(atoms)}
    xyz = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
    pairs = []
    for bond in mol.bonds:
        i = index[id(bond.itom)]
        j = index[id(bond.jtom)]
        elems = frozenset((str(atoms[i]["element"]), str(atoms[j]["element"])))
        pairs.append((elems, float(np.linalg.norm(xyz[i] - xyz[j]))))
    return pairs


def test_generate_returns_3d_coords():
    mol = parse_molecule("CCO")  # ethanol, heavy-atom graph
    out, _ = Conformer(seed=42).generate(mol)

    atoms = list(out.atoms)
    assert len(atoms) >= 3  # hydrogens added by default
    xyz = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
    # Real 3D structure: not collapsed to a point, spans all three axes.
    assert xyz.shape[1] == 3
    assert np.ptp(xyz, axis=0).min() > 1e-6


def test_input_molecule_immutable():
    mol = parse_molecule("CCO")
    n_before = len(list(mol.atoms))
    coords_before = [(a.get("x"), a.get("y"), a.get("z")) for a in mol.atoms]

    out, _ = Conformer(seed=7).generate(mol)

    assert out is not mol
    assert len(list(mol.atoms)) == n_before  # no atoms added to input
    coords_after = [(a.get("x"), a.get("y"), a.get("z")) for a in mol.atoms]
    assert coords_after == coords_before  # input coordinates untouched


@pytest.mark.parametrize(
    "smiles,name",
    [("O", "water"), ("C", "methane"), ("CCO", "ethanol")],
)
def test_conformer_physical_sanity(smiles, name):
    """Generated geometries have bond lengths within 10% of literature."""
    mol = parse_molecule(smiles)
    out, _ = Conformer(seed=42).generate(mol)

    pairs = _bond_lengths_by_pair(out)
    assert pairs, f"{name}: no bonds found"
    for elems, length in pairs:
        lit = _LIT_BOND_LENGTH.get(elems)
        assert lit is not None, f"{name}: no reference for {set(elems)}"
        rel = abs(length - lit) / lit
        assert rel <= 0.10, (
            f"{name}: {set(elems)} bond {length:.3f} A deviates "
            f"{rel:.1%} from literature {lit} A"
        )
