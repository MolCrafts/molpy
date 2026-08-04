"""MMFF94 typifier + force field (molrs), including RDKit-locked goldens.

No RDKit import. Type/energy numbers were produced offline with RDKit MMFF94
(Embed + MMFFOptimize, seed=0) and hardcoded below.

# subset reason: ethane + ethanol only — full multi-molecule MMFF energy matrix
# is not yet a directory-scanned fixture set; ethane is NOT charge-zero-only for
# typing (C/H types) and ethanol exercises O/HO. Expand when a fixture dir lands.
"""

from __future__ import annotations

import molrs
import numpy as np
import pytest

import molpy as mp
from molpy.optimize import LBFGS, ForceFieldPotential
from molpy.typifier import MMFFTypifier

# RDKit MMFF94 ethane geometry + types (C,C then H's) — seed=0 Embed+Optimize
_ETHANE_XYZ = np.array(
    [
        [-0.755534, 0.016504, 0.021793],
        [0.755534, -0.016505, -0.021794],
        [-1.142993, -0.844097, 0.575198],
        [-1.106038, 0.928669, 0.513854],
        [-1.169933, -0.009884, -0.990430],
        [1.106036, -0.928669, -0.513858],
        [1.142994, 0.844101, -0.575196],
        [1.169934, 0.009881, 0.990431],
    ],
    dtype=np.float64,
)
_ETHANE_ELEMS = ["C", "C", "H", "H", "H", "H", "H", "H"]
_ETHANE_BONDS = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
_ETHANE_MMFF_TYPES = [1, 1, 5, 5, 5, 5, 5, 5]
_ETHANE_RDKIT_ENERGY = -4.7343652914613745  # kcal/mol on same geometry


def _ethane_rdkit_geom() -> molrs.Atomistic:
    mol = molrs.Atomistic()
    ids = [
        mol.add_atom(el, float(x), float(y), float(z))
        for el, (x, y, z) in zip(_ETHANE_ELEMS, _ETHANE_XYZ)
    ]
    for i, j in _ETHANE_BONDS:
        mol.add_bond(ids[i], ids[j])
    return mol


@pytest.fixture(scope="module")
def ethanol_typed(mmff: MMFFTypifier):
    mol = mp.io.read_smiles("CCO")
    mol, _ = molrs.conformer.Conformer(seed=7).generate(mol)
    return mmff.typify(mol)


def test_typify_returns_atomistic_and_explicit_compilation_is_finite(
    mmff: MMFFTypifier, ethanol_typed
):
    assert isinstance(ethanol_typed, molrs.Atomistic)
    frame = ethanol_typed.to_frame()
    pots = mmff.forcefield().to_potentials(frame)
    energy = pots.calc_energy(molrs.ff.extract_coords(frame))
    assert np.isfinite(energy)


def test_forcefield_is_usable_by_forcefield_potential(
    mmff: MMFFTypifier, ethanol_typed
):
    frame = ethanol_typed.to_frame()
    pot = ForceFieldPotential(mmff.forcefield())
    assert np.isfinite(pot.calc_energy(frame))


def test_lbfgs_energy_non_increasing(mmff: MMFFTypifier, ethanol_typed):
    frame = ethanol_typed.to_frame()
    pot = ForceFieldPotential(mmff.forcefield())
    e0 = pot.calc_energy(frame)
    result = LBFGS(pot, maxstep=0.04, memory=20).run(frame, fmax=0.05, steps=300)
    e1 = pot.calc_energy(result.frame)
    assert e1 <= e0 + 1e-6


def test_mmff_typifier_takes_no_variant_argument():
    with pytest.raises(TypeError):
        MMFFTypifier(variant="MMFF95")


def test_ethane_types_match_rdkit_mmff94(mmff: MMFFTypifier):
    """Hardcoded RDKit MMFF atom types on locked ethane geometry."""
    typed = mmff.typify(_ethane_rdkit_geom())
    got = [int(t) for t in typed.to_frame()["atoms"]["type"]]
    assert got == _ETHANE_MMFF_TYPES


def test_ethane_energy_near_rdkit_on_locked_geometry(mmff: MMFFTypifier):
    """Same coords as RDKit optimise; energy within 0.5 kcal/mol of RDKit."""
    typed = mmff.typify(_ethane_rdkit_geom())
    frame = typed.to_frame()
    e = float(
        mmff.forcefield()
        .to_potentials(frame)
        .calc_energy(molrs.ff.extract_coords(frame))
    )
    assert np.isfinite(e)
    assert abs(e - _ETHANE_RDKIT_ENERGY) < 0.5


def test_ethanol_type_multiset_matches_rdkit_mmff94(mmff: MMFFTypifier, ethanol_typed):
    """RDKit multiset: C×2 type 1, O type 6, H×5 type 5, HO type 21."""
    got = sorted(int(t) for t in ethanol_typed.to_frame()["atoms"]["type"])
    assert got == sorted([1, 1, 6, 5, 5, 5, 5, 5, 21])
