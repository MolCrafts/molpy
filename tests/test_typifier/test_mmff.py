"""Native (RDKit-free) MMFF94 typifier + force field, backed by molrs."""

import molrs
import numpy as np
import pytest

from molpy.optimize import LBFGS, ForceFieldPotential
from molpy.typifier import MMFFTypifier


def _ethanol():
    mol = molrs.parse_smiles("CCO").to_atomistic()
    mol, _ = molrs.Conformer(seed=7).generate(mol)
    return mol


def test_typify_returns_atomistic_and_explicit_compilation_is_finite():
    """Typification and typed-frame potential compilation compose explicitly."""
    typ = MMFFTypifier()
    mol = _ethanol()
    typed = typ.typify(mol)
    assert isinstance(typed, molrs.Atomistic)
    frame = typed.to_frame()
    pots = typ.forcefield().to_potentials(frame)
    energy = pots.calc_energy(molrs.extract_coords(frame))
    assert np.isfinite(energy)


def test_forcefield_is_usable_by_forcefield_potential():
    typ = MMFFTypifier()
    frame = typ.typify(_ethanol()).to_frame()
    pot = ForceFieldPotential(typ.forcefield())
    assert np.isfinite(pot.calc_energy(frame))


def test_lbfgs_energy_non_increasing():
    """LBFGS over the MMFF force field (Frame-native) lowers, never raises, energy."""
    typ = MMFFTypifier()
    frame = typ.typify(_ethanol()).to_frame()
    pot = ForceFieldPotential(typ.forcefield())
    e0 = pot.calc_energy(frame)
    result = LBFGS(pot, maxstep=0.04, memory=20).run(frame, fmax=0.05, steps=300)
    e1 = pot.calc_energy(result.frame)
    assert e1 <= e0 + 1e-6


def test_mmff_typifier_takes_no_variant_argument():
    with pytest.raises(TypeError):
        MMFFTypifier(variant="MMFF95")
