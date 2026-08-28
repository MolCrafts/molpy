"""Tests for the ForceField + Frame MD driver (molpy.md.MD)."""

import numpy as np
import pytest

import molrs
from molpy.md import MD


def _bond_frame() -> tuple[object, object]:
    ff = molrs.ff.ForceField("bond-only")
    ff.def_bondtype("harmonic", "CT", "CT", {"k": 300.0, "r0": 1.5})

    frame = molrs.Frame()
    atoms = molrs.Block()
    atoms.insert("x", np.array([0.0, 2.0]))
    atoms.insert("y", np.array([0.0, 0.0]))
    atoms.insert("z", np.array([0.0, 0.0]))
    atoms.insert("mass", np.array([12.0, 12.0]))
    frame["atoms"] = atoms
    bonds = molrs.Block()
    bonds.insert("atomi", np.array([0], dtype=np.uint64))
    bonds.insert("atomj", np.array([1], dtype=np.uint64))
    bonds.insert("type", np.array(["CT-CT"], dtype=str))
    frame["bonds"] = bonds
    return ff, frame


def test_set_potential_runs_bonded_dimer():
    ff, frame = _bond_frame()
    pots = ff.to_potentials(frame)
    state = MD().set_potential(pots).run(frame, 20, dt=0.1)
    assert state.pos.shape == (2, 3)
    assert np.all(np.isfinite(state.pos))


def test_set_forcefield_compiles_per_run():
    ff, frame = _bond_frame()
    driver = MD().set_forcefield(ff)
    first = driver.run(frame, 20, dt=0.1)
    assert np.isfinite(first.energy)
    second = driver.run(frame, 20, dt=0.1)
    assert np.isfinite(second.energy)


def test_set_forcefield_returns_self():
    ff, _frame = _bond_frame()
    driver = MD()
    assert driver.set_forcefield(ff) is driver


def test_md_requires_forcefield():
    with pytest.raises(RuntimeError, match="set_forcefield"):
        MD().run(molrs.Frame(), 1, dt=0.01)


def test_thermo_requires_kb():
    ff, frame = _bond_frame()
    with pytest.raises(ValueError, match="kb="):
        MD().set_forcefield(ff).run(frame, 1, dt=0.01, thermo=1)
