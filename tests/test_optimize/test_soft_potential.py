"""Force-field-free relaxation with :class:`molpy.optimize.SoftPotential`.

The soft potential reads only coordinates + bonds from a ``molrs.Frame`` (no
force field, no atom types). Driven by :class:`~molpy.optimize.LBFGS`, an
over-stretched bond must relax toward its covalent-radii target ``r0`` and the
potential energy must decrease.
"""

import molrs
import numpy as np

from molpy.core.atomistic import Atomistic
from molpy.optimize import LBFGS, SoftPotential


def _coords(frame: "molrs.Frame") -> np.ndarray:
    return np.asarray(molrs.extract_coords(frame), dtype=float).reshape(-1, 3)


def _cc_frame(length: float) -> "molrs.Frame":
    """Two carbons bonded at separation ``length`` (covalent r0 ≈ 1.52 Å)."""
    s = Atomistic()
    a = s.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    b = s.def_atom(element="C", x=length, y=0.0, z=0.0)
    s.def_bond(a, b, order=1.0)
    return s.to_frame()


def test_stretched_bond_relaxes_toward_r0_and_energy_decreases():
    frame = _cc_frame(3.0)
    potential = SoftPotential()

    energy_before = potential.calc_energy(frame)
    result = LBFGS(potential).run(frame, fmax=0.01, steps=1000)

    coords = _coords(result.frame)
    bond_length = float(np.linalg.norm(coords[0] - coords[1]))

    # C-C covalent target is 0.76 + 0.76 = 1.52 Å; the stretched 3.0 Å bond
    # must move decisively toward it and the energy must drop.
    assert bond_length < 2.0
    assert abs(bond_length - 1.52) < 0.1
    assert result.energy < energy_before


def test_calc_forces_returns_n_by_3():
    frame = _cc_frame(3.0)
    forces = SoftPotential().calc_forces(frame)
    assert forces.shape == (2, 3)
    # Newton's third law: the two bonded atoms feel equal-and-opposite forces.
    assert np.allclose(forces[0], -forces[1])


def test_explicit_scalar_r0_overrides_covalent_default():
    frame = _cc_frame(3.0)
    result = LBFGS(SoftPotential(r0=2.5)).run(frame, fmax=0.01, steps=1000)
    coords = _coords(result.frame)
    bond_length = float(np.linalg.norm(coords[0] - coords[1]))
    assert abs(bond_length - 2.5) < 0.1


def test_bonds_only_path_when_repulsion_disabled():
    """With ``repulsion=False`` no neighbor query is built (bonds-only path)."""
    frame = _cc_frame(3.0)
    potential = SoftPotential(repulsion=False)

    energy_before = potential.calc_energy(frame)
    result = LBFGS(potential).run(frame, fmax=0.01, steps=1000)

    coords = _coords(result.frame)
    bond_length = float(np.linalg.norm(coords[0] - coords[1]))
    assert abs(bond_length - 1.52) < 0.1
    assert result.energy < energy_before


def test_repulsion_pushes_overlapping_nonbonded_atoms_apart():
    """A close non-bonded contact gains a positive repulsive energy."""
    s = Atomistic()
    s.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    s.def_atom(element="C", x=1.0, y=0.0, z=0.0)  # 1.0 Å < rc, not bonded
    frame = s.to_frame()

    with_rep = SoftPotential(repulsion=True, rc=2.0).calc_energy(frame)
    without_rep = SoftPotential(repulsion=False).calc_energy(frame)
    assert with_rep > without_rep


def test_frame_without_bonds_is_bonds_free_and_finite():
    """A frame carrying no bond block evaluates without crashing."""
    s = Atomistic()
    s.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    frame = s.to_frame()
    energy = SoftPotential().calc_energy(frame)
    forces = SoftPotential().calc_forces(frame)
    assert np.isfinite(energy)
    assert forces.shape == (1, 3)
