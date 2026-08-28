"""Tests for VelocityVerlet / Langevin — constructor-owned LJCut + VerletSkin."""

import numpy as np
import pytest

from molpy import Box, NeighborList, VerletSkin
from molpy.md import LJCut, Langevin, MD, VelocityVerlet


def _dimer(*, skin: float = 0.3, rc: float = 2.5):
    """Two atoms inside cutoff; returns pos, lj, skin_nl, mass."""
    pos = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float64)
    box = Box.cubic(20.0)
    nl = VerletSkin(NeighborList(rc + skin), rc, pos, box, skin=skin)
    lj = LJCut(1.0, 1.0, rc, shifted=True)
    mass = np.ones(2, dtype=np.float64)
    return pos, lj, nl, mass


def _nve(dt: float = 0.01) -> tuple[np.ndarray, VelocityVerlet]:
    pos, lj, nl, mass = _dimer()
    return pos, VelocityVerlet(dt, potential=lj, neighbors=nl, mass=mass)


def _langevin(**kw) -> tuple[np.ndarray, Langevin]:
    dt = kw.pop("dt", 0.05)
    gamma = kw.pop("gamma", 2.0)
    kbt = kw.pop("kbt", 1.5)
    seed = kw.pop("seed", 0)
    mass = kw.pop("mass", None)
    pos, lj, nl, default_mass = _dimer()
    if mass is None:
        mass = default_mass
    else:
        mass = np.atleast_1d(np.asarray(mass, dtype=np.float64))
        if mass.size == 1:
            mass = np.full(2, float(mass[0]))
    return pos, Langevin(
        dt, gamma=gamma, kbt=kbt, potential=lj, neighbors=nl, mass=mass, seed=seed
    )


def test_langevin_constants_match_closed_form():
    dt, gamma, kbt, mass = 0.05, 2.0, 1.5, 2.0
    _, ig = _langevin(dt=dt, gamma=gamma, kbt=kbt, mass=mass)
    assert ig.c1 == pytest.approx(np.exp(-gamma * dt))
    assert ig.c2 == pytest.approx(np.sqrt(1.0 - np.exp(-2.0 * gamma * dt)))
    assert float(ig.sigma[0, 0]) == pytest.approx(np.sqrt(kbt / mass))
    assert float(ig.inv_mass[0, 0]) == pytest.approx(1.0 / mass)


def test_langevin_rejects_gamma_zero():
    pos, lj, nl, mass = _dimer()
    with pytest.raises(ValueError, match="VelocityVerlet"):
        Langevin(0.01, gamma=0.0, kbt=1.0, potential=lj, neighbors=nl, mass=mass)


def test_removed_dof_follows_the_scheme():
    _, nve = _nve()
    _, lgv = _langevin(dt=0.01, gamma=2.0, kbt=1.0)
    assert nve.removed_dof == 3
    assert lgv.removed_dof == 0


def test_mass_must_be_positive():
    pos, lj, nl, _ = _dimer()
    with pytest.raises(ValueError, match="strictly positive"):
        VelocityVerlet(0.01, potential=lj, neighbors=nl, mass=-1.0)
    pos, lj, nl, _ = _dimer()
    with pytest.raises(ValueError, match="strictly positive"):
        VelocityVerlet(0.01, potential=lj, neighbors=nl, mass=np.array([1.0, -2.0]))


def test_non_double_dtype_is_reserved_on_the_driver():
    with pytest.raises(ValueError, match="float64"):
        MD(dtype=np.float32)


def test_advance_n_matches_manual_advance_loop():
    pos0 = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float64)
    vel0 = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]], dtype=np.float64)
    _, a = _langevin(dt=0.05, gamma=3.0, kbt=1.0, mass=1.0, seed=11)
    end_a = a.advance_n(a.initial(pos0.copy(), vel0.copy()), 5)
    _, b = _langevin(dt=0.05, gamma=3.0, kbt=1.0, mass=1.0, seed=11)
    state = b.initial(pos0.copy(), vel0.copy())
    for _ in range(5):
        state = b.advance(state)
    np.testing.assert_array_equal(end_a.pos, state.pos)
    np.testing.assert_array_equal(end_a.vel, state.vel)


def test_nve_force_changes_when_atoms_move_with_skin():
    """Skin>0 must not freeze forces on live geometry (stale sorted_pos bug)."""
    pos, lj, nl, mass = _dimer(skin=1.0)
    ig = VelocityVerlet(0.01, potential=lj, neighbors=nl, mass=mass)
    vel = np.zeros_like(pos)
    s0 = ig.initial(pos, vel)
    pos1 = pos.copy()
    pos1[1, 0] += 0.05
    s1 = ig.initial(pos1, vel)
    assert not np.allclose(s0.forces, s1.forces)
