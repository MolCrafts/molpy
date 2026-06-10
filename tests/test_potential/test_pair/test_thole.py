"""Tests for the Thole dipole-dipole screening potential."""

import numpy as np
import pytest

from molpy.potential.pair import Thole


def _pair(r):
    """A single i-j pair at separation r along x; returns (dr, dr_norm)."""
    dr = np.array([[r, 0.0, 0.0]])
    return dr, np.linalg.norm(dr, axis=1)


TI, TJ = np.array([0]), np.array([1])
PAIR_IDX = np.array([[0, 1]])


def _thole():
    return Thole(charge=[1.0, -1.0], alpha=[1.0, 1.122], a_thole=[2.6, 2.6])


def _expected_T(r, ai=2.6, aj=2.6, alpha_i=1.0, alpha_j=1.122):
    s = 0.5 * (ai + aj) / (alpha_i * alpha_j) ** (1 / 6)
    x = s * r
    return 1.0 - (1.0 + x / 2.0) * np.exp(-x)


# --- ac-001/002/003 -----------------------------------------------------------
def test_importable():
    from molpy.potential.pair import Thole as T

    assert T is Thole


def test_calc_energy_finite_scalar():
    dr, drn = _pair(2.5)
    e = _thole().calc_energy(dr, drn, TI, TJ)
    assert isinstance(e, float) and np.isfinite(e)


def test_calc_forces_shape_and_empty():
    dr, drn = _pair(2.5)
    f = _thole().calc_forces(dr, drn, TI, TJ, PAIR_IDX, 2)
    assert f.shape == (2, 3) and np.all(np.isfinite(f))
    empty = np.empty((0,), dtype=int)
    f0 = _thole().calc_forces(
        np.empty((0, 3)), np.empty((0,)), empty, empty, np.empty((0, 2), int), 2
    )
    assert f0.shape == (2, 3) and not f0.any()


# --- ac-004: closed form ------------------------------------------------------
@pytest.mark.parametrize("r", [0.5, 1.0, 2.5, 5.0])
def test_thole_closed_form(r):
    dr, drn = _pair(r)
    t = _thole().damping(drn, TI, TJ)[0]
    assert t == pytest.approx(_expected_T(r), abs=1e-10)


# --- ac-006: analytic force == finite difference ------------------------------
def test_force_matches_finite_difference():
    pot = _thole()
    pos = np.array([[0.0, 0.0, 0.0], [2.3, 0.4, -0.7]])

    def energy(p):
        dr = (p[1] - p[0]).reshape(1, 3)
        return pot.calc_energy(dr, np.linalg.norm(dr, axis=1), TI, TJ)

    analytic = pot.calc_forces(
        (pos[1] - pos[0]).reshape(1, 3),
        np.linalg.norm((pos[1] - pos[0]).reshape(1, 3), axis=1),
        TI,
        TJ,
        PAIR_IDX,
        2,
    )
    h = 1e-6
    fd = np.zeros((2, 3))
    for a in range(2):
        for c in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[a, c] += h
            pm[a, c] -= h
            fd[a, c] = -(energy(pp) - energy(pm)) / (2 * h)
    assert np.allclose(analytic, fd, rtol=1e-5, atol=1e-7)


# --- ac-007/008: limits -------------------------------------------------------
def test_no_damping_at_long_range():
    dr, drn = _pair(50.0)
    assert _thole().damping(drn, TI, TJ)[0] == pytest.approx(1.0, abs=1e-6)


def test_strong_damping_at_short_range():
    rs = np.array([0.2, 0.5, 1.0, 2.0])
    t = np.array([_thole().damping(_pair(r)[1], TI, TJ)[0] for r in rs])
    assert np.all(t < 1.0)
    assert np.all(np.diff(t) > 0)  # monotonically increasing toward 1
    # T_ij -> 0 as r -> 0 (full screening): (1 + x/2) e^{-x} -> 1
    assert _thole().damping(_pair(1e-3)[1], TI, TJ)[0] < 0.01
