"""Tests for the Tang-Toennies charge/induced-dipole damping potential."""

from math import factorial

import numpy as np
import pytest

from molpy.potential.pair import TangToennies


def _pair(r):
    dr = np.array([[r, 0.0, 0.0]])
    return dr, np.linalg.norm(dr, axis=1)


TI, TJ = np.array([0]), np.array([1])
PAIR_IDX = np.array([[0, 1]])


def _tt():
    return TangToennies(charge=[1.0, -1.0], b=4.5, n=4, c=1.0)


def _expected_f(r, b=4.5, n=4, c=1.0):
    br = b * r
    series = sum((br**k) / factorial(k) for k in range(n + 1))
    return 1.0 - c * np.exp(-br) * series


# --- ac-001/002/003 -----------------------------------------------------------
def test_importable():
    from molpy.potential.pair import TangToennies as T

    assert T is TangToennies


def test_calc_energy_finite_scalar():
    dr, drn = _pair(1.5)
    e = _tt().calc_energy(dr, drn, TI, TJ)
    assert isinstance(e, float) and np.isfinite(e)


def test_calc_forces_shape_and_empty():
    dr, drn = _pair(1.5)
    f = _tt().calc_forces(dr, drn, TI, TJ, PAIR_IDX, 2)
    assert f.shape == (2, 3) and np.all(np.isfinite(f))
    empty = np.empty((0,), dtype=int)
    f0 = _tt().calc_forces(
        np.empty((0, 3)), np.empty((0,)), empty, empty, np.empty((0, 2), int), 2
    )
    assert f0.shape == (2, 3) and not f0.any()


# --- ac-005: closed form ------------------------------------------------------
@pytest.mark.parametrize("r", [0.3, 0.8, 1.5, 3.0])
def test_tt_closed_form(r):
    dr, drn = _pair(r)
    f = _tt().damping(drn)[0]
    assert f == pytest.approx(_expected_f(r), abs=1e-10)


# --- ac-006: analytic force == finite difference ------------------------------
def test_force_matches_finite_difference():
    pot = _tt()
    pos = np.array([[0.0, 0.0, 0.0], [1.2, -0.5, 0.3]])

    def energy(p):
        dr = (p[1] - p[0]).reshape(1, 3)
        return pot.calc_energy(dr, np.linalg.norm(dr, axis=1), TI, TJ)

    d = (pos[1] - pos[0]).reshape(1, 3)
    analytic = pot.calc_forces(d, np.linalg.norm(d, axis=1), TI, TJ, PAIR_IDX, 2)
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
    dr, drn = _pair(20.0)
    assert _tt().damping(drn)[0] == pytest.approx(1.0, abs=1e-6)


def test_strong_damping_at_short_range():
    rs = np.array([0.05, 0.15, 0.3, 0.6])
    f = np.array([_tt().damping(_pair(r)[1])[0] for r in rs])
    assert np.all(f < 1.0)
    assert np.all(np.diff(f) > 0)
    assert f[0] < 0.05
