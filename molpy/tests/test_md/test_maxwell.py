"""Tests for MaxwellBoltzmann."""

import numpy as np
import pytest
import molpy as mp


def test_same_seed_is_reproducible():
    pos = np.zeros((8, 3))
    mass = np.ones(8)
    a = mp.md.MaxwellBoltzmann(300.0, units="real", seed=7).velocities(pos, mass)
    b = mp.md.MaxwellBoltzmann(300.0, units="real", seed=7).velocities(pos, mass)
    np.testing.assert_array_equal(a, b)


def test_remove_com_leaves_zero_com():
    pos = np.zeros((6, 3))
    mass = np.full(6, 2.0)
    vel = mp.md.MaxwellBoltzmann(200.0, units="real", seed=1).velocities(pos, mass)
    com = (mass.reshape(-1, 1) * vel).sum(0) / mass.sum()
    np.testing.assert_allclose(com, 0.0, atol=1e-12)


def test_rejects_a_nonpositive_temperature():
    with pytest.raises(ValueError, match="strictly positive"):
        mp.md.MaxwellBoltzmann(0.0, units="real")


def test_the_temperature_is_exact_over_3n_minus_3_degrees_of_freedom():
    # LAMMPS `velocity all create 300 7`: T = sum(m v^2) mvv2e / ((3N - 3) boltz).
    mass = np.array([12.0, 1.0, 16.0, 14.0])
    vel = mp.md.MaxwellBoltzmann(300.0, units="real", seed=7).velocities(
        np.zeros((4, 3)), mass
    )
    mvv2e = 48.88821291 * 48.88821291
    boltz = 0.0019872067
    t = (mass[:, None] * vel**2).sum() * mvv2e / (9 * boltz)
    assert t == pytest.approx(300.0, rel=1e-14)


def test_the_default_style_is_lj():
    assert mp.md.MaxwellBoltzmann(1.0).units == "lj"
