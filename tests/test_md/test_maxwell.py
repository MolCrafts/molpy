"""Tests for MaxwellBoltzmann."""

import numpy as np
import pytest

from molpy.md import MaxwellBoltzmann


def test_same_seed_is_reproducible():
    pos = np.zeros((8, 3))
    mass = np.ones(8)
    a = MaxwellBoltzmann(300.0, seed=7).velocities(pos, mass)
    b = MaxwellBoltzmann(300.0, seed=7).velocities(pos, mass)
    np.testing.assert_array_equal(a, b)


def test_remove_com_leaves_zero_com():
    pos = np.zeros((6, 3))
    mass = np.full(6, 2.0)
    vel = MaxwellBoltzmann(200.0, seed=1).velocities(pos, mass)
    com = (mass.reshape(-1, 1) * vel).sum(0) / mass.sum()
    np.testing.assert_allclose(com, 0.0, atol=1e-12)


def test_rejects_nonpositive_kbt():
    with pytest.raises(ValueError, match="strictly positive"):
        MaxwellBoltzmann(0.0)
