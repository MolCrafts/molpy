"""Dielectric kernels — static ε, dipole assembly, acf_fft."""

from __future__ import annotations

import numpy as np
import pytest

from molpy.compute import Dielectric
from molpy.compute.dielectric import acf_fft


def test_static_dielectric_constant_finite():
    dm = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]
    )
    eps = Dielectric.static_dielectric_constant(dm, 1000.0, 300.0, 1.0)
    assert np.isfinite(eps)
    assert eps >= 1.0


def test_dipole_moment_assembly():
    charges = np.array([1.0, -1.0])
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    m = Dielectric.compute_dipole_moment(charges, pos)
    assert m.shape == (3,)
    assert m[0] == pytest.approx(1.0)


def test_acf_fft_length():
    x = np.random.default_rng(0).standard_normal(64)
    c = acf_fft(x, 10)
    assert len(c) == 11
