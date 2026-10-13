"""``Oacf`` — the orientational autocorrelation of user-built unit vectors.

The vectors are the caller's: molpy passes ``(n_frames, n_vectors, 3)`` unit
vectors straight to the molrs kernel and reads a ``TimeCorrelation`` back.
"""

from __future__ import annotations

import numpy as np
import pytest

from molpy.compute import Compute, Oacf, TimeCorrelation


def _rotating(n_frames: int, n_vectors: int, step_deg: float) -> np.ndarray:
    """Vectors in the xy-plane turning by ``step_deg`` per frame."""
    angles = np.deg2rad(step_deg) * np.arange(n_frames)[:, None] + np.linspace(
        0.0, np.pi, n_vectors
    )
    u = np.zeros((n_frames, n_vectors, 3))
    u[:, :, 0] = np.cos(angles)
    u[:, :, 1] = np.sin(angles)
    return u


def test_oacf_is_a_compute_and_returns_a_time_correlation():
    assert isinstance(Oacf(order=2), Compute)
    tc = Oacf().compute(_rotating(10, 3, 5.0), max_lag=4)
    assert isinstance(tc, TimeCorrelation)
    assert tc.kind == "oacf" and tc.order == 1


@pytest.mark.parametrize("order", [1, 2])
def test_static_vectors_correlate_to_one(order):
    u = _rotating(12, 4, 0.0)
    tc = Oacf(order=order).compute(u, max_lag=6)
    np.testing.assert_allclose(tc.values, 1.0, atol=1e-12)


def test_p2_is_even_under_reversal():
    u = _rotating(30, 5, 7.0)
    a = Oacf(order=2).compute(u, max_lag=10).values
    b = Oacf(order=2).compute(-u, max_lag=10).values
    np.testing.assert_array_equal(a, b)


def test_quarter_turns_give_the_textbook_values():
    u = _rotating(16, 2, 90.0)
    c1 = Oacf(order=1).compute(u, max_lag=2).values
    c2 = Oacf(order=2).compute(u, max_lag=2).values
    np.testing.assert_allclose(c1, [1.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(c2, [1.0, -0.5, 1.0], atol=1e-12)


def test_integrate_of_a_constant_curve_is_the_window_length():
    u = _rotating(12, 4, 0.0)
    tc = Oacf(order=2).compute(u, max_lag=6, dt=0.5)
    assert tc.integrate() == pytest.approx(6 * 0.5)
    np.testing.assert_allclose(tc.lag_times, np.arange(7) * 0.5)


def test_non_unit_vectors_are_rejected():
    with pytest.raises(ValueError, match="unit"):
        Oacf().compute(2.0 * _rotating(8, 2, 5.0), max_lag=3)
