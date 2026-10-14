"""``Oacf`` — orientational autocorrelation of a unit-vector series.

The FFT route (``order`` 1 and 2) is pinned to the definition written out in
numpy, and to ``LegendreReorientation`` on the same vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

import molrs
from molrs.compute import Acf, LegendreReorientation, Oacf


def _unit_vectors(n_frames=40, n_vectors=6, seed=5):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n_frames, n_vectors, 3))
    # Smooth so neighbouring frames are correlated.
    for t in range(1, n_frames):
        u[t] = 0.9 * u[t - 1] + 0.1 * u[t]
    return u / np.linalg.norm(u, axis=2, keepdims=True)


def _legendre(order, x):
    return {1: x, 2: 0.5 * (3 * x * x - 1), 3: 0.5 * (5 * x**3 - 3 * x)}[order]


def _oacf_direct(u, order, max_lag, stride=1):
    n_frames = u.shape[0]
    out = []
    for lag in range(max_lag + 1):
        origins = range(0, n_frames - lag, stride)
        acc = sum(float(_legendre(order, (u[tau] * u[tau + lag]).sum(axis=1)).sum()) for tau in origins)
        out.append(acc / (u.shape[1] * len(origins)))
    return np.array(out)


class TestOacf:
    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("stride", [1, 3])
    def test_matches_the_direct_definition(self, order, stride):
        u = _unit_vectors()
        got = Oacf(order=order, stride=stride).compute(u, max_lag=12, dt=0.5)
        np.testing.assert_allclose(got.values, _oacf_direct(u, order, 12, stride), atol=1e-12)
        assert got.values[0] == pytest.approx(1.0)
        assert got.kind == "oacf" and got.order == order and got.stride == stride
        assert got.lag_times[12] == 6.0
        assert got.counts.tolist() == [(40 - 1 - lag) // stride + 1 for lag in range(13)]

    def test_p2_is_the_quadratic_component_identity(self):
        u = _unit_vectors()
        q = np.einsum("tni,tnj->tnij", u, u).reshape(u.shape[0], u.shape[1], 9)
        c_q = Acf().compute(q, max_lag=10).values
        c2 = Oacf(order=2).compute(u, max_lag=10).values
        np.testing.assert_allclose(1.5 * c_q - 0.5, c2, atol=1e-12)

    def test_c2_is_even_under_reversal(self):
        u = _unit_vectors()
        a = Oacf(order=2).compute(u, max_lag=8).values
        b = Oacf(order=2).compute(-u, max_lag=8).values
        np.testing.assert_array_equal(a, b)

    def test_quarter_turns(self):
        angles = np.arange(16) * np.pi / 2
        u = np.zeros((16, 2, 3))
        u[:, :, 0] = np.cos(angles)[:, None]
        u[:, :, 1] = np.sin(angles)[:, None]
        np.testing.assert_allclose(Oacf(order=1).compute(u, max_lag=4).values, [1, 0, -1, 0, 1], atol=1e-12)
        np.testing.assert_allclose(Oacf(order=2).compute(u, max_lag=4).values, [1, -0.5, 1, -0.5, 1], atol=1e-12)

    def test_per_entity_rows_and_integrated_time(self):
        u = _unit_vectors()
        got = Oacf(order=2).compute(u, max_lag=10, dt=0.5, per_entity=True)
        assert got.per_entity.shape == (11, 6)
        np.testing.assert_allclose(got.per_entity.mean(axis=1), got.values, atol=1e-12)
        assert got.integrate() == pytest.approx(np.trapezoid(got.values, dx=0.5))

    def test_parity_with_legendre_reorientation(self):
        u = _unit_vectors(n_frames=24, n_vectors=4)
        rng = np.random.default_rng(9)
        centres = rng.uniform(0.0, 50.0, size=(4, 3))
        frames = []
        for t in range(24):
            pos = np.concatenate([centres, centres + u[t]])
            f, b = molrs.core.Frame(), molrs.core.Block()
            b["x"], b["y"], b["z"] = pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy()
            f["atoms"] = b
            bonds = molrs.core.Block()
            bonds["atomi"] = np.arange(4, dtype=np.uint32)
            bonds["atomj"] = np.arange(4, 8, dtype=np.uint32)
            f["bonds"] = bonds
            frames.append(f)
        legendre = LegendreReorientation(max_lag=10, stride=3).compute(frames)
        c1 = Oacf(order=1, stride=3).compute(u, max_lag=10).values
        c2 = Oacf(order=2, stride=3).compute(u, max_lag=10).values
        np.testing.assert_allclose(legendre.c1, c1, atol=1e-12)
        np.testing.assert_allclose(legendre.c2, c2, atol=1e-12)

    def test_rejects_bad_input(self):
        u = _unit_vectors(n_frames=8, n_vectors=2)
        with pytest.raises(ValueError, match="unit"):
            Oacf().compute(1.01 * u, max_lag=3)
        with pytest.raises(ValueError):
            Oacf().compute(u[:, :, :2], max_lag=3)
        with pytest.raises(ValueError):
            Oacf(order=0)
        with pytest.raises(ValueError):
            Oacf(stride=0)
        with pytest.raises(ValueError):
            Oacf().compute(u[:1], max_lag=3)
