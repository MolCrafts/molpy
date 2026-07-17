"""PMSDCompute — public signature + minimum-image unwrap correctness.

Phase 4 rewires the periodic unwrap to molrs ``Box.delta(minimum_image=True)``.
The public signature is unchanged and the unwrap must still recover continuous
polarization displacement across a periodic boundary: a cation drifting at
constant velocity (anion fixed) yields the analytic PMSD ``(kΔ)²``.
"""

from __future__ import annotations

import numpy as np
import pytest

import molrs

import molpy as mp
from molpy.compute import PMSDCompute
from molpy.compute.base import Compute
from molpy.compute.result import PMSDResult
from molpy.core.trajectory import Trajectory


def _drift_trajectory(n_frames=13, box_len=10.0, velocity=1.0):
    """Cation (type 1) drifts +velocity/frame in x (wrapped); anion (type 2) fixed."""
    frames = []
    for i in range(n_frames):
        xc = (i * velocity) % box_len
        frame = molrs.Frame()
        frame["atoms"] = {
            "x": np.array([xc, 0.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "type": np.array([1, 2]),
        }
        frame.simbox = mp.Box.cubic(box_len)
        frames.append(frame)
    return Trajectory(frames)


def test_pmsdcompute_is_compute_subclass():
    assert issubclass(PMSDCompute, Compute)


def test_pmsdcompute_public_signature():
    pmsd = PMSDCompute(cation_type=1, anion_type=2, max_dt=5.0, dt=1.0)
    assert pmsd.cation_type == 1
    assert pmsd.anion_type == 2
    assert pmsd.max_dt == 5.0
    assert pmsd.dt == 1.0
    assert pmsd.n_cache == 5


def test_pmsd_returns_result():
    traj = _drift_trajectory()
    result = PMSDCompute(cation_type=1, anion_type=2, max_dt=5.0, dt=1.0)(traj)
    assert isinstance(result, PMSDResult)
    assert result.pmsd.shape == (5,)


def test_pmsd_minimum_image_unwrap_recovers_linear_pmsd():
    """A wrapping constant-velocity cation must give PMSD(k) = (k·v)²."""
    traj = _drift_trajectory(velocity=1.0)
    result = PMSDCompute(cation_type=1, anion_type=2, max_dt=5.0, dt=1.0)(traj)
    expected = np.array([float(k) ** 2 for k in range(5)])
    np.testing.assert_allclose(result.pmsd, expected, atol=1e-9)
