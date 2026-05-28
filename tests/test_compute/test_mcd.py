"""MCDCompute — public signature + minimum-image unwrap correctness.

Phase 4 rewires the periodic unwrap to molrs ``Box.delta(minimum_image=True)``.
The public signature is unchanged and the minimum-image unwrap must still
recover continuous displacement across a periodic boundary, so a constant-
velocity atom that wraps the box yields the analytic MSD ``<|Δr|²> = (kΔ)²``.
"""

from __future__ import annotations

import numpy as np
import pytest

import molpy as mp
from molpy.compute import MCDCompute
from molpy.compute.base import Compute
from molpy.compute.result import MCDResult
from molpy.core.trajectory import Trajectory


def _linear_drift_trajectory(n_frames=13, box_len=10.0, velocity=1.0):
    """Single type-1 atom drifting +velocity/frame in x, wrapped into the box."""
    frames = []
    for i in range(n_frames):
        x = (i * velocity) % box_len
        frame = mp.Frame()
        frame["atoms"] = {
            "x": np.array([x]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "type": np.array([1]),
        }
        frame.box = mp.Box.cubic(box_len)
        frames.append(frame)
    return Trajectory(frames)


def test_mcdcompute_is_compute_subclass():
    assert issubclass(MCDCompute, Compute)


def test_mcdcompute_public_signature():
    mcd = MCDCompute(tags=["1"], max_dt=5.0, dt=1.0, center_of_mass=None)
    assert mcd.tags == ["1"]
    assert mcd.max_dt == 5.0
    assert mcd.dt == 1.0
    assert mcd.n_cache == 5


def test_mcd_returns_result_with_tag():
    traj = _linear_drift_trajectory()
    result = MCDCompute(tags=["1"], max_dt=5.0, dt=1.0)(traj)
    assert isinstance(result, MCDResult)
    assert "1" in result.correlations
    assert result.correlations["1"].shape == (5,)


def test_mcd_minimum_image_unwrap_recovers_linear_msd():
    """A wrapping constant-velocity atom must give MSD(k) = (k·v)²."""
    traj = _linear_drift_trajectory(velocity=1.0)
    result = MCDCompute(tags=["1"], max_dt=5.0, dt=1.0)(traj)
    expected = np.array([float(k) ** 2 for k in range(5)])
    # Without correct minimum-image unwrap the boundary crossing at frame 10
    # injects a -9 displacement and the MSD is wildly wrong.
    np.testing.assert_allclose(result.correlations["1"], expected, atol=1e-9)


def test_mcd_missing_type_field_raises():
    frame = mp.Frame()
    frame["atoms"] = {
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
    }
    frame.box = mp.Box.cubic(10.0)
    traj = Trajectory([frame, frame])
    with pytest.raises(ValueError, match="type"):
        MCDCompute(tags=["1"], max_dt=2.0, dt=1.0)(traj)
