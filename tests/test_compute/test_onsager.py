"""Onsager — public signature + collective-displacement correctness.

The diagonal Onsager term L_ii is the collective MSD of species i. A single
type-1 atom drifting at constant velocity (wrapped across the box) must give the
analytic L_11(k) = (k*v)^2 once the minimum-image unwrap is applied; an
uncorrelated fixed species gives L_12 = 0.
"""

from __future__ import annotations

import numpy as np

import molrs

import molpy as mp
from molpy.compute import Onsager
from molpy.compute.base import Compute
from molpy.compute.result import OnsagerResult
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


def test_onsager_is_compute_subclass():
    assert issubclass(Onsager, Compute)


def test_onsager_public_signature():
    ons = Onsager(tags=["1,1", "1,2"], max_dt=5.0, dt=1.0)
    assert ons.tags == ["1,1", "1,2"]
    assert ons.n_cache == 5


def test_onsager_returns_result():
    traj = _drift_trajectory()
    result = Onsager(tags=["1,1", "1,2"], max_dt=5.0, dt=1.0)(traj)
    assert isinstance(result, OnsagerResult)
    assert result.correlations["1,1"].shape == (5,)
    assert result.time.shape == (5,)


def test_onsager_diagonal_recovers_collective_msd():
    traj = _drift_trajectory(velocity=1.0)
    result = Onsager(tags=["1,1", "1,2"], max_dt=5.0, dt=1.0)(traj)
    expected = np.array([float(k) ** 2 for k in range(5)])
    np.testing.assert_allclose(result.correlations["1,1"], expected, atol=1e-9)
    # Type 2 is fixed ⇒ cross-correlation is identically zero.
    np.testing.assert_allclose(result.correlations["1,2"], np.zeros(5), atol=1e-9)
