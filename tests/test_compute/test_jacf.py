"""JACF — public signature + Green-Kubo current-autocorrelation conductivity.

A sustained charge current (cation drifting, anion fixed) gives a flat current
autocorrelation and a positive DC conductivity that scales inversely with volume
and temperature.
"""

from __future__ import annotations

import numpy as np

import molrs

import molpy as mp
from molpy.compute import JACF
from molpy.compute.base import Compute
from molpy.compute.result import JACFResult
from molpy.core.trajectory import Trajectory


def _current_trajectory(n_frames=11, box_len=10.0, vcat=1.0):
    """Cation (type 1) moves at +vcat in x; anion (type 2) fixed ⇒ J=(vcat,0,0)."""
    frames = []
    for _ in range(n_frames):
        frame = molrs.Frame()
        frame["atoms"] = {
            "x": np.array([0.0, 5.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "vx": np.array([vcat, 0.0]),
            "vy": np.array([0.0, 0.0]),
            "vz": np.array([0.0, 0.0]),
            "type": np.array([1, 2]),
        }
        frame.box = mp.Box.cubic(box_len)
        frames.append(frame)
    return Trajectory(frames)


def test_jacf_is_compute_subclass():
    assert issubclass(JACF, Compute)


def test_jacf_public_signature():
    jacf = JACF(cation_type=1, anion_type=2, max_dt=5.0, dt=1.0, temperature=300.0)
    assert jacf.cation_type == 1
    assert jacf.anion_type == 2
    assert jacf.n_cache == 5


def test_jacf_returns_result_and_flat_acf():
    traj = _current_trajectory(vcat=1.0)
    result = JACF(cation_type=1, anion_type=2, max_dt=5.0, dt=1.0, temperature=300.0)(
        traj
    )
    assert isinstance(result, JACFResult)
    assert result.jacf.shape == (5,)
    # Constant J=(1,0,0) ⇒ <J(0).J(t)> = 1 at every lag.
    np.testing.assert_allclose(result.jacf, np.ones(5), atol=1e-9)
    assert result.sigma > 0.0


def test_jacf_conductivity_scales_inversely_with_volume_and_temperature():
    traj = _current_trajectory(vcat=1.0)
    a = JACF(
        cation_type=1, anion_type=2, max_dt=5.0, dt=1.0, temperature=300.0, volume=100.0
    )(traj)
    b = JACF(
        cation_type=1, anion_type=2, max_dt=5.0, dt=1.0, temperature=300.0, volume=200.0
    )(traj)
    c = JACF(
        cation_type=1, anion_type=2, max_dt=5.0, dt=1.0, temperature=600.0, volume=100.0
    )(traj)
    assert abs(a.sigma / b.sigma - 2.0) < 1e-9
    assert abs(a.sigma / c.sigma - 2.0) < 1e-9
