"""Persist — public signature + pair-survival correctness.

A permanently bonded cation-anion pair gives C(tau) = 1 at every lag; a pair
that leaves the cutoff dies permanently under ``continuous`` but re-forms under
``intermittent``.
"""

from __future__ import annotations

import numpy as np

import molpy as mp
from molpy.compute import Persist
from molpy.compute.base import Compute
from molpy.compute.persist import _parse_tag
from molpy.compute.result import PersistResult
from molpy.core.trajectory import Trajectory


def _pair_trajectory(j_positions, box_len=100.0):
    """Type-1 atom fixed at origin; type-2 atom at the given per-frame x."""
    frames = []
    for xj in j_positions:
        frame = mp.Frame()
        frame["atoms"] = {
            "x": np.array([0.0, xj]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "type": np.array([1, 2]),
        }
        frame.box = mp.Box.cubic(box_len)
        frames.append(frame)
    return Trajectory(frames)


def test_persist_is_compute_subclass():
    assert issubclass(Persist, Compute)


def test_parse_tag_grammar():
    assert _parse_tag("3,4:ssp:3.0,4.0") == (3, 4, "ssp", 3.0, 4.0)
    assert _parse_tag("1,1:continuous:3.5") == (1, 1, "continuous", 3.5, 3.5)


def test_persist_returns_result():
    traj = _pair_trajectory([0.5, 0.5, 0.5, 0.5])
    result = Persist(tags=["1,2:continuous:1.0"], max_dt=4.0, dt=1.0)(traj)
    assert isinstance(result, PersistResult)
    assert result.correlations["1,2:continuous:1.0"].shape == (4,)


def test_persist_permanent_bond_survives_fully():
    traj = _pair_trajectory([0.5, 0.5, 0.5, 0.5])
    result = Persist(tags=["1,2:continuous:1.0"], max_dt=4.0, dt=1.0)(traj)
    np.testing.assert_allclose(
        result.correlations["1,2:continuous:1.0"], np.ones(4), atol=1e-12
    )


def test_persist_continuous_vs_intermittent():
    # Bonded at frames 0,1; out at 2; back in at 3.
    traj = _pair_trajectory([0.5, 0.5, 50.0, 0.5])
    cont = Persist(tags=["1,2:continuous:1.0"], max_dt=4.0, dt=1.0)(traj)
    imm = Persist(tags=["1,2:intermittent:1.0"], max_dt=4.0, dt=1.0)(traj)
    # At tau=3 only origin 0 is valid: continuous dead, intermittent alive.
    assert cont.correlations["1,2:continuous:1.0"][3] == 0.0
    assert abs(imm.correlations["1,2:intermittent:1.0"][3] - 1.0) < 1e-12
