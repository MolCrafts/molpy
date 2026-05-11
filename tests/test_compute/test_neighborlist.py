"""molpy.compute.NeighborList — molrs-backed spatial neighbor list.

Acceptance criteria covered:
- compute-neighborlist-class-exposed
- neighborlist-parity-with-molrs
- input-frame-immutable
"""

import numpy as np
import pytest

import molpy
import molrs
from molpy.compute import NeighborList
from molpy.compute.base import Compute


def _make_random_frame(n: int = 200, box_len: float = 10.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, box_len, size=(n, 3))
    frame = molpy.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.box = molpy.Box.cubic(box_len)
    return frame, xyz


def test_neighborlist_is_a_compute_subclass():
    assert issubclass(NeighborList, Compute)


def test_basic_periodic():
    frame, _ = _make_random_frame()
    nlist = NeighborList(cutoff=2.0)(frame)
    assert nlist.n_pairs > 0
    distances = np.asarray(nlist.distances)
    assert (distances <= 2.0).all()
    assert (distances >= 0.0).all()


def test_parity_with_molrs_direct():
    """molpy.compute.NeighborList must produce the same pairs as a direct
    molrs.NeighborQuery call on the same inputs.
    """
    frame, xyz = _make_random_frame(seed=42)

    via_molpy = NeighborList(cutoff=2.5)(frame)
    via_molrs = molrs.NeighborQuery(frame.box, xyz, 2.5).query_self()

    assert via_molpy.n_pairs == via_molrs.n_pairs
    np.testing.assert_array_equal(
        np.sort(via_molpy.distances), np.sort(via_molrs.distances)
    )


def test_input_frame_immutable():
    frame, xyz = _make_random_frame(seed=7)
    box_matrix_before = frame.box.matrix.copy()
    x_before = frame["atoms"]["x"].copy()
    y_before = frame["atoms"]["y"].copy()
    z_before = frame["atoms"]["z"].copy()
    pbc_before = frame.box.pbc.copy()

    NeighborList(cutoff=3.0)(frame)

    np.testing.assert_array_equal(frame.box.matrix, box_matrix_before)
    np.testing.assert_array_equal(frame["atoms"]["x"], x_before)
    np.testing.assert_array_equal(frame["atoms"]["y"], y_before)
    np.testing.assert_array_equal(frame["atoms"]["z"], z_before)
    np.testing.assert_array_equal(frame.box.pbc, pbc_before)


def test_distances_within_cutoff():
    """Spot-check: every reported pair distance ≤ cutoff."""
    frame, _ = _make_random_frame(n=500, seed=1)
    cutoff = 1.8
    nlist = NeighborList(cutoff=cutoff)(frame)
    distances = np.asarray(nlist.distances)
    assert (distances <= cutoff + 1e-10).all()
