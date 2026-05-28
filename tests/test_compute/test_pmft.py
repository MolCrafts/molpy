"""Potential of mean force and torque (PMFT-XY) — molrs-backed thin shell."""

from __future__ import annotations

import numpy as np

import molrs

from molpy.compute import NeighborList, PMFTXY
from molpy.compute.base import Compute

from .parity_helpers import (
    assert_nested_equal,
    frame_coords_snapshot,
    random_periodic_frame,
)


def _frame_nlist():
    frame = random_periodic_frame()
    return frame, NeighborList(cutoff=3.0)(frame)


def test_pmftxy_is_compute_subclass():
    assert issubclass(PMFTXY, Compute)


def test_pmftxy_smoke():
    frame, nlist = _frame_nlist()
    out = PMFTXY(x_max=5.0, y_max=5.0, n_x=20, n_y=20)(frame, nlist)
    assert isinstance(out, list) and len(out) >= 1


def test_pmftxy_parity_with_molrs_direct():
    frame, nlist = _frame_nlist()
    mine = PMFTXY(x_max=5.0, y_max=5.0, n_x=20, n_y=20)(frame, nlist)
    direct = molrs.compute.pmft.PMFTXY(5.0, 5.0, 20, 20).compute([frame], [nlist])
    assert_nested_equal(mine, direct)


def test_pmftxy_input_frame_immutable():
    frame, nlist = _frame_nlist()
    before = frame_coords_snapshot(frame)
    PMFTXY(5.0, 5.0, 20, 20)(frame, nlist)
    np.testing.assert_array_equal(before, frame_coords_snapshot(frame))
