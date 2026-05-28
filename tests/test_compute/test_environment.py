"""Bond-order environment — molrs-backed thin shell."""

from __future__ import annotations

import numpy as np

import molrs

from molpy.compute import BondOrder, NeighborList
from molpy.compute.base import Compute

from .parity_helpers import (
    assert_nested_equal,
    frame_coords_snapshot,
    random_periodic_frame,
)


def _frame_nlist():
    frame = random_periodic_frame()
    return frame, NeighborList(cutoff=3.0)(frame)


def test_bondorder_is_compute_subclass():
    assert issubclass(BondOrder, Compute)


def test_bondorder_smoke():
    frame, nlist = _frame_nlist()
    out = BondOrder(n_theta=6, n_phi=6)(frame, nlist)
    assert isinstance(out, list) and len(out) >= 1


def test_bondorder_parity_with_molrs_direct():
    frame, nlist = _frame_nlist()
    mine = BondOrder(n_theta=6, n_phi=6)(frame, nlist)
    direct = molrs.compute.environment.BondOrder(6, 6).compute([frame], [nlist])
    assert_nested_equal(mine, direct)


def test_bondorder_input_frame_immutable():
    frame, nlist = _frame_nlist()
    before = frame_coords_snapshot(frame)
    BondOrder(6, 6)(frame, nlist)
    np.testing.assert_array_equal(before, frame_coords_snapshot(frame))
