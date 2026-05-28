"""Bond-orientational order operators — molrs-backed thin shells.

Steinhardt / Hexatic / Nematic / SolidLiquid forward verbatim to
``molrs.compute.order.*``; parity vs the direct molrs call is the core
guarantee, plus smoke and input-immutability.
"""

from __future__ import annotations

import numpy as np
import pytest

import molrs

from molpy.compute import Hexatic, Nematic, SolidLiquid, Steinhardt
from molpy.compute.base import Compute
from molpy.compute import NeighborList

from .parity_helpers import (
    assert_nested_equal,
    frame_coords_snapshot,
    random_periodic_frame,
)


@pytest.fixture
def frame_and_nlist():
    frame = random_periodic_frame()
    nlist = NeighborList(cutoff=3.0)(frame)
    return frame, nlist


def test_order_classes_are_compute_subclasses():
    assert all(
        issubclass(c, Compute) for c in (Steinhardt, Hexatic, Nematic, SolidLiquid)
    )


def test_steinhardt_smoke(frame_and_nlist):
    frame, nlist = frame_and_nlist
    out = Steinhardt([6])(frame, nlist)
    assert isinstance(out, list) and len(out) >= 1


def test_steinhardt_parity_with_molrs_direct(frame_and_nlist):
    frame, nlist = frame_and_nlist
    mine = Steinhardt([4, 6], average=True)(frame, nlist)
    direct = molrs.compute.order.Steinhardt([4, 6], average=True).compute(
        [frame], [nlist]
    )
    assert_nested_equal(mine, direct)


def test_hexatic_parity_with_molrs_direct(frame_and_nlist):
    frame, nlist = frame_and_nlist
    mine = Hexatic(6)(frame, nlist)
    direct = molrs.compute.order.Hexatic(6).compute([frame], [nlist])
    assert_nested_equal(mine, direct)


def test_solidliquid_parity_with_molrs_direct(frame_and_nlist):
    frame, nlist = frame_and_nlist
    mine = SolidLiquid(6, q_threshold=0.7, n_threshold=6)(frame, nlist)
    direct = molrs.compute.order.SolidLiquid(6, 0.7, 6).compute([frame], [nlist])
    assert_nested_equal(mine, direct)


def test_nematic_parity_with_molrs_direct(frame_and_nlist):
    frame, _ = frame_and_nlist
    rng = np.random.default_rng(1)
    directors = rng.standard_normal((200, 3))
    mine = Nematic()(frame, directors)
    direct = molrs.compute.order.Nematic().compute([frame], directors)
    assert_nested_equal(mine, direct)


def test_order_input_frame_immutable(frame_and_nlist):
    frame, nlist = frame_and_nlist
    before = frame_coords_snapshot(frame)
    Steinhardt([6])(frame, nlist)
    Hexatic(6)(frame, nlist)
    SolidLiquid(6)(frame, nlist)
    after = frame_coords_snapshot(frame)
    np.testing.assert_array_equal(before, after)
