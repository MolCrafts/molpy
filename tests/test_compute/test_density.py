"""Density-field operators — molrs-backed thin shells.

LocalDensity (needs a neighbor list) and GaussianDensity (frame only) forward
verbatim to ``molrs.compute.density.*``.
"""

from __future__ import annotations

import numpy as np
import pytest

import molrs

from molpy.compute import GaussianDensity, LocalDensity, NeighborList
from molpy.compute.base import Compute

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


def test_density_classes_are_compute_subclasses():
    assert issubclass(LocalDensity, Compute)
    assert issubclass(GaussianDensity, Compute)


def test_local_density_smoke(frame_and_nlist):
    frame, nlist = frame_and_nlist
    out = LocalDensity(r_max=3.0)(frame, nlist)
    assert isinstance(out, list) and len(out) >= 1


def test_local_density_parity_with_molrs_direct(frame_and_nlist):
    frame, nlist = frame_and_nlist
    mine = LocalDensity(r_max=3.0, diameter=0.0)(frame, nlist)
    direct = molrs.compute.density.LocalDensity(3.0, 0.0).compute([frame], [nlist])
    assert_nested_equal(mine, direct)


def test_gaussian_density_parity_with_molrs_direct(frame_and_nlist):
    frame, _ = frame_and_nlist
    mine = GaussianDensity(nx=8, ny=8, nz=8, sigma=1.0)(frame)
    direct = molrs.compute.density.GaussianDensity(8, 8, 8, 1.0).compute([frame])
    assert_nested_equal(mine, direct)


def test_density_input_frame_immutable(frame_and_nlist):
    frame, nlist = frame_and_nlist
    before = frame_coords_snapshot(frame)
    LocalDensity(r_max=3.0)(frame, nlist)
    GaussianDensity(8, 8, 8, 1.0)(frame)
    np.testing.assert_array_equal(before, frame_coords_snapshot(frame))
