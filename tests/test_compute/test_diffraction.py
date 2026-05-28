"""Static structure factor S(k) — molrs-backed thin shell (Debye)."""

from __future__ import annotations

import numpy as np

import molrs

from molpy.compute import StaticStructureFactorDebye
from molpy.compute.base import Compute

from .parity_helpers import (
    assert_nested_equal,
    frame_coords_snapshot,
    random_periodic_frame,
)


def test_ssf_is_compute_subclass():
    assert issubclass(StaticStructureFactorDebye, Compute)


def test_ssf_parity_with_molrs_direct():
    frame = random_periodic_frame()
    k = np.linspace(0.5, 8.0, 40)
    mine = StaticStructureFactorDebye(k)(frame)
    direct = molrs.compute.diffraction.StaticStructureFactorDebye(k).compute([frame])
    assert_nested_equal(mine, direct)


def test_ssf_ideal_gas_large_k_approaches_one():
    """Uniform random points -> S(k) ~ 1 at large k (no structure)."""
    frame = random_periodic_frame(n=2000, box_len=30.0, seed=3)
    k = np.linspace(1.0, 15.0, 60)
    out = StaticStructureFactorDebye(k)(frame)
    # out is per-frame (k_values, S, n_particles); take the single frame's S.
    s_of_k = np.asarray(out[0][1])
    upper_third = s_of_k[len(s_of_k) * 2 // 3 :]
    assert np.all(np.abs(upper_third - 1.0) < 0.3)


def test_ssf_input_frame_immutable():
    frame = random_periodic_frame()
    before = frame_coords_snapshot(frame)
    StaticStructureFactorDebye(np.linspace(0.5, 8.0, 20))(frame)
    np.testing.assert_array_equal(before, frame_coords_snapshot(frame))
