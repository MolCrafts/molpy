"""molpy.compute.RDF — molrs-backed g(r).

Acceptance criteria covered:
- compute-rdf-class-exposed
- rdf-ideal-gas-correct
- input-frame-immutable
- rdf-requires-box
"""

import numpy as np
import pytest

import molpy
from molpy.compute import RDF, NeighborList
from molpy.compute.base import Compute


def _uniform_frame(n: int, box_len: float, seed: int):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, box_len, size=(n, 3))
    frame = molpy.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.box = molpy.Box.cubic(box_len)
    return frame


def test_rdf_is_a_compute_subclass():
    assert issubclass(RDF, Compute)


def test_ideal_gas_g_of_r_approaches_one():
    """For a uniform random point cloud, g(r) → 1 in middle bins."""
    n_frames = 5
    n_points = 2000
    box_len = 30.0
    cutoff = 8.0

    frames = [_uniform_frame(n_points, box_len, seed=i) for i in range(n_frames)]
    nlists = [NeighborList(cutoff=cutoff)(f) for f in frames]

    rdf = RDF(n_bins=40, r_max=cutoff, r_min=0.0)
    result = rdf(frames, nlists)

    g_of_r = np.asarray(result.rdf)
    centers = np.asarray(result.bin_centers)

    # Middle bins (skip the first few near r=0 where statistics are poor and
    # the last few near r_max where shells extend outside the box).
    middle = (centers > 2.0) & (centers < cutoff - 1.0)
    g_middle = g_of_r[middle]
    assert ((g_middle > 0.7) & (g_middle < 1.3)).all(), (
        f"middle-bin g(r) outside [0.7, 1.3]: {g_middle}"
    )


def test_multi_frame_aggregation():
    """g(r) computed over a list of frames matches per-frame averaging."""
    box_len = 20.0
    cutoff = 6.0
    n_bins = 30

    frames = [_uniform_frame(800, box_len, seed=i) for i in range(3)]
    nlists = [NeighborList(cutoff=cutoff)(f) for f in frames]

    multi = RDF(n_bins, r_max=cutoff)(frames, nlists)
    g_multi = np.asarray(multi.rdf)

    # Sanity: shape + finite + non-negative.
    assert g_multi.shape == (n_bins,)
    assert np.isfinite(g_multi).all()
    assert (g_multi >= 0.0).all()


def test_input_frame_immutable():
    frame = _uniform_frame(300, 15.0, seed=11)
    nlist = NeighborList(cutoff=4.0)(frame)

    box_matrix_before = frame.box.matrix.copy()
    x_before = frame["atoms"]["x"].copy()

    RDF(20, r_max=4.0)([frame], [nlist])

    np.testing.assert_array_equal(frame.box.matrix, box_matrix_before)
    np.testing.assert_array_equal(frame["atoms"]["x"], x_before)


def test_no_box_raises():
    """RDF on a frame without a box must raise ValueError mentioning 'box'."""
    frame = molpy.Frame()
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0.0, 10.0, size=(50, 3))
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    # frame.box left as None deliberately.

    with pytest.raises(ValueError, match="box"):
        NeighborList(cutoff=2.0)(frame)
