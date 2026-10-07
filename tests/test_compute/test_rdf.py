"""molpy.compute.Rdf — native g(r) over core neighbour tables."""

import numpy as np

import molpy as mp
from molpy.compute import Rdf


def _uniform_frame(n: int, box_len: float, seed: int):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, box_len, size=(n, 3))
    frame = mp.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.box = mp.Box.cube(box_len)
    return frame


def test_ideal_gas_g_of_r_approaches_one(self_neighbors):
    """For a uniform random point cloud, g(r) → 1 in middle bins."""
    n_frames = 5
    n_points = 2000
    box_len = 30.0
    cutoff = 8.0

    frames = [_uniform_frame(n_points, box_len, seed=i) for i in range(n_frames)]
    nlists = [self_neighbors(f, cutoff) for f in frames]

    rdf = Rdf(n_bins=40, r_max=cutoff, r_min=0.0)
    result = rdf.compute(frames, nlists)

    g_of_r = np.asarray(result.rdf)
    centers = np.asarray(result.bin_centers)

    # Middle bins (skip the first few near r=0 where statistics are poor and
    # the last few near r_max where shells extend outside the box).
    middle = (centers > 2.0) & (centers < cutoff - 1.0)
    g_middle = g_of_r[middle]
    assert ((g_middle > 0.7) & (g_middle < 1.3)).all(), (
        f"middle-bin g(r) outside [0.7, 1.3]: {g_middle}"
    )


def test_multi_frame_aggregation(self_neighbors):
    """g(r) computed over a list of frames matches per-frame averaging."""
    box_len = 20.0
    cutoff = 6.0
    n_bins = 30

    frames = [_uniform_frame(800, box_len, seed=i) for i in range(3)]
    nlists = [self_neighbors(f, cutoff) for f in frames]

    multi = Rdf(n_bins, r_max=cutoff).compute(frames, nlists)
    g_multi = np.asarray(multi.rdf)

    # Sanity: shape + finite + non-negative.
    assert g_multi.shape == (n_bins,)
    assert np.isfinite(g_multi).all()
    assert (g_multi >= 0.0).all()


def test_input_frame_immutable(self_neighbors):
    frame = _uniform_frame(300, 15.0, seed=11)
    nlist = self_neighbors(frame, 4.0)

    box_matrix_before = frame.box.h.copy()
    x_before = frame["atoms"]["x"].copy()

    Rdf(20, r_max=4.0).compute([frame], [nlist])

    np.testing.assert_array_equal(frame.box.h, box_matrix_before)
    np.testing.assert_array_equal(frame["atoms"]["x"], x_before)
