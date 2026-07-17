"""Shared helpers for molrs analysis-operator parity tests.

Not a test module (no ``test_`` prefix), imported explicitly by the
per-operator test files.
"""

from __future__ import annotations

import numpy as np

import molrs

import molpy as mp


def random_periodic_frame(n=200, box_len=12.0, seed=0):
    """A frame of ``n`` uniformly random points in a cubic periodic box."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, box_len, size=(n, 3))
    frame = molrs.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.simbox = mp.Box.cubic(box_len)
    return frame


def assert_nested_equal(a, b, atol=1e-12):
    """Element-wise deep equality for molrs return values.

    Handles the tuple / list / ndarray / dict / scalar shapes the molrs
    analysis kernels return.
    """
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k], atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y, atol)
    elif isinstance(a, np.ndarray) or hasattr(a, "__array__"):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol)
    else:
        assert a == b or abs(a - b) <= atol


def frame_coords_snapshot(frame):
    block = frame["atoms"]
    return np.column_stack([block["x"], block["y"], block["z"]]).copy()


def attach_orientations(frame, heads, tails):
    """Attach an ``orientations`` topology block to ``frame`` (in place).

    One ``(head, tail)`` atom-index row per particle, using the same on-disk
    schema as the core ``bonds`` block — the two endpoint columns ``atomi``
    (head) and ``atomj`` (tail), stored as unsigned-int atom indices. The
    orientation-aware compute ops (Nematic / SpatialDistribution / PMFTXY) read
    their per-particle axis ``normalize(pos[head] - pos[tail])`` from this block.
    """
    frame["orientations"] = {
        "atomi": np.asarray(heads, dtype=np.uint32),
        "atomj": np.asarray(tails, dtype=np.uint32),
    }
    return frame
