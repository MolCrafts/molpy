"""Orientation-axis compute ops (Nematic / SpatialDistribution / PMFTXY).

Regression guard: each op reads its per-particle orientation axis from the
frame's core ``orientations`` topology block — one ``(head, tail)`` atom pair
per row, the same on-disk schema as ``bonds`` (endpoint columns ``atomi`` /
``atomj``). The molpy wrappers therefore forward ``(frames)`` / ``(frames,
nlists)`` ONLY; no separate director / angle / orientation-pair array is passed.
A prior signature passed those external arrays — these tests pin the
no-external-array contract for all three ops (the director/axis is the internal
expansion ``normalize(pos[head] - pos[tail])``).
"""

from __future__ import annotations

import numpy as np
import pytest

import molpy as mp
from molpy.compute import PMFTXY, NeighborList, Nematic, SpatialDistribution
from molpy.compute.base import Compute

from .parity_helpers import attach_orientations

_TEMPLATE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _axis_frame(n_particles: int = 8, box_len: float = 10.0, seed: int = 0):
    """``2 * n_particles`` atoms; particle ``k``'s axis is atoms ``(2k+1, 2k)``.

    The ``orientations`` block holds one ``(head, tail)`` row per particle with
    ``head = 2k+1`` displaced ``+z`` from ``tail = 2k``, so every axis points
    along ``+z`` (a near-perfectly aligned nematic ensemble).
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_particles
    xyz = rng.uniform(0.0, box_len, size=(n, 3))
    for k in range(n_particles):
        xyz[2 * k + 1] = xyz[2 * k] + np.array([0.0, 0.0, 1.0])
    frame = mp.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.box = mp.Box.cubic(box_len)
    heads = [2 * k + 1 for k in range(n_particles)]
    tails = [2 * k for k in range(n_particles)]
    attach_orientations(frame, heads=heads, tails=tails)
    return frame


def _random_frame(n: int, box_len: float, seed: int):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, box_len, size=(n, 3))
    frame = mp.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.box = mp.Box.cubic(box_len)
    return frame


def test_orientation_ops_are_compute_subclasses():
    for cls in (Nematic, SpatialDistribution, PMFTXY):
        assert issubclass(cls, Compute)


def test_frame_carries_orientations_block():
    frame = _axis_frame()
    assert "orientations" in frame.keys()


# --------------------------------------------------------------------------- #
# Nematic                                                                      #
# --------------------------------------------------------------------------- #


def test_nematic_reads_orientations_from_frame():
    # No director array — directors are the unit head-tail vectors of the
    # `orientations` block. All axes point +z, so order ~ 1, director ~ z.
    frame = _axis_frame()
    order, eigenvalues, director, q_tensor = Nematic()(frame)
    assert np.asarray(eigenvalues).shape == (3,)
    assert np.asarray(q_tensor).shape == (3, 3)
    assert order > 0.9
    assert abs(np.asarray(director)[2]) > 0.9


def test_nematic_rejects_external_directors():
    frame = _axis_frame()
    directors = np.zeros((8, 3))
    with pytest.raises(TypeError):
        Nematic()(frame, directors)


# --------------------------------------------------------------------------- #
# PMFTXY                                                                       #
# --------------------------------------------------------------------------- #


def _pmft_frame(n: int = 20, box_len: float = 12.0, seed: int = 1):
    frame = _random_frame(n, box_len, seed)
    idx = np.arange(n, dtype=np.uint32)
    # One (head, tail) row per particle (query-point index order).
    attach_orientations(frame, heads=(idx + 1) % n, tails=idx)
    return frame


def test_pmftxy_reads_orientations_from_frame():
    frame = _pmft_frame()
    nlist = NeighborList(cutoff=3.0)(frame)
    out = PMFTXY(x_max=5.0, y_max=5.0, n_x=20, n_y=20)(frame, nlist)
    assert isinstance(out, list) and len(out) == 1
    counts, _density, _pmf = out[0]
    assert np.asarray(counts).shape == (20, 20)


def test_pmftxy_lab_frame_without_block():
    # No orientations block => lab frame (the old `orientations=None` path).
    frame = _random_frame(20, 12.0, seed=2)
    nlist = NeighborList(cutoff=3.0)(frame)
    out = PMFTXY(x_max=5.0, y_max=5.0, n_x=8, n_y=8)(frame, nlist)
    assert isinstance(out, list) and len(out) == 1


def test_pmftxy_rejects_external_orientations():
    frame = _pmft_frame()
    nlist = NeighborList(cutoff=3.0)(frame)
    with pytest.raises(TypeError):
        PMFTXY(5.0, 5.0, 20, 20)(frame, nlist, [[0.0] * 20])


# --------------------------------------------------------------------------- #
# SpatialDistribution                                                          #
# --------------------------------------------------------------------------- #


def _sdf_kwargs(target):
    return dict(
        reference=[0, 1, 2],
        template=_TEMPLATE,
        target=target,
        n=(8, 8, 8),
        extent=(6.0, 6.0, 6.0),
    )


def test_sdf_reads_orientations_from_frame():
    n = 12
    frame = _random_frame(n, 10.0, seed=3)
    target = list(range(3, n))
    # One (head, tail) row per target atom, in target order.
    attach_orientations(frame, heads=target, tails=[(t + 1) % n for t in target])
    res = SpatialDistribution(**_sdf_kwargs(target))([frame])
    assert np.asarray(res.density).size > 0
    assert res.orientation is not None  # per-voxel mean-orientation field present


def test_sdf_without_orientations_block_is_isotropic():
    n = 12
    frame = _random_frame(n, 10.0, seed=3)
    res = SpatialDistribution(**_sdf_kwargs(list(range(3, n))))([frame])
    assert res.orientation is None


def test_sdf_rejects_orientation_pairs_kwarg():
    n = 12
    with pytest.raises(TypeError):
        SpatialDistribution(
            orientation_pairs=np.zeros((n - 3, 2), dtype=np.int64),
            **_sdf_kwargs(list(range(3, n))),
        )
