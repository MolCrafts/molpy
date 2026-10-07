"""Shared builders for the compute tests."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike

import molpy as mp


@pytest.fixture
def random_periodic_frame():
    """Factory: ``n`` uniformly random points in a cubic periodic box."""

    def build(n: int = 200, box_len: float = 12.0, seed: int = 0) -> mp.Frame:
        rng = np.random.default_rng(seed)
        xyz = rng.uniform(0.0, box_len, size=(n, 3))
        frame = mp.Frame()
        frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
        frame.box = mp.Box.cube(box_len)
        return frame

    return build


@pytest.fixture
def self_neighbors():
    """Factory: the half-shell neighbour table of a frame within ``cutoff``."""

    def search(frame: mp.Frame, cutoff: float) -> mp.core.Neighbors:
        nl = mp.core.NeighborList(cutoff)
        nl.build(frame.coords, frame.box)
        return nl.neighbors()

    return search


@pytest.fixture
def frame_coords_snapshot():
    """Factory: an owned (n, 3) copy of a frame's coordinates."""

    def snapshot(frame: mp.Frame) -> np.ndarray:
        block = frame["atoms"]
        return np.column_stack([block["x"], block["y"], block["z"]]).copy()

    return snapshot


@pytest.fixture
def orientations_frame():
    """Factory: attach an ``orientations`` topology block to a frame in place.

    One ``(head, tail)`` atom-index row per particle, using the same on-disk
    schema as the core ``bonds`` block — the two endpoint columns ``atomi``
    (head) and ``atomj`` (tail), stored as unsigned-int atom indices. The
    orientation-aware compute ops (Nematic / SpatialDistribution / PmftXy) read
    their per-particle axis ``normalize(pos[head] - pos[tail])`` from this block.
    """

    def attach(frame: mp.Frame, heads: ArrayLike, tails: ArrayLike) -> mp.Frame:
        frame["orientations"] = {
            "atomi": np.asarray(heads, dtype=np.uint32),
            "atomj": np.asarray(tails, dtype=np.uint32),
        }
        return frame

    return attach


@pytest.fixture
def axis_frame(orientations_frame):
    """Factory: ``2 * n_particles`` atoms; particle ``k``'s axis is atoms ``(2k+1, 2k)``.

    The ``orientations`` block holds one ``(head, tail)`` row per particle with
    ``head = 2k+1`` displaced ``+z`` from ``tail = 2k``, so every axis points
    along ``+z`` (a near-perfectly aligned nematic ensemble).
    """

    def build(n_particles: int = 8, box_len: float = 10.0, seed: int = 0) -> mp.Frame:
        rng = np.random.default_rng(seed)
        n = 2 * n_particles
        xyz = rng.uniform(0.0, box_len, size=(n, 3))
        for k in range(n_particles):
            xyz[2 * k + 1] = xyz[2 * k] + np.array([0.0, 0.0, 1.0])
        frame = mp.Frame()
        frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
        frame.box = mp.Box.cube(box_len)
        heads = [2 * k + 1 for k in range(n_particles)]
        tails = [2 * k for k in range(n_particles)]
        orientations_frame(frame, heads=heads, tails=tails)
        return frame

    return build
