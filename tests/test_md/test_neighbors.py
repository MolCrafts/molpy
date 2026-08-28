"""Tests for MD using the core :class:`molpy.NeighborList`."""

import numpy as np

from molpy import Box, NeighborList
from molpy.md import LJCut


def test_md_neighborlist_is_the_core_engine():
    import molrs

    assert NeighborList is molrs.NeighborList


def test_pair_inside_cutoff_is_half_shell():
    nl = NeighborList(2.5)
    nl.build(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), Box.cubic(20.0))
    neigh = nl.neighbors()
    assert neigh.n_pairs == 1
    pairs = set(zip(neigh.query_point_indices(), neigh.point_indices(), strict=True))
    assert pairs == {(0, 1)}


def test_pair_outside_cutoff_is_absent():
    nl = NeighborList(1.0)
    nl.build(np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]), Box.cubic(20.0))
    assert nl.neighbors().n_pairs == 0


def test_lj_flags_bake_the_kernel():
    cut = LJCut(1.0, 1.0, 2.5, shifted=False, smeared=False)
    assert cut.n == 12 and cut.m == 6
    assert not cut.shifted
    assert not cut.smeared
    shifted = LJCut(1.0, 1.0, 2.5, shifted=True)
    assert shifted.shifted and not shifted.smeared
    smeared = LJCut(1.0, 1.0, 2.5, smeared=True)
    assert smeared.smeared and smeared.shifted


def test_lj_consumes_neighbors_table():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    nl = NeighborList(2.5)
    nl.build(pos, Box.cubic(20.0))
    lj = LJCut(1.0, 1.0, 2.5, shifted=True)
    energy, forces = lj.eval_table(2, nl.neighbors())
    assert forces.shape == (2, 3)
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-12)
    e2, f2 = lj.eval_pairs(
        2,
        nl.neighbors().query_point_indices(),
        nl.neighbors().point_indices(),
        nl.neighbors().disp(),
        nl.neighbors().dist_sq(),
    )
    np.testing.assert_allclose(f2, forces)
    np.testing.assert_allclose(e2, energy)


def test_update_reindexes_moved_points():
    nl = NeighborList(2.5)
    box = Box.cubic(20.0)
    nl.build(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), box)
    assert nl.neighbors().n_pairs == 1
    nl.update(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    assert nl.neighbors().n_pairs == 0
