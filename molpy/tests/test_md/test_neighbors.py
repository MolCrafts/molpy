"""Tests for MD using the core :class:`molpy.core.NeighborList`."""

import numpy as np
import molpy as mp


def test_pair_inside_cutoff_is_half_shell():
    nl = mp.core.NeighborList(2.5)
    nl.build(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        mp.Box(mp.core.Cell.lengths([20.0] * 3)),
    )
    neigh = nl.neighbors()
    assert neigh.n_pairs == 1
    pairs = set(zip(neigh.query_point_indices(), neigh.point_indices(), strict=True))
    assert pairs == {(0, 1)}


def test_pair_outside_cutoff_is_absent():
    nl = mp.core.NeighborList(1.0)
    nl.build(
        np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        mp.Box(mp.core.Cell.lengths([20.0] * 3)),
    )
    assert nl.neighbors().n_pairs == 0


def test_lj_flags_bake_the_kernel():
    cut = mp.ff.potential.PairLjCut(1.0, 1.0, 2.5, shifted=False, smeared=False)
    assert cut.n == 12 and cut.m == 6
    assert not cut.shifted
    assert not cut.smeared
    shifted = mp.ff.potential.PairLjCut(1.0, 1.0, 2.5, shifted=True)
    assert shifted.shifted and not shifted.smeared
    smeared = mp.ff.potential.PairLjCut(1.0, 1.0, 2.5, smeared=True)
    assert smeared.smeared and smeared.shifted


def test_lj_consumes_neighbors_table():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    nl = mp.core.NeighborList(2.5)
    nl.build(pos, mp.Box(mp.core.Cell.lengths([20.0] * 3)))
    lj = mp.ff.potential.PairLjCut(1.0, 1.0, 2.5, shifted=True)
    energy, forces = lj.evaluate_pairs(pos, nl.neighbors())
    assert forces.shape == (2, 3)
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-12)
    e2, f2 = lj.evaluate_pairs(pos, nl.neighbors())
    np.testing.assert_allclose(f2, forces)
    np.testing.assert_allclose(e2, energy)


def test_update_reindexes_moved_points():
    nl = mp.core.NeighborList(2.5)
    box = mp.Box(mp.core.Cell.lengths([20.0] * 3))
    nl.build(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), box)
    assert nl.neighbors().n_pairs == 1
    nl.update(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    assert nl.neighbors().n_pairs == 0
