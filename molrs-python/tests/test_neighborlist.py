"""Seam tests for the neighbor-search surface: ``NeighborList``, ``Neighbors``.

Mirrors the Rust engine contract (``molrs::core::NeighborList``) across the
PyO3 boundary:

* ``NeighborList`` is the **engine** — cutoff + backend; ``build`` / ``update``
  index coordinates and enumerate nothing, ``neighbors(...)`` materializes.
* ``Neighbors`` is the **table** — read-only columns; ``dist_sq`` and ``disp``
  are opt-in and a column that was not stored reads as ``None``, never as a
  fabricated zero array.
* A self search is **half-shell**: each unordered pair appears once, ``i < j``.
  ``FULL`` (the binder default) names *columns*, not a bidirectional list.
* ``NeighborQuery`` stays the **cross** door: directed, no ``i < j`` rule.

Numbers here are hand-derived from tiny configurations (unit square, unit
octahedron, a pair straddling a periodic face); numerical depth lives in the
Rust unit tests. Nothing in this file needs third-party scientific software.
"""

from __future__ import annotations

import molrs
import numpy as np
import pytest
from numpy.typing import NDArray

Points = NDArray[np.float64]

# --------------------------------------------------------------------------
# Hand-derived configurations (all coordinates in Å, all boxes cubic 10 Å)
# --------------------------------------------------------------------------


def _two_close() -> Points:
    """Two atoms 1.0 Å apart along +x — exactly one pair below a 2 Å cutoff."""
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)


def _two_far() -> Points:
    """Two atoms sqrt(75) ≈ 8.66 Å apart — no pair below a 1 Å cutoff."""
    return np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)


def _pair_across_face() -> Points:
    """Two atoms 0.2 Å apart *through* the x face of a 10 Å cube (9.8 Å direct)."""
    return np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]], dtype=np.float64)


def _unit_square() -> Points:
    """Four atoms on a 1 Å square in the z=5 plane.

    Edges are 1.0 Å (pairs 0-1, 0-2, 1-3, 2-3); diagonals are sqrt(2) ≈ 1.414 Å
    (pairs 0-3, 1-2). A 1.2 Å cutoff therefore keeps the 4 edges and drops the
    2 diagonals; a 1.5 Å cutoff keeps all 6.
    """
    return np.array(
        [
            [5.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
            [5.0, 6.0, 5.0],
            [6.0, 6.0, 5.0],
        ],
        dtype=np.float64,
    )


def _octahedron() -> Points:
    """Centre atom plus its six ±x/±y/±z neighbors at 1 Å."""
    c = 5.0
    return np.array(
        [
            [c, c, c],
            [c + 1.0, c, c],
            [c - 1.0, c, c],
            [c, c + 1.0, c],
            [c, c - 1.0, c],
            [c, c, c + 1.0],
            [c, c, c - 1.0],
        ],
        dtype=np.float64,
    )


def _pair_set(neighbors: molrs.core.Neighbors) -> set[tuple[int, int]]:
    """Materialized table -> ``{(i, j), ...}`` with Python ints (order-free)."""
    qi = neighbors.query_point_indices()
    pi = neighbors.point_indices()
    return {(int(a), int(b)) for a, b in zip(qi, pi)}


# --------------------------------------------------------------------------
# NeighborList — construction, build, pair goldens
# --------------------------------------------------------------------------


class TestNeighborListBuild:
    def test_two_close_atoms_give_one_pair(self, cubic_box: molrs.core.Box) -> None:
        nl = molrs.core.NeighborList(2.0)
        nl.build(_two_close(), cubic_box)
        assert nl.neighbors().n_pairs == 1

    def test_build_rejects_points_that_are_not_n_by_3(
        self, cubic_box: molrs.core.Box
    ) -> None:
        nl = molrs.core.NeighborList(2.0)
        with pytest.raises(ValueError, match=r"\(N,\s*3\)"):
            nl.build(np.ones((3, 2), dtype=np.float64), cubic_box)

    def test_constructor_rejects_non_positive_cutoff(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            molrs.core.NeighborList(0.0)


# --------------------------------------------------------------------------
# Neighbors — column policy, dtypes, half-shell, physics consistency
# --------------------------------------------------------------------------


class TestNeighborsColumns:
    def test_default_materialization_keeps_both_columns(
        self, cubic_box: molrs.core.Box
    ) -> None:
        """ac-003: ``neighbors()`` with no arguments is FULL — no missing column."""
        nl = molrs.core.NeighborList(1.2)
        nl.build(_unit_square(), cubic_box)
        neigh = nl.neighbors()

        assert neigh.n_pairs > 0
        assert neigh.dist_sq() is not None
        assert neigh.disp() is not None

    def test_default_materialization_column_shapes(self, cubic_box: molrs.core.Box) -> None:
        nl = molrs.core.NeighborList(1.2)
        nl.build(_unit_square(), cubic_box)
        neigh = nl.neighbors()

        assert neigh.dist_sq().shape == (neigh.n_pairs,)
        assert neigh.disp().shape == (neigh.n_pairs, 3)

    def test_lean_storage_reports_dropped_disp_as_none(
        self, cubic_box: molrs.core.Box
    ) -> None:
        """A column that was not stored is ``None`` — not zeros, not empty."""
        nl = molrs.core.NeighborList(1.2)
        nl.build(_unit_square(), cubic_box)
        lean = nl.neighbors(dist_sq=True, disp=False)

        assert lean.disp() is None
        assert lean.dist_sq() is not None

    def test_lean_storage_reports_dropped_dist_sq_as_none(
        self, cubic_box: molrs.core.Box
    ) -> None:
        nl = molrs.core.NeighborList(1.2)
        nl.build(_unit_square(), cubic_box)
        lean = nl.neighbors(dist_sq=False, disp=True)

        assert lean.dist_sq() is None
        assert lean.disp() is not None

    def test_index_columns_are_uint32(self, cubic_box: molrs.core.Box) -> None:
        nl = molrs.core.NeighborList(1.2)
        nl.build(_unit_square(), cubic_box)
        neigh = nl.neighbors()

        assert neigh.query_point_indices().dtype == np.uint32
        assert neigh.point_indices().dtype == np.uint32

    def test_dist_sq_is_float64(self, cubic_box: molrs.core.Box) -> None:
        nl = molrs.core.NeighborList(2.0)
        nl.build(_two_close(), cubic_box)

        assert nl.neighbors().dist_sq().dtype == np.float64

    def test_self_search_is_half_shell(self, cubic_box: molrs.core.Box) -> None:
        nl = molrs.core.NeighborList(1.5)
        nl.build(_unit_square(), cubic_box)
        neigh = nl.neighbors()

        assert np.all(neigh.query_point_indices() < neigh.point_indices())

    def test_self_search_is_tagged_as_self_query(self, cubic_box: molrs.core.Box) -> None:
        nl = molrs.core.NeighborList(1.5)
        nl.build(_unit_square(), cubic_box)

        assert nl.neighbors().is_self_query is True

    def test_self_search_reports_one_population(self, cubic_box: molrs.core.Box) -> None:
        nl = molrs.core.NeighborList(1.5)
        nl.build(_unit_square(), cubic_box)
        neigh = nl.neighbors()

        assert neigh.n_points == 4
        assert neigh.n_query_points == 4


# --------------------------------------------------------------------------
# NeighborList.update — re-index in the box captured by build
# --------------------------------------------------------------------------


class TestNeighborListUpdate:
    def test_update_follows_new_coordinates(self, cubic_box: molrs.core.Box) -> None:
        nl = molrs.core.NeighborList(2.0)
        nl.build(_two_close(), cubic_box)
        assert nl.neighbors().n_pairs == 1

        nl.update(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64))
        assert nl.neighbors().n_pairs == 0

    def test_update_before_build_raises_naming_build(self) -> None:
        """The box is unknown before ``build`` — a loud error, never a guess.

        Must be a genuine ``Exception`` subclass (``pytest.raises(Exception)``
        does not catch PyO3's ``PanicException``, which derives from
        ``BaseException``): FFI Rule 1 forbids a panic crossing the seam.
        """
        nl = molrs.core.NeighborList(2.0)
        with pytest.raises(Exception, match="build"):
            nl.update(_two_close())

    def test_update_rejects_points_that_are_not_n_by_3(
        self, cubic_box: molrs.core.Box
    ) -> None:
        nl = molrs.core.NeighborList(2.0)
        nl.build(_two_close(), cubic_box)
        with pytest.raises(ValueError, match=r"\(N,\s*3\)"):
            nl.update(np.ones((3, 2), dtype=np.float64))


# --------------------------------------------------------------------------
# Backend selection — the O(N²) reference must agree with the cell list
# --------------------------------------------------------------------------


class TestNeighborListBackends:
    @pytest.mark.parametrize("cutoff", [1.2, 1.5])
    def test_brute_force_agrees_with_the_cell_list(
        self, cubic_box: molrs.core.Box, cutoff: float
    ) -> None:
        cell = molrs.core.NeighborList(cutoff)
        oracle = molrs.core.NeighborList(cutoff, brute_force=True)
        cell.build(_unit_square(), cubic_box)
        oracle.build(_unit_square(), cubic_box)
        assert _pair_set(cell.neighbors()) == _pair_set(oracle.neighbors())

    def test_brute_force_wraps_across_a_periodic_face(
        self, cubic_box: molrs.core.Box
    ) -> None:
        oracle = molrs.core.NeighborList(0.5, brute_force=True)
        oracle.build(_pair_across_face(), cubic_box)
        assert _pair_set(oracle.neighbors()) == {(0, 1)}

    def test_brute_force_is_keyword_or_second_positional_only(self) -> None:
        with pytest.raises(TypeError):
            molrs.core.NeighborList(1.0, _two_close())  # type: ignore[arg-type]


class TestNeighborListPickle:
    def test_an_unbuilt_engine_round_trips(self) -> None:
        import pickle

        back = pickle.loads(pickle.dumps(molrs.core.NeighborList(2.5)))
        assert back.cutoff == 2.5
        with pytest.raises(Exception, match="build"):
            back.update(_two_close())

    def test_a_built_engine_round_trips_with_its_pairs(
        self, cubic_box: molrs.core.Box
    ) -> None:
        import pickle

        nl = molrs.core.NeighborList(1.2, brute_force=True)
        nl.build(_unit_square(), cubic_box)
        back = pickle.loads(pickle.dumps(nl))
        assert _pair_set(back.neighbors()) == _pair_set(nl.neighbors())
        # The box travelled: update re-indexes without a new build.
        back.update(_unit_square())
        assert back.neighbors().n_pairs == 4


# --------------------------------------------------------------------------
# NeighborQuery — the cross door survives (ac-006)
# --------------------------------------------------------------------------


class TestNeighborQueryCross:
    def test_cross_query_is_not_tagged_as_self_query(
        self, cubic_box: molrs.core.Box
    ) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        query_points = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        nq = molrs.core.NeighborQuery(cubic_box, points, 1.5)

        assert nq.query(query_points).is_self_query is False

    def test_cross_query_keeps_directed_pairs_with_i_greater_than_j(
        self, cubic_box: molrs.core.Box
    ) -> None:
        """Query point 1 (at x=0) neighbors reference point 0 — the pair (1, 0)."""
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        query_points = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        nq = molrs.core.NeighborQuery(cubic_box, points, 1.5)

        assert (1, 0) in _pair_set(nq.query(query_points))

    def test_cross_query_reports_two_populations(self, cubic_box: molrs.core.Box) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        query_points = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        cross = molrs.core.NeighborQuery(cubic_box, points, 1.5).query(query_points)

        assert cross.n_query_points == 2
        assert cross.n_points == 3

    def test_query_self_returns_a_half_shell_table(self, cubic_box: molrs.core.Box) -> None:
        neigh = molrs.core.NeighborQuery(cubic_box, _unit_square(), 1.5).query_self()

        assert neigh.is_self_query is True
        assert np.all(neigh.query_point_indices() < neigh.point_indices())

    def test_query_self_table_carries_both_columns(self, cubic_box: molrs.core.Box) -> None:
        neigh = molrs.core.NeighborQuery(cubic_box, _unit_square(), 1.5).query_self()

        assert neigh.dist_sq() is not None
        assert neigh.disp() is not None


# --------------------------------------------------------------------------
# ac-003 runtime: the default path feeds an order parameter without a trap
# --------------------------------------------------------------------------


def test_neighbor_query_in_a_free_bounding_box() -> None:
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
    box = molrs.core.Box.bounding(points, (1.0, 1.0, 1.0), (False, False, False))
    query = molrs.core.NeighborQuery(box, points, 1.0)
    result = query.query(np.array([[0.0, 0.0, 0.0]]))

    assert np.array_equal(result.point_indices(), np.array([0, 1], dtype=np.uint32))
    assert np.allclose(np.sqrt(result.dist_sq()), [0.0, 0.5])


def test_neighbor_query_rejects_non_positive_cutoff() -> None:
    points = np.zeros((1, 3))
    box = molrs.core.Box(molrs.core.Cell.lengths([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="positive"):
        molrs.core.NeighborQuery(box, points, 0.0)
    with pytest.raises(ValueError, match="positive"):
        molrs.core.NeighborQuery(box, points, -1.0)
