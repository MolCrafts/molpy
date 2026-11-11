"""FFI seam of ``CloseContacts`` and ``RingPiercing``: arguments in, result
arrays out, refusals as ``ValueError``. The geometry is proven by the unit
tests in ``crates/molrs/src/compute/{contacts,piercing}/tests.rs``."""

from __future__ import annotations

import numpy as np
import pytest

import molpy as mp
from molpy.compute import CloseContacts, RingPiercing

# A four-atom chain along x with 1 Å bonds, and a loose atom 0.5 Å off its end.
CHAIN = np.array([[1.0, 1, 1], [2.0, 1, 1], [3.0, 1, 1], [4.0, 1, 1], [1.0, 1.5, 1]])
CHAIN_BONDS = [[0, 1], [1, 2], [2, 3]]


def test_close_contacts_drop_bonded_pairs() -> None:
    table = mp.core.NeighborQuery(mp.Box.cube(20.0), CHAIN, 1.6).query_self()
    out = CloseContacts(exclude_bonds=1).compute(pairs=table, bonds=CHAIN_BONDS)
    assert out.pairs.tolist() == [[0, 4], [1, 4]]
    np.testing.assert_allclose(out.distances, [0.5, np.sqrt(1.25)], atol=1e-12)
    assert out.n_contacts == 2


def test_close_contacts_refuse_a_bond_past_the_table() -> None:
    table = mp.core.NeighborQuery(mp.Box.cube(20.0), CHAIN, 1.6).query_self()
    with pytest.raises(ValueError):
        CloseContacts(exclude_bonds=1).compute(pairs=table, bonds=[[0, 9]])


def _hexagon_and_bond(offset: float) -> np.ndarray:
    angles = np.arange(6) * np.pi / 3
    ring = np.column_stack(
        [5 + 1.4 * np.cos(angles), 5 + 1.4 * np.sin(angles), np.full(6, 5.0)]
    )
    bond = np.array([[5.0 + offset, 5, 4], [5.0 + offset, 5, 6]])
    return np.vstack([ring, bond])


def test_a_bond_through_a_ring_pierces_it() -> None:
    out = RingPiercing(radius_fraction=1.0).compute(
        positions=_hexagon_and_bond(0.0),
        simbox=mp.Box.cube(20.0),
        bonds=[[6, 7]],
        rings=[list(range(6))],
    )
    assert (out.ring_idx.tolist(), out.bond_idx.tolist()) == ([0], [0])
    assert out.n_piercings == 1


def test_a_bond_beside_a_ring_does_not() -> None:
    out = RingPiercing(radius_fraction=1.0).compute(
        positions=_hexagon_and_bond(4.0),
        simbox=mp.Box.cube(20.0),
        bonds=[[6, 7]],
        rings=[list(range(6))],
    )
    assert out.n_piercings == 0


def test_ring_piercing_refuses_a_fraction_that_is_not_positive() -> None:
    with pytest.raises(ValueError):
        RingPiercing(radius_fraction=0.0)
