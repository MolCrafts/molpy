"""Nullable Frame columns at the Python seam.

A node component is per-atom optional — an atom either carries ``frag_id`` /
``h_count`` or it does not — while a :class:`molrs.core.Block` column is dense. The
seam therefore needs a validity mask: ``Block.validity(key)`` is ``None`` when
every cell is real, and a boolean array marking the holes when some are not.
Without it a hole crosses as a zero, which is a *stated* value: a partially
labelled fragment loses its labels and a partially declared ``h_count`` tells
hydrogen repletion that the undeclared atoms are saturated.

These are FFI-seam tests. The chemistry they lean on is owned by the Rust unit
tests (``molrs/src/core/molgraph.rs`` for ``frag_id``,
``molrs/src/perceive/hydrogens.rs`` for repletion); nothing here re-derives a
number, and no third-party scientific software runs.
"""

from __future__ import annotations

import pickle

import molrs
import numpy as np
import pytest


def _partially_labelled_fragment() -> molrs.core.Atomistic:
    """``C–C–H`` with a ``frag_id`` on the first carbon only.

    The two unlabelled atoms are the holes under test: ``frag_id`` is a node
    prop, and only one node has it.
    """
    fragment = molrs.core.Atomistic()
    first = fragment.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0)
    second = fragment.add_atom(atomic_number=6, x=1.54, y=0.0, z=0.0)
    hydrogen = fragment.add_atom(atomic_number=1, x=2.63, y=0.0, z=0.0)
    fragment.add_bond(first, second)
    fragment.add_bond(second, hydrogen)
    first["frag_id"] = 7
    return fragment


# ---------------------------------------------------------------------------
# Block.validity
# ---------------------------------------------------------------------------


def test_a_fully_valid_column_has_no_validity_mask() -> None:
    """``None``, not an all-``True`` array: a mask means "this column has
    holes", so a dense column must not manufacture one."""
    block = molrs.core.Block()
    block.insert("x", np.array([1.0, 2.0, 3.0], dtype=np.float64))

    assert block.validity("x") is None


def test_a_column_with_holes_crosses_as_a_boolean_mask() -> None:
    frame = _partially_labelled_fragment().to_frame()

    mask = np.asarray(frame["atoms"].validity("frag_id"))

    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, False, False]


def test_a_fully_labelled_column_has_no_validity_mask() -> None:
    fragment = _partially_labelled_fragment()
    for atom in fragment.atoms:
        atom["frag_id"] = 7

    assert fragment.to_frame()["atoms"].validity("frag_id") is None


# ---------------------------------------------------------------------------
# frag_id survives the Frame round trip with its holes
# ---------------------------------------------------------------------------


def test_partial_frag_id_survives_the_frame_round_trip() -> None:
    """The labelled atom keeps its id and the unlabelled ones stay unlabelled.

    ``to_frame`` writes a partially labelled ``frag_id`` column with a
    validity mask, so ``from_frame`` restores the label where it was set and
    leaves the unlabelled atoms without one.
    """
    frame = _partially_labelled_fragment().to_frame()

    restored = molrs.core.Atomistic.from_frame(frame)

    assert [atom.get("frag_id") for atom in restored.atoms] == [
        7,
        None,
        None,
    ]


# ---------------------------------------------------------------------------
# h_count: a hole is not a declared zero
# ---------------------------------------------------------------------------


def test_hydrogen_repletion_survives_a_frame_round_trip() -> None:
    """``CC[OH]`` declares ``h_count`` on the bracket oxygen only.

    The other two atoms carry no ``h_count`` at all, so repletion derives
    theirs from valence. Crossing the frame must not turn those holes into a
    declared ``h_count = 0``, which an explicit count short-circuits
    (``n_implicit_hydrogens``) into "already saturated".
    """
    molecule = molrs.io.smiles.read_smiles("CC[OH]").to_atomistic()

    direct = molrs.perceive.add_hydrogens(molecule).n_atoms
    round_tripped = molrs.perceive.add_hydrogens(
        molrs.core.Atomistic.from_frame(molecule.to_frame())
    ).n_atoms

    assert round_tripped == direct


def test_hydrogen_repletion_survives_a_round_trip_without_declared_counts() -> None:
    """``CCO`` declares no ``h_count`` anywhere, so the column is absent rather
    than holed — the case the mask must leave alone."""
    molecule = molrs.io.smiles.read_smiles("CCO").to_atomistic()

    direct = molrs.perceive.add_hydrogens(molecule).n_atoms
    round_tripped = molrs.perceive.add_hydrogens(
        molrs.core.Atomistic.from_frame(molecule.to_frame())
    ).n_atoms

    assert round_tripped == direct


# ---------------------------------------------------------------------------
# A mask is part of the column: copying and pickling must carry it
# ---------------------------------------------------------------------------


def test_a_masked_column_survives_block_copy() -> None:
    """``copy`` rebuilds the block column by column from ``view``, which hands
    out the dense buffer only — so the holes silently become stated zeros in
    the copy while the original still knows they are holes."""
    block = _partially_labelled_fragment().to_frame()["atoms"]
    live = block.validity("frag_id")
    assert live is not None  # guarded by the mask tests above

    copied = block.copy().validity("frag_id")

    assert copied is not None
    assert np.asarray(copied).tolist() == np.asarray(live).tolist()


def test_a_masked_column_survives_block_pickling() -> None:
    """``Block.__reduce__`` reconstructs from ``view`` alone, so the unpickled
    block cannot tell a hole from a zero."""
    block = _partially_labelled_fragment().to_frame()["atoms"]
    live = block.validity("frag_id")
    assert live is not None

    restored = pickle.loads(pickle.dumps(block, protocol=pickle.HIGHEST_PROTOCOL))
    mask = restored.validity("frag_id")

    assert mask is not None
    assert np.asarray(mask).tolist() == np.asarray(live).tolist()


def test_a_masked_column_survives_frame_pickling() -> None:
    """The same hole, reached through the Frame: pickling a whole frame must
    not flatten a masked block inside it."""
    frame = _partially_labelled_fragment().to_frame()
    live = frame["atoms"].validity("frag_id")
    assert live is not None

    restored = pickle.loads(pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL))
    mask = restored["atoms"].validity("frag_id")

    assert mask is not None
    assert np.asarray(mask).tolist() == np.asarray(live).tolist()


# ---------------------------------------------------------------------------
# An absent column is a question, not an answer
# ---------------------------------------------------------------------------


def test_validity_of_an_absent_column_raises_key_error() -> None:
    """``None`` means "this column has no holes", so answering ``None`` for a
    column that does not exist merges a typo with a dense column. ``view`` and
    ``dtype`` already raise ``KeyError`` for the same key; ``validity`` is the
    third reader of the same column and must agree.

    The present, dense column keeps answering ``None`` — see
    ``test_a_fully_valid_column_has_no_validity_mask``; it is repeated here so
    the two answers are pinned side by side.
    """
    block = molrs.core.Block()
    block.insert("x", np.array([1.0, 2.0, 3.0], dtype=np.float64))

    assert block.validity("x") is None
    with pytest.raises(KeyError):
        block.validity("no_such_column")


# ---------------------------------------------------------------------------
# Writing a masked column from Python
# ---------------------------------------------------------------------------


def test_insert_stores_the_mask_it_is_given() -> None:
    """The write side of :meth:`validity`: Python can state which cells are
    holes instead of having to route through a ported graph to get a mask."""
    block = molrs.core.Block()

    block.insert(
        "tag", np.array([1, 2, 3], dtype=np.int32), np.array([True, False, True])
    )

    mask = block.validity("tag")
    assert mask is not None
    assert np.asarray(mask).tolist() == [True, False, True]


def test_insert_refuses_a_mask_shorter_than_the_column() -> None:
    """A mask that does not cover every row cannot say which cells are holes,
    and must be refused rather than padded — and the block is left unchanged."""
    block = molrs.core.Block()

    with pytest.raises(ValueError):
        block.insert("x", np.array([1, 2, 3], dtype=np.int64), np.array([True, False]))
    assert "x" not in block


def test_an_all_true_mask_is_dropped() -> None:
    """An all-``True`` mask states nothing, so ``validity`` keeps answering
    ``None``."""
    block = molrs.core.Block()

    block.insert("x", np.array([1.0, 2.0]), [True, True])

    assert block.validity("x") is None


def test_insert_without_a_mask_replaces_a_masked_column_whole() -> None:
    """An insert replaces the column outright, mask included: the rows the
    old mask described are gone."""
    block = molrs.core.Block()
    block.insert("tag", np.array([1, 0], dtype=np.int32), [True, False])

    block.insert("tag", np.array([3, 4], dtype=np.int32))

    assert block.validity("tag") is None


def test_insert_refuses_a_mask_that_is_not_bool() -> None:
    block = molrs.core.Block()

    with pytest.raises(TypeError):
        block.insert("x", np.array([1.0, 2.0]), np.array([1, 0]))
