"""Copy-gating semantics of Reacter.run / BondReactReacter.run.

Perf contract (spec builder-reacter-05-perf):

- Plain ``Reacter`` with ``record_intermediates=False`` makes ZERO copies
  of the merged structure (only the left/right input copies), and
  ``result.reactants`` is ``None``.
- ``BondReactReacter`` (``_needs_reactants_snapshot = True``) makes
  exactly ONE merged-structure copy — the reactants snapshot — and
  ``result.reactants`` stays a full Atomistic.
- ``record_intermediates=True`` still records intermediates.

Merged-size copies are discriminated by source atom count: left (7) and
right (5) fragments have different sizes, so only post-merge,
pre-removal copies have exactly ``N_LEFT + N_RIGHT`` atoms.
"""

import pytest

from molpy.core.atomistic import Atom, Atomistic, Bond
from molpy.reacter import (
    Reacter,
    form_single_bond,
    select_one_hydrogen,
    select_port,
)
from molpy.reacter.bond_react import BondReactReacter
from molpy.reacter.selectors import find_port

# ── fixture builders (pattern from test_bond_react.py) ───────────────

N_LEFT = 7
N_RIGHT = 5
N_MERGED = N_LEFT + N_RIGHT


def _build_left() -> Atomistic:
    """Left reactant (7 atoms): h01-c0-c1(-h1)-c2(>)(h21)(h22)."""
    struct = Atomistic()
    c0 = Atom(element="C", type="CT", charge=-0.18)
    h01 = Atom(element="H", type="HC", charge=0.06)
    c1 = Atom(element="C", type="CM", charge=-0.12)
    h1 = Atom(element="H", type="HC", charge=0.06)
    c2 = Atom(element="C", type="CT", charge=-0.18)
    h21 = Atom(element="H", type="HC", charge=0.06)
    h22 = Atom(element="H", type="HC", charge=0.06)
    struct.add_entity(c0, h01, c1, h1, c2, h21, h22)
    struct.add_link(
        Bond(c0, h01),
        Bond(c0, c1),
        Bond(c1, h1),
        Bond(c1, c2),
        Bond(c2, h21),
        Bond(c2, h22),
    )
    c2["port"] = ">"
    return struct


def _build_right() -> Atomistic:
    """Right reactant (5 atoms): c3(<)(h31)(h32)-c4-h41."""
    struct = Atomistic()
    c3 = Atom(element="C", type="CT", charge=-0.18)
    h31 = Atom(element="H", type="HC", charge=0.06)
    h32 = Atom(element="H", type="HC", charge=0.06)
    c4 = Atom(element="C", type="CT", charge=-0.18)
    h41 = Atom(element="H", type="HC", charge=0.06)
    struct.add_entity(c3, h31, h32, c4, h41)
    struct.add_link(
        Bond(c3, h31),
        Bond(c3, h32),
        Bond(c3, c4),
        Bond(c4, h41),
    )
    c3["port"] = "<"
    return struct


def _make_plain_reacter() -> Reacter:
    return Reacter(
        name="cc_coupling",
        anchor_selector_left=select_port,
        anchor_selector_right=select_port,
        leaving_selector_left=select_one_hydrogen,
        leaving_selector_right=select_one_hydrogen,
        bond_former=form_single_bond,
    )


def _make_bond_react_reacter() -> BondReactReacter:
    return BondReactReacter(
        name="cc_coupling_template",
        anchor_selector_left=select_port,
        anchor_selector_right=select_port,
        leaving_selector_left=select_one_hydrogen,
        leaving_selector_right=select_one_hydrogen,
        bond_former=form_single_bond,
        radius=4,
    )


@pytest.fixture
def copy_sizes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the atom count of every Atomistic.copy() source."""
    sizes: list[int] = []
    original_copy = Atomistic.copy

    def recording_copy(self: Atomistic) -> Atomistic:
        sizes.append(len(list(self.atoms)))
        return original_copy(self)

    monkeypatch.setattr(Atomistic, "copy", recording_copy)
    return sizes


class TestCopyGating:
    """Merged-structure copy counts under record_intermediates=False."""

    def test_plain_reacter_zero_merged_copies(self, copy_sizes: list[int]) -> None:
        """Plain Reacter must not copy the merged structure at all."""
        left = _build_left()
        right = _build_right()
        reacter = _make_plain_reacter()

        reacter.run(
            left,
            right,
            port_atom_L=find_port(left, ">"),
            port_atom_R=find_port(right, "<"),
            record_intermediates=False,
        )

        merged_copies = copy_sizes.count(N_MERGED)
        assert merged_copies == 0, (
            f"Plain Reacter made {merged_copies} merged-size "
            f"({N_MERGED}-atom) copies; expected 0 with "
            f"record_intermediates=False (copy sizes: {copy_sizes})"
        )

    def test_bond_react_one_merged_copy(self, copy_sizes: list[int]) -> None:
        """BondReactReacter copies the merged structure exactly once."""
        left = _build_left()
        right = _build_right()
        reacter = _make_bond_react_reacter()

        reacter.run(
            left,
            right,
            port_atom_L=find_port(left, ">"),
            port_atom_R=find_port(right, "<"),
            record_intermediates=False,
        )

        merged_copies = copy_sizes.count(N_MERGED)
        assert merged_copies == 1, (
            f"BondReactReacter made {merged_copies} merged-size "
            f"({N_MERGED}-atom) copies; expected exactly 1 (the reactants "
            f"snapshot) with record_intermediates=False "
            f"(copy sizes: {copy_sizes})"
        )


class TestReactantsContract:
    """result.reactants gating between plain and bond/react reacters."""

    def test_plain_reacter_reactants_none(self) -> None:
        """Plain Reacter no longer carries a reactants snapshot."""
        left = _build_left()
        right = _build_right()
        reacter = _make_plain_reacter()

        result = reacter.run(
            left,
            right,
            port_atom_L=find_port(left, ">"),
            port_atom_R=find_port(right, "<"),
            record_intermediates=False,
        )

        assert result.reactants is None, (
            "Plain Reacter must not snapshot reactants "
            f"(got {type(result.reactants).__name__})"
        )

    def test_bond_react_reactants_snapshot(self) -> None:
        """BondReactReacter keeps the full pre-reaction snapshot."""
        left = _build_left()
        right = _build_right()
        reacter = _make_bond_react_reacter()

        result = reacter.run(
            left,
            right,
            port_atom_L=find_port(left, ">"),
            port_atom_R=find_port(right, "<"),
            record_intermediates=False,
        )

        assert isinstance(result.reactants, Atomistic)
        assert len(list(result.reactants.atoms)) == N_MERGED


class TestRecordIntermediates:
    """record_intermediates=True keeps working after the copy gating."""

    def test_record_intermediates_still_works(self) -> None:
        left = _build_left()
        right = _build_right()
        reacter = _make_plain_reacter()

        result = reacter.run(
            left,
            right,
            port_atom_L=find_port(left, ">"),
            port_atom_R=find_port(right, "<"),
            record_intermediates=True,
        )

        assert len(result.intermediates) > 0
        for entry in result.intermediates:
            assert isinstance(entry["product"], Atomistic), (
                f"intermediate step {entry.get('step')!r} has no Atomistic product"
            )
