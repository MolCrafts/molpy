"""Python surface for ``mp.perceive.SubgraphMatcher`` (backmap-primitives-07).

These are FFI-seam tests: they prove that the matcher is published from
``mp.perceive``, that ``find`` hands back live target bead views in
pattern order as ``list[list[Bead]]``, that no match is ``[]``, and that a
non-``CoarseGrain`` target is PyO3's ``TypeError``. Matching semantics (bead
type equality, group enumeration, no partitioning of overlaps) are proven by
the Rust unit tests of ``molrs::perceive::SubgraphMatcher``. Fixtures are
built in process.
"""

from __future__ import annotations

import itertools

import molpy as mp
import pytest


def _chain(*bead_types: str) -> tuple[mp.core.CoarseGrain, list[mp.core.Bead]]:
    """A linear bead chain of ``bead_types``, and its beads in that order."""
    cg = mp.core.CoarseGrain()
    beads = [cg.add_bead(bead_type=bead_type) for bead_type in bead_types]
    for a, b in itertools.pairwise(beads):
        cg.add_bond(a, b)
    return cg, beads


def test_find_returns_one_group_of_live_beads_in_pattern_order() -> None:
    pattern, _ = _chain("1", "4")
    # The target is built type-4 first, so pattern order differs from the
    # target's own bead order.
    target = mp.core.CoarseGrain()
    four = target.add_bead(bead_type="4")
    one = target.add_bead(bead_type="1")
    target.add_bond(four, one)

    groups = mp.perceive.SubgraphMatcher(pattern).find(target)

    assert groups == [[one, four]]
    assert all(bead.world is target for bead in groups[0])


def test_find_without_a_match_is_an_empty_list() -> None:
    pattern, _ = _chain("1", "4")
    target, _ = _chain("1", "1")

    assert mp.perceive.SubgraphMatcher(pattern).find(target) == []


def test_find_on_an_atomistic_target_is_a_type_error() -> None:
    pattern, _ = _chain("1", "4")
    target = mp.core.Atomistic()
    target.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0)

    with pytest.raises(TypeError):
        mp.perceive.SubgraphMatcher(pattern).find(target)
