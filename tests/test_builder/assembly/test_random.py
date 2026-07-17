"""Unit tests for :mod:`molpy.builder.assembly._random`."""

import pytest

import molpy as mp
from molpy.builder.assembly import MatchContext, RandomSelector


def _context(count: int) -> MatchContext:
    graph = mp.Atomistic()
    nitrogens = [graph.def_atom(element="N") for _ in range(count)]
    oxygens = [graph.def_atom(element="O") for _ in range(count)]
    return MatchContext(
        graph,
        [
            [{1: atom.handle} for atom in nitrogens],
            [{2: atom.handle} for atom in oxygens],
        ],
        1,
        2,
        0,
        1,
    )


class TestRandomSelector:
    def test_conversion_is_bounded(self):
        with pytest.raises(ValueError, match=r"conversion must be in \[0, 1\]"):
            RandomSelector(conversion=1.1)

    def test_seed_reproduces_the_same_bindings(self):
        context = _context(count=6)
        first = list(RandomSelector(conversion=0.5, seed=42).select(context))
        second = list(RandomSelector(conversion=0.5, seed=42).select(context))
        assert first == second

    def test_conversion_targets_limiting_reactant_sites(self):
        context = _context(count=5)
        selected = list(RandomSelector(conversion=0.4, seed=1).select(context))
        assert len(selected) == 2
