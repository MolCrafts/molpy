"""Unit tests for :mod:`molpy.builder.assembly._proximity`."""

import inspect

import pytest

import molpy as mp
from molpy.builder.assembly import (
    ExhaustiveSelector,
    ExplicitPairSelector,
    MatchContext,
    ProximitySelector,
    SpacingSelector,
)
from molpy.builder.assembly._proximity import Candidate
from molpy.core import fields


def _context(*, count: int = 3, gap: float = 1.0) -> MatchContext:
    graph = mp.Atomistic()
    nitrogens = []
    oxygens = []
    for index in range(count):
        nitrogens.append(
            graph.def_atom(
                element="N",
                x=float(index) * gap,
                y=0.0,
                z=0.0,
                mol_id=index + 1,
            )
        )
        oxygens.append(
            graph.def_atom(
                element="O",
                x=float(index) * gap,
                y=1.0,
                z=0.0,
                mol_id=index + 1,
            )
        )
    return MatchContext(
        world=graph,
        occurrences=[
            [{1: atom.handle} for atom in nitrogens],
            [{2: atom.handle} for atom in oxygens],
        ],
        map_a=1,
        map_b=2,
        comp_a=0,
        comp_b=1,
    )


class TestCandidate:
    def test_binding_merges_the_two_occurrences(self):
        candidate = Candidate({1: 4}, {2: 8}, 1.5)
        assert candidate.binding() == {1: 4, 2: 8}


class TestProximitySelector:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            ProximitySelector()

    def test_missing_distance_sorts_after_real_distance(self):
        missing = Candidate({1: 1}, {2: 2}, None)
        measured = Candidate({1: 1}, {2: 2}, 3.0)
        assert ProximitySelector._sort_key(measured) < ProximitySelector._sort_key(
            missing
        )

    def test_cutoff_requires_coordinates(self):
        graph = mp.Atomistic()
        n = graph.def_atom(element="N")
        o = graph.def_atom(element="O")
        context = MatchContext(graph, [[{1: n.handle}], [{2: o.handle}]], 1, 2, 0, 1)
        with pytest.raises(ValueError, match="requires atom coordinates"):
            list(ExhaustiveSelector(cutoff=2.0).select(context))

    def test_same_molecule_filter_uses_connected_components(self):
        context = _context(count=2)
        atoms = list(context.world.atoms)
        context.world.def_bond(atoms[0], atoms[1])
        selected = list(ExhaustiveSelector(exclude_same_molecule=True).select(context))
        assert all(
            binding != {1: atoms[0].handle, 2: atoms[1].handle} for binding in selected
        )


class TestExhaustiveSelector:
    def test_consumes_each_site_at_most_once(self):
        selected = list(ExhaustiveSelector().select(_context(count=4)))
        handles = [handle for binding in selected for handle in binding.values()]
        assert len(selected) == 4
        assert len(handles) == len(set(handles))


class TestExplicitPairSelector:
    def test_selects_named_pairs_in_caller_order(self):
        context = _context(count=3)
        ns = [entry[1] for entry in context.occurrences[0]]
        os = [entry[2] for entry in context.occurrences[1]]
        selected = list(
            ExplicitPairSelector([(ns[2], os[0]), (ns[0], os[2])]).select(context)
        )
        assert selected == [{1: ns[2], 2: os[0]}, {1: ns[0], 2: os[2]}]

    def test_rejects_pair_that_is_not_a_candidate(self):
        with pytest.raises(ValueError, match="not matched site pairings"):
            list(ExplicitPairSelector([(999, 1000)]).select(_context(count=2)))

    def test_constructor_does_not_accept_spacing(self):
        assert (
            "spacing" not in inspect.signature(ExplicitPairSelector.__init__).parameters
        )


class TestSpacingSelector:
    def test_spacing_must_be_positive(self):
        with pytest.raises(ValueError, match="spacing must be >= 1"):
            SpacingSelector(0)

    def test_constructor_does_not_accept_explicit_pairs(self):
        assert "pairs" not in inspect.signature(SpacingSelector.__init__).parameters

    def test_larger_spacing_selects_fewer_sites(self):
        graph = mp.Atomistic()
        chains: list[list[object]] = []
        for chain_index in range(2):
            chain = []
            for index in range(8):
                atom = graph.def_atom(
                    element="C",
                    x=float(index),
                    y=float(chain_index) * 2.0,
                    z=0.0,
                    mol_id=chain_index + 1,
                )
                if chain:
                    graph.def_bond(chain[-1], atom)
                chain.append(atom)
            chains.append(chain)
        occurrences = [[{1: atom.handle} for chain in chains for atom in chain]]
        context = MatchContext(
            graph,
            [*occurrences, [{2: entry[1]} for entry in occurrences[0]]],
            1,
            2,
            0,
            1,
        )

        dense = list(SpacingSelector(2, cutoff=2.5).select(context))
        sparse = list(SpacingSelector(5, cutoff=2.5).select(context))

        assert len(dense) > len(sparse)
