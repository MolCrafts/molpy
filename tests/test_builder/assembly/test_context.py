"""Unit tests for :mod:`molpy.builder.assembly._context`."""

import molpy as mp
from molpy.builder.assembly import MatchContext


class TestMatchContext:
    def test_sites_deduplicates_by_bonding_atom_and_sorts_handles(self):
        graph = mp.Atomistic()
        first = graph.def_atom(element="N")
        second = graph.def_atom(element="N")
        duplicate = {1: second.handle, 7: 99}
        context = MatchContext(
            world=graph,
            occurrences=[[duplicate, {1: first.handle}, {1: second.handle}]],
            map_a=1,
            map_b=2,
            comp_a=0,
            comp_b=0,
        )

        assert context.sites(0, 1) == [{1: first.handle}, duplicate]
