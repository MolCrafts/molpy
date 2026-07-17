"""Unit tests for :mod:`molpy.builder.assembly._topology`."""

import pytest

from molpy.builder.assembly import ExplicitPairSelector, TopologySelector
from molpy.parser.smiles import parse_cgsmiles


class TestTopologySelector:
    def test_residue_ids_are_contiguous_in_notation_order(self):
        topology = parse_cgsmiles("{[#EO]|4}").base_graph
        assert list(TopologySelector.residue_ids(topology).values()) == [1, 2, 3, 4]

    def test_selection_equals_the_resolved_explicit_pairs(
        self, polymer_context_factory
    ):
        _, topology, context = polymer_context_factory("{[#EO]|5}")
        selected = list(TopologySelector(topology).select(context))
        pairs = [
            (binding[context.map_a], binding[context.map_b]) for binding in selected
        ]

        explicit = list(ExplicitPairSelector(pairs).select(context))

        assert [set(binding.values()) for binding in selected] == [
            set(binding.values()) for binding in explicit
        ]

    def test_missing_residue_id_is_rejected(self, polymer_context_factory):
        _, topology, context = polymer_context_factory("{[#EO]|2}")
        for atom in context.world.atoms:
            if atom.handle in context.occurrences[context.comp_a][0].values():
                atom["res_id"] = None
        with pytest.raises(ValueError, match="carries no res_id"):
            list(TopologySelector(topology).select(context))
