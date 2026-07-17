"""Unit tests for :mod:`molpy.builder.assembly._assembler`."""

import inspect

import molrs
import pytest

import molpy as mp
from molpy.builder.assembly import (
    ExhaustiveSelector,
    Finalization,
    GraphAssembler,
    RandomSelector,
)
from molpy.core import fields

NO_PLUS_O = "[N:1].[O:2]>>[N:1][O:2]"


class TestGraphAssembler:
    def test_selector_is_an_assemble_argument(self):
        assert "selector" in inspect.signature(GraphAssembler.assemble).parameters
        assert "selector" not in inspect.signature(GraphAssembler.__init__).parameters

    def test_assemble_returns_a_reacted_copy(self, no_cloud_factory):
        world = no_cloud_factory(3)
        result = GraphAssembler(mp.Reaction(NO_PLUS_O)).assemble(
            world, ExhaustiveSelector(cutoff=2.0)
        )
        assert len(list(result.bonds)) == 3
        assert len(list(world.bonds)) == 0

    def test_one_instance_accepts_different_pairing_rules(self, no_cloud_factory):
        assembler = GraphAssembler(mp.Reaction(NO_PLUS_O))
        exhaustive = assembler.assemble(
            no_cloud_factory(3), ExhaustiveSelector(cutoff=2.0)
        )
        random = assembler.assemble(
            no_cloud_factory(3),
            RandomSelector(conversion=1.0, seed=1, cutoff=2.0),
        )
        assert len(list(exhaustive.bonds)) == len(list(random.bonds)) == 3

    def test_empty_selection_warns_and_returns_an_unchanged_copy(
        self, no_cloud_factory
    ):
        world = no_cloud_factory(2)
        with pytest.warns(UserWarning, match="selected no bindings"):
            result = GraphAssembler(mp.Reaction(NO_PLUS_O)).assemble(
                world, ExhaustiveSelector(cutoff=0.5)
            )
        assert result is not world
        assert not list(result.bonds)

    def test_reaction_type_is_validated_at_construction(self):
        with pytest.raises(TypeError, match="must be a molpy.Reaction"):
            GraphAssembler(NO_PLUS_O)  # type: ignore[arg-type]

    def test_typifier_must_inherit_the_native_base(self):
        class NotATypifier:
            def typify(self, graph):
                return graph

        with pytest.raises(TypeError, match="is not a molrs.Typifier"):
            GraphAssembler(
                mp.Reaction(NO_PLUS_O),
                typifier=NotATypifier(),  # type: ignore[arg-type]
                reach=2,
            )

    def test_native_typifier_is_accepted(self):
        GraphAssembler(
            mp.Reaction(NO_PLUS_O),
            typifier=molrs.typifier.OPLSAATypifier(),
            reach=2,
        )

    def test_typifier_requires_an_explicit_reach(self, element_typifier):
        with pytest.raises(TypeError, match="reach= is required"):
            GraphAssembler(mp.Reaction(NO_PLUS_O), typifier=element_typifier)

    def test_reach_cannot_be_negative(self, element_typifier):
        with pytest.raises(ValueError, match="reach must be >= 0"):
            GraphAssembler(mp.Reaction(NO_PLUS_O), typifier=element_typifier, reach=-1)

    def test_overlapping_bindings_are_rejected(self):
        with pytest.raises(ValueError, match="both name atom"):
            GraphAssembler._assert_disjoint([{1: 10, 2: 11}, {1: 10, 2: 12}])

    def test_missing_forming_bond_map_is_rejected(self):
        with pytest.raises(ValueError, match="appears in no reactant pattern"):
            GraphAssembler._find_component([{1}, {2}], 7)

    def test_uncharged_graph_skips_charge_accounting(self, no_cloud_factory):
        result = GraphAssembler(mp.Reaction(NO_PLUS_O)).assemble(
            no_cloud_factory(1), ExhaustiveSelector()
        )
        assert result.n_atoms == 2

    def test_charged_leaving_group_cannot_silently_change_net_charge(self, eo_factory):
        from molpy.builder.assembly import MonomerLibrary, PolymerBuilder

        builder = PolymerBuilder(
            MonomerLibrary({"EO": eo_factory(charge=0.1)}),
            mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]"),
        )
        with pytest.raises(ValueError, match="changed the net charge"):
            builder.build("{[#EO]|3}")

    def test_atoms_stage_writes_compiled_atom_data_only(
        self, no_cloud_factory, atom_and_bonded_typifier
    ):
        result = GraphAssembler(
            mp.Reaction(NO_PLUS_O),
            typifier=atom_and_bonded_typifier,
            reach=1,
            finalize=Finalization.ATOMS,
        ).assemble(no_cloud_factory(2), ExhaustiveSelector())

        assert all(atom.get(fields.TYPE) for atom in result.atoms)
        assert all(
            atom.get(fields.CHARGE) == pytest.approx(0.125) for atom in result.atoms
        )
        assert all(term.get(fields.TYPE) is None for term in result.bonds)

    def test_constructor_has_no_charge_redistribution_switch(self):
        assert "charges" not in inspect.signature(GraphAssembler.__init__).parameters
