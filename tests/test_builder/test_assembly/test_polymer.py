"""Unit tests for :mod:`molpy.builder.assembly._polymer`."""

import pytest

import molpy as mp
from molpy.builder.assembly import GraphAssembler, MonomerLibrary, PolymerBuilder
from molpy.core import fields


def _trifunctional_core() -> mp.Atomistic:
    struct = mp.Atomistic()
    carbon = struct.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    for x, y in ((1.0, 0.0), (-0.5, 0.87), (-0.5, -0.87)):
        oxygen = struct.def_atom(element="O", x=x, y=y, z=0.0)
        struct.def_bond(carbon, oxygen)
        struct.def_bond(
            oxygen,
            struct.def_atom(element="H", x=x * 1.5, y=y * 1.5, z=0.0),
        )
        oxygen[fields.SITE] = "a"
    struct.generate_topology(gen_angle=True, gen_dihedral=True)
    return struct


class TestPolymerBuilder:
    def test_is_the_graph_assembler_with_a_library(self, builder_factory):
        builder = builder_factory()
        assert isinstance(builder, GraphAssembler)
        assert isinstance(builder.library, MonomerLibrary)

    def test_build_linear_stamps_the_requested_residue_count(self, builder_factory):
        chain = builder_factory().build_linear("EO", 6)
        assert sorted({int(atom[fields.RES_ID]) for atom in chain.atoms}) == list(
            range(1, 7)
        )

    def test_build_linear_matches_explicit_topology(self, builder_factory):
        from molpy.builder.assembly import linear_topology

        builder = builder_factory()
        via_helper = builder.build_linear("EO", 5)
        via_build = builder.build(linear_topology(["EO"] * 5))
        assert via_helper.n_atoms == via_build.n_atoms
        assert len(list(via_helper.bonds)) == len(list(via_build.bonds))

    def test_build_sequence_preserves_residue_names(self, eo_factory):
        builder = PolymerBuilder(
            MonomerLibrary({"A": eo_factory(), "B": eo_factory()}),
            mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]"),
        )
        chain = builder.build_sequence(["A", "A", "B"])
        names = {
            int(atom[fields.RES_ID]): str(atom[fields.RES_NAME]) for atom in chain.atoms
        }
        assert [names[index] for index in sorted(names)] == ["A", "A", "B"]

    def test_build_ring_closes_the_cycle(self, builder_factory):
        ring = builder_factory().build_ring("EO", 4)
        assert len(list(ring.bonds)) == len(list(ring.atoms))

    def test_linear_path_has_tree_bond_count(self, builder_factory):
        # Branching CGSmiles strings are gone with Lark; a linear path stays a tree.
        chain = builder_factory().build_linear("EO", 3)
        assert len(list(chain.bonds)) == len(list(chain.atoms)) - 1

    def test_build_star_uses_every_core_site(self, eo_factory):
        builder = PolymerBuilder(
            MonomerLibrary({"X3": _trifunctional_core(), "EO": eo_factory()}),
            mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]"),
        )
        star = builder.build_star("X3", "EO", n_arms=3, arm_length=2)
        assert len({int(atom[fields.RES_ID]) for atom in star.atoms}) == 7

    def test_build_linear_rejects_zero_length(self, builder_factory):
        with pytest.raises(ValueError, match="n >= 1"):
            builder_factory().build_linear("EO", 0)

    def test_build_rejects_a_notation_string_with_the_topology_constructors(
        self, builder_factory
    ):
        with pytest.raises(TypeError) as excinfo:
            builder_factory().build("{[#EO]|3}")  # type: ignore[arg-type]
        message = str(excinfo.value)
        for constructor in ("linear_topology", "ring_topology", "star_topology"):
            assert constructor in message

    def test_build_places_the_expanded_world_once(
        self, builder_factory, recording_placer
    ):
        builder_factory(placer=recording_placer).build_linear("EO", 3)

        assert len(recording_placer.calls) == 1
        placed_world, formed = recording_placer.calls[0]
        assert len({int(a[fields.RES_ID]) for a in placed_world.atoms}) == 3
        assert len(formed) == 2
        assert all(type(i) is int and type(j) is int for i, j in formed)

    def test_placer_error_propagates_out_of_build(
        self, builder_factory, raising_placer
    ):
        builder = builder_factory(placer=raising_placer)
        with pytest.raises(raising_placer.error_type, match="cannot place"):
            builder.build_linear("EO", 3)


# ---------------------------------------------------------------------------
# OPLS whole-graph oracle (graph-assembler-02 ac-016 / ac-018)
# ---------------------------------------------------------------------------


# Same reaction string as conftest.ETHER — avoid importing the tests package.
ETHER = "[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]"
