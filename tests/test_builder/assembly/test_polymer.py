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

    def test_build_linear_is_only_notation_sugar(self, builder_factory):
        builder = builder_factory()
        via_helper = builder.build_linear("EO", 5)
        via_build = builder.build("{[#EO]|5}")
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

    def test_difunctional_monomer_cannot_create_a_branch(self, builder_factory):
        chain = builder_factory().build("{[#EO]([#EO])[#EO]}")
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
