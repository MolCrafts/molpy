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


# ---------------------------------------------------------------------------
# OPLS whole-graph oracle (graph-assembler-02 ac-016 / ac-018)
# ---------------------------------------------------------------------------


import molrs

from molpy.builder.assembly import MonomerLibrary, PolymerBuilder
from molpy.core import fields
from molpy.core.atomistic import Atomistic
from molpy.typifier.base import Match, Typifier

# Same reaction string as conftest.ETHER — avoid importing the tests package.
ETHER = "[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]"


class ElementTypifier(Typifier):
    """Minimal atom-only typifier for scaling checks (mirrors conftest)."""

    def match(self, graph) -> Match:
        return Match(
            nodes=tuple(
                {fields.TYPE.key: f"t_{atom[fields.ELEMENT]}"} for atom in graph.atoms
            )
        )


class OplsTypifier(Typifier[Atomistic]):
    """molpy Typifier façade over molrs.OPLSAATypifier."""

    def __init__(self) -> None:
        self._inner = molrs.OPLSAATypifier()

    def typify(self, graph: Atomistic) -> Atomistic:
        typed = self._inner.typify(graph)
        return typed if isinstance(typed, Atomistic) else Atomistic.adopt(typed)

    def match(self, graph: Atomistic) -> Match:
        typed = self.typify(graph)
        return Match(
            nodes=tuple({fields.TYPE.key: str(atom["type"])} for atom in typed.atoms)
        )


def _eo_monomer_opls_types() -> Atomistic:
    """Fully hydrogenated EO with OPLS types; charges zeroed for condensation."""
    struct = mp.Atomistic()
    heavy: list = []
    for index, element in enumerate(["O", "C", "C", "O"]):
        heavy.append(struct.def_atom(element=element, x=float(index), y=0.0, z=0.0))
    for left, right in zip(heavy[:-1], heavy[1:], strict=True):
        struct.def_bond(left, right)
    for oxygen, dx in ((heavy[0], -0.5), (heavy[3], 0.5)):
        struct.def_bond(
            oxygen,
            struct.def_atom(element="H", x=oxygen["x"] + dx, y=1.0, z=0.0),
        )
    for carbon in (heavy[1], heavy[2]):
        for dy in (1.0, -1.0):
            struct.def_bond(
                carbon,
                struct.def_atom(element="H", x=carbon["x"], y=dy, z=0.5),
            )
    heavy[0][fields.SITE] = "a"
    heavy[3][fields.SITE] = "b"
    struct.generate_topology(gen_angle=True, gen_dihedral=True)
    typed = OplsTypifier().typify(struct)
    for atom in typed.atoms:
        if "charge" in atom:
            atom["charge"] = 0.0
    oxygens = [atom for atom in typed.atoms if atom["element"] == "O"]
    oxygens[0][fields.SITE] = "a"
    oxygens[-1][fields.SITE] = "b"
    return typed


class TestPolymerBuilderOplsOracle:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_opls_types_match_whole_graph_oracle(self, n: int):
        typifier = OplsTypifier()
        builder = PolymerBuilder(
            MonomerLibrary({"EO": _eo_monomer_opls_types()}),
            mp.Reaction(ETHER),
            typifier=typifier,
            reach=2,
        )
        product = builder.build_linear("EO", n)
        assert all("type" in atom for atom in product.atoms)
        oracle_src = product.copy()
        for atom in oracle_src.atoms:
            if "charge" in atom:
                atom["charge"] = 0.0
        oracle = typifier.typify(oracle_src)
        assert [str(a["type"]) for a in product.atoms] == [
            str(a["type"]) for a in oracle.atoms
        ]

    def test_assembler_source_has_no_force_field_code_ids(self):
        import ast
        import inspect

        from molpy.builder.assembly import _assembler, _plan

        for module in (_assembler, _plan):
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in {
                    "OPLS",
                    "OPLSAA",
                    "AmberTools",
                }:
                    raise AssertionError(f"force-field name used as code id: {node.id}")
