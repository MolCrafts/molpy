"""Small chemistry fixtures shared by the assembly unit-test mirror."""

from __future__ import annotations

from collections.abc import Callable

import pytest

import molpy as mp
from molpy.builder.assembly import MatchContext, MonomerLibrary, PolymerBuilder
from molpy.core import fields
from molpy.parser.smiles import parse_cgsmiles
from molpy.typifier.base import Match, Typifier

ETHER = "[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]"
NO_PLUS_O = "[N:1].[O:2]>>[N:1][O:2]"


class ElementTypifier(Typifier):
    """Minimal atom-only typifier used as a compiler test double."""

    def match(self, graph) -> Match:
        return Match(
            nodes=tuple(
                {fields.TYPE.key: f"t_{atom[fields.ELEMENT]}"} for atom in graph.atoms
            )
        )


class AtomAndBondedTypifier(Typifier):
    """Returns atom and relation data so write-back boundaries can be tested."""

    def match(self, graph) -> Match:
        return Match(
            nodes=tuple(
                {
                    fields.TYPE.key: f"compiled_{atom[fields.ELEMENT]}",
                    fields.CHARGE.key: 0.125,
                }
                for atom in graph.atoms
            ),
            links={
                cls: tuple(
                    {fields.TYPE.key: "must_not_be_copied"}
                    for _ in graph.links.bucket(cls)
                )
                for cls in graph.links.classes()
            },
        )


class CarbonClassTypifier(Typifier):
    def match(self, graph) -> Match:
        return Match(nodes=tuple({fields.TYPE.key: "CT"} for _ in graph.atoms))


@pytest.fixture
def element_typifier() -> ElementTypifier:
    return ElementTypifier()


@pytest.fixture
def atom_and_bonded_typifier() -> AtomAndBondedTypifier:
    return AtomAndBondedTypifier()


@pytest.fixture
def carbon_class_typifier() -> CarbonClassTypifier:
    return CarbonClassTypifier()


@pytest.fixture
def eo_factory() -> Callable[..., mp.Atomistic]:
    def make(*, typed: bool = True, charge: float | None = None) -> mp.Atomistic:
        struct = mp.Atomistic()
        heavy = [
            struct.def_atom(element=element, x=float(index), y=0.0, z=0.0)
            for index, element in enumerate("OCCO")
        ]
        for left, right in zip(heavy, heavy[1:], strict=False):
            struct.def_bond(left, right)
        for oxygen in (heavy[0], heavy[3]):
            struct.def_bond(
                oxygen,
                struct.def_atom(element="H", x=oxygen["x"], y=1.0, z=0.0),
            )
        heavy[0][fields.SITE] = "a"
        heavy[3][fields.SITE] = "b"
        struct.generate_topology(gen_angle=True, gen_dihedral=True)
        if typed:
            for atom in struct.atoms:
                atom[fields.TYPE] = f"t_{atom[fields.ELEMENT]}"
        if charge is not None:
            for atom in struct.atoms:
                atom[fields.CHARGE] = charge
        return struct

    return make


@pytest.fixture
def no_cloud_factory() -> Callable[[int], mp.Atomistic]:
    def make(count: int = 3) -> mp.Atomistic:
        cloud = mp.Atomistic()
        for index in range(count):
            cloud.def_atom(element="N", x=float(index), y=0.0, z=0.0)
            cloud.def_atom(element="O", x=float(index), y=1.0, z=0.0)
        return cloud

    return make


@pytest.fixture
def builder_factory(eo_factory) -> Callable[..., PolymerBuilder]:
    def make(*, typifier=None, reach=None, library=None, **kwargs) -> PolymerBuilder:
        if typifier is not None and reach is None:
            reach = 2
        return PolymerBuilder(
            library or MonomerLibrary({"EO": eo_factory()}),
            mp.Reaction(ETHER),
            typifier=typifier,
            reach=reach,
            **kwargs,
        )

    return make


@pytest.fixture
def polymer_context_factory(builder_factory):
    def make(cgsmiles: str = "{[#EO]|4}"):
        builder = builder_factory()
        topology = parse_cgsmiles(cgsmiles).base_graph
        world = builder.library.expand(topology)
        labels = builder._labels(world)
        context = MatchContext(
            world=world,
            occurrences=builder._match(world, labels),
            map_a=builder._map_a,
            map_b=builder._map_b,
            comp_a=builder._comp_a,
            comp_b=builder._comp_b,
        )
        return builder, topology, context

    return make
