"""Small chemistry fixtures shared by the assembly unit-test mirror."""

from __future__ import annotations

from collections.abc import Callable

import pytest

import molpy as mp
from molpy.builder.assembly import MatchContext, MonomerLibrary, PolymerBuilder
from molpy.core import fields
from molpy.builder.assembly._residue_graph import linear_topology
from molpy.typifier.base import Match, Typifier

ETHER = "[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]"
NO_PLUS_O = "[N:1].[O:2]>>[N:1][O:2]"


class ElementTypifier(Typifier):
    """Minimal atom-only typifier used as a compiler test double."""

    def match(self, graph) -> Match:
        return Match(
            nodes=tuple(
                {fields.TYPE: f"t_{atom[fields.ELEMENT]}"} for atom in graph.atoms
            )
        )


class AtomAndBondedTypifier(Typifier):
    """Returns atom and relation data so write-back boundaries can be tested."""

    def match(self, graph) -> Match:
        return Match(
            nodes=tuple(
                {
                    fields.TYPE: f"compiled_{atom[fields.ELEMENT]}",
                    fields.CHARGE: 0.125,
                }
                for atom in graph.atoms
            ),
            links={
                cls: tuple(
                    {fields.TYPE: "must_not_be_copied"} for _ in graph.links.bucket(cls)
                )
                for cls in graph.links.classes()
            },
        )


class CarbonClassTypifier(Typifier):
    def match(self, graph) -> Match:
        return Match(nodes=tuple({fields.TYPE: "CT"} for _ in graph.atoms))


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
def two_junction_factory() -> Callable[..., mp.Atomistic]:
    """Two parallel chains whose two crosslinks sit exactly ``2 * reach`` apart.

    Each chain is five carbons long and carries a labelled site on both ends, so
    the two junctions are four bonds apart along the backbone — the frontier of a
    ``reach=2`` extraction. The leaving hydrogen of the *other* junction is one
    bond further out, which is what a cut that reasons about mapped atoms alone
    drops.
    """

    def make(*, res_id: bool = False) -> mp.Atomistic:
        world = mp.Atomistic()
        for chain, y in enumerate((0.0, 3.0)):
            carbons = [
                world.def_atom(element="C", x=1.5 * index, y=y, z=0.0)
                for index in range(5)
            ]
            for left, right in zip(carbons, carbons[1:], strict=False):
                world.def_bond(left, right)
            members = list(carbons)
            for carbon in (carbons[0], carbons[-1]):
                carbon[fields.SITE] = "x"
                hydrogen = world.def_atom(
                    element="H", x=carbon["x"], y=y + (1.0 if y == 0.0 else -1.0), z=0.0
                )
                hydrogen[fields.SITE] = "h"
                world.def_bond(carbon, hydrogen)
                members.append(hydrogen)
            if res_id:
                # Residue ids repeat across chains, exactly as a per-molecule
                # numbering (a tleap ``sequence``, a PDB) writes them.
                for index, atom in enumerate(members):
                    atom[fields.RES_ID] = index // 3
                    atom[fields.MOL_ID] = chain
        return world

    return make


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
    def make(n: int = 4, label: str = "EO"):
        builder = builder_factory()
        topology = linear_topology([label] * n)
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
