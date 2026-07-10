"""Tests for :class:`~molpy.typifier.scope.TypeScope` (graph-assembler-01-reach).

A region retype has two radii, and they are ``reach`` and ``interior_reach +
reach``. These tests pin:

* the arithmetic (ac-004),
* that the write-back set is exactly ``ball(touched, interior_reach)`` and
  nothing outside it is touched (ac-001),
* that region typing reproduces whole-graph typing atom-for-atom once ``reach``
  is big enough, and that the declared ``reach`` is the *smallest* such value
  (ac-002 / ac-003),
* that an untyped interior atom raises unconditionally — not only when the
  typifier happens to expose a ``strict`` flag (ac-011),
* that ``touched`` is validated rather than assumed (ac-006).
"""

from __future__ import annotations

import math

import molrs
import pytest

import molpy as mp
from molpy.typifier.affected_region import AffectedRegion
from molpy.core.atomistic import Atom, Atomistic
from molpy.parser import parse_smiles, smilesir_to_atomistic
from molpy.typifier.region import RegionTypes
from molpy.typifier.scope import TypeScope

# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------


def _carbon_chain(m: int) -> tuple[Atomistic, list[Atom]]:
    """C0-C1-...-C(m-1), one hydrogen each, with coordinates."""
    s = mp.Atomistic()
    carbons: list[Atom] = []
    prev: Atom | None = None
    for i in range(m):
        c = s.def_atom(element="C", x=float(i), y=0.0, z=0.0)
        s.def_bond(c, s.def_atom(element="H", x=float(i), y=1.0, z=0.0))
        if prev is not None:
            s.def_bond(prev, c, order=1.0)
        carbons.append(c)
        prev = c
    return s, carbons


class _StubTypifier:
    """Minimal region typifier: declares a scope, types every atom.

    Deliberately exposes **neither** ``atom_typifier`` nor ``strict`` — the
    interior guard must fire without them.
    """

    def __init__(self, reach: int, *, type_within: int | None = None) -> None:
        self._scope = TypeScope(reach=reach)
        # when set, only atoms within this many hops of the edit get a type,
        # so deeper interior atoms come back untyped.
        self._type_within = type_within

    @property
    def scope(self) -> TypeScope:
        return self._scope

    def typify_region(self, region: AffectedRegion) -> RegionTypes:
        before = [dict(atom.data) for atom in region.atoms]
        typed = region.copy()
        for atom, source in zip(typed.atoms, region.atoms, strict=True):
            if (
                self._type_within is not None
                and region.hops[source.handle] > self._type_within
            ):
                continue
            atom["type"] = f"t_{atom['element']}"
        after = [dict(atom.data) for atom in typed.atoms]
        return RegionTypes.capture(region, typed, before, after)


# --------------------------------------------------------------------------
# ac-004 — the arithmetic lives in exactly one place
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reach", "interior_reach", "extract_radius"),
    [(1, 2, 3), (2, 2, 4), (3, 3, 6), (4, 4, 8)],
)
def test_scope_arithmetic(reach, interior_reach, extract_radius):
    scope = TypeScope(reach=reach)
    assert scope.interior_reach == interior_reach
    assert scope.extract_radius == extract_radius
    # extraction is always interior_reach + reach, never anything else.
    assert scope.extract_radius == scope.interior_reach + scope.reach


def test_term_reach_floors_the_write_back_set():
    # A dihedral spans 2 hops from a newly formed bond; those atoms must be
    # typed for the term to be looked up, so interior_reach never drops below 2.
    assert TypeScope.TERM_REACH == 2
    assert TypeScope(reach=1).interior_reach == TypeScope.TERM_REACH


def test_scope_rejects_a_zero_reach():
    with pytest.raises(ValueError, match="reach must be >= 1"):
        TypeScope(reach=0)


# --------------------------------------------------------------------------
# ac-001 — write-back set == ball(touched, interior_reach), nothing outside
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reach", [2, 3])
def test_interior_is_the_ball_of_radius_interior_reach(reach):
    chain, carbons = _carbon_chain(12)
    scope = TypeScope(reach=reach)
    seed = carbons[6]

    region = scope.region(chain, [seed])

    expected = {
        h for h, d in chain.topo_distances(seed.handle, max_hops=scope.interior_reach)
    }
    got = {region.entity_map[atom].handle for atom in region.interior}
    assert got == expected

    # the extracted ball is strictly wider than the write-back set
    assert len(list(region.atoms)) > len(region.interior)


@pytest.mark.parametrize("reach", [2, 3])
def test_write_back_touches_only_the_interior(reach):
    chain, carbons = _carbon_chain(12)
    scope = TypeScope(reach=reach)
    seed = carbons[6]
    typifier = _StubTypifier(reach)

    inside = {
        h for h, d in chain.topo_distances(seed.handle, max_hops=scope.interior_reach)
    }

    region = scope.region(chain, [seed])
    typifier.typify_region(region).apply_to(region)

    for atom in chain.atoms:
        if atom.handle in inside:
            assert atom.get("type") is not None, "interior atom was not written"
        else:
            assert atom.get("type") is None, (
                f"atom {atom.handle} outside ball(touched, {scope.interior_reach}) "
                "was written back"
            )


def test_boundary_is_the_outer_shell_not_the_write_back_complement():
    # The old code wrote back "everything that is not boundary" = ball(R-1).
    # boundary is the shell of the *extracted* ball, far outside the interior.
    chain, carbons = _carbon_chain(12)
    scope = TypeScope(reach=2)  # interior 2, extract 4
    region = scope.region(chain, [carbons[6]])

    interior = {a.handle for a in region.interior}
    boundary = {a.handle for a in region.boundary}
    assert interior.isdisjoint(boundary)
    # there are atoms that are neither: the context shell between them.
    assert len(interior) + len(boundary) < len(list(region.atoms))


# --------------------------------------------------------------------------
# ac-011 — an untyped interior atom raises, whatever the typifier looks like
# --------------------------------------------------------------------------


def test_untyped_interior_atom_raises_without_a_strict_flag():
    chain, carbons = _carbon_chain(12)
    # types only atoms within 1 hop, so the hops==2 interior shell stays untyped
    typifier = _StubTypifier(2, type_within=1)
    assert not hasattr(typifier, "atom_typifier")
    assert not hasattr(typifier, "strict")

    region = typifier.scope.region(chain, [carbons[6]])
    with pytest.raises(ValueError, match="left untyped"):
        typifier.typify_region(region)


def test_fully_typed_interior_does_not_raise():
    chain, carbons = _carbon_chain(12)
    typifier = _StubTypifier(2)
    region = typifier.scope.region(chain, [carbons[6]])
    types = typifier.typify_region(region)
    assert len(types.atoms) == len(region.interior)


# --------------------------------------------------------------------------
# ac-005 — RegionTypes carries impropers
# --------------------------------------------------------------------------


def test_region_types_has_an_impropers_field():
    chain, carbons = _carbon_chain(8)
    typifier = _StubTypifier(2)
    region = typifier.scope.region(chain, [carbons[4]])
    types = typifier.typify_region(region)
    assert hasattr(types, "impropers")
    assert isinstance(types.impropers, tuple)


# --------------------------------------------------------------------------
# ac-006 — touched is validated, not assumed
# --------------------------------------------------------------------------


def test_empty_touched_raises():
    chain, _ = _carbon_chain(5)
    with pytest.raises(ValueError, match="touched is empty"):
        TypeScope(reach=2).region(chain, [])


def test_dead_handle_in_touched_raises():
    chain, carbons = _carbon_chain(5)
    dead = max(a.handle for a in chain.atoms) + 1000
    with pytest.raises(ValueError, match=f"touched handle {dead}"):
        TypeScope(reach=2).region(chain, [dead])


def test_interior_reach_may_not_exceed_extraction():
    chain, carbons = _carbon_chain(5)
    with pytest.raises(ValueError, match="exceeds extract_radius"):
        AffectedRegion._from(chain, [carbons[2]], extract_radius=1, interior_reach=3)


def test_assembler_asserts_molrs_reports_the_forming_bond_endpoints():
    """molrs must report both new-bond endpoints as touched; molpy checks it."""
    from molpy.builder.assembly import ExhaustiveSelector, GraphAssembler

    cloud = mp.Atomistic()
    for i in range(3):
        cloud.def_atom(element="N", x=float(i), y=0.0, z=0.0)
        cloud.def_atom(element="O", x=float(i), y=1.0, z=0.0)

    assembler = GraphAssembler(mp.Reaction("[N:1].[O:2]>>[N:1][O:2]"))
    # the real molrs return value satisfies the contract
    out = assembler.assemble(cloud, ExhaustiveSelector(cutoff=2.0))
    assert isinstance(out, mp.Atomistic)

    # a molrs that dropped an endpoint would be caught, not silently absorbed
    with pytest.raises(RuntimeError, match="omitted forming-bond endpoint"):
        assembler._assert_touched_covers_forming_bond({1: 10, 2: 20}, [10])


# --------------------------------------------------------------------------
# ac-002 / ac-003 — region typing == whole-graph typing, and reach is minimal
#
# The oracle is whole-graph typing: an atom's type *is* what the typifier assigns
# it in the full structure. Region typing is an optimisation that claims to agree.
# Boundary atoms are capped to a valid molecule (as AmberToolsTypifier does),
# otherwise SMARTS matches the wrong rule on a truncated valence.
# --------------------------------------------------------------------------

OPLS_SYSTEMS = {
    "PEO COCCOC": "COCCOC",
    "p-xylene": "Cc1ccc(C)cc1",
    "methyl acrylate": "C=CC(=O)OC",
}


def _opls_molecule(smiles: str) -> Atomistic:
    ir = smilesir_to_atomistic(parse_smiles(smiles))
    graph = Atomistic.adopt(molrs.add_hydrogens(ir))
    for i, atom in enumerate(graph.atoms):
        atom["x"] = 1.5 * i
        atom["y"] = 0.7 * math.sin(i)
        atom["z"] = 0.7 * math.cos(i)
    molrs.perceive_aromaticity(graph)
    graph.generate_topology(gen_angle=True, gen_dihedral=True)
    return graph


def _opls_types(typifier, graph: Atomistic) -> list[str]:
    return [str(t) for t in typifier.typify(graph).to_frame()["atoms"]["type"]]


def _mistyped_interior(typifier, graph: Atomistic, reach: int) -> int:
    """Interior atoms whose region type disagrees with the whole-graph type."""
    scope = TypeScope(reach=reach)
    whole = _opls_types(typifier, graph)
    atoms = list(graph.atoms)
    index_of = {atom.handle: i for i, atom in enumerate(atoms)}

    wrong = 0
    for seed in atoms:
        hops = dict(graph.topo_distances(seed.handle, max_hops=scope.extract_radius))
        ball = sorted(hops)
        sub, _ = graph.extract_subgraph([seed], scope.extract_radius)
        got = _opls_types(typifier, sub.complete_valence())
        for pos, handle in enumerate(ball):
            if hops[handle] > scope.interior_reach:
                continue
            if got[pos] != whole[index_of[handle]]:
                wrong += 1
    return wrong


@pytest.mark.parametrize("name", list(OPLS_SYSTEMS))
def test_region_typing_reproduces_whole_graph_typing_at_the_declared_reach(name):
    typifier = molrs.typifier.OPLSAATypifier()
    graph = _opls_molecule(OPLS_SYSTEMS[name])
    assert _mistyped_interior(typifier, graph, reach=2) == 0


def test_declared_reach_is_the_smallest_that_works():
    """reach=2 is measured, not chosen: reach=1 mistypes PEO."""
    typifier = molrs.typifier.OPLSAATypifier()
    peo = _opls_molecule(OPLS_SYSTEMS["PEO COCCOC"])

    assert _mistyped_interior(typifier, peo, reach=1) > 0, (
        "reach=1 must be insufficient, otherwise the declared reach is too large"
    )
    assert _mistyped_interior(typifier, peo, reach=2) == 0


def test_force_field_typifier_declares_the_measured_reach():
    from molpy.typifier.atomistic import ForceFieldTypifier

    scope = ForceFieldTypifier.scope.fget(object.__new__(ForceFieldTypifier))
    assert scope == TypeScope(reach=2)
    assert scope.interior_reach == 2
    assert scope.extract_radius == 4
