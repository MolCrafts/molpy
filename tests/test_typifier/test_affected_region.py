"""Tests for :class:`AffectedRegion` (incremental-typify-01-region).

An ``AffectedRegion`` is the radius-N ball a graph edit touched, extracted as a
first-class ``Atomistic`` subgraph carrying ``interior`` / ``boundary`` /
``entity_map`` plus an isomorphism-invariant structural ``__hash__`` / ``__eq__``.
Producers (``Reacter``, ``Crosslinker``) build it from the atoms an edit touched.
"""

import inspect

import molrs
import pytest

import molpy as mp
from molpy.typifier.affected_region import AffectedRegion
from molpy.core.atomistic import Atom, Atomistic
from molpy.typifier.base import Match, Typifier
from molpy.wrapper.antechamber import write_antechamber_input_pdb


def _carbon_chain(m: int) -> tuple[Atomistic, list[Atom]]:
    """Linear chain of ``m`` carbons C0-C1-...-C(m-1) with one hydrogen each."""
    s = mp.Atomistic()
    carbons: list[Atom] = []
    prev: Atom | None = None
    for i in range(m):
        c = s.def_atom(element="C", x=float(i), y=0.0, z=0.0)
        s.def_bond(c, s.def_atom(element="H", x=float(i), y=1.0, z=0.0))
        if prev is not None:
            s.def_bond(prev, c, bond_type=1, bond_number=1)
        carbons.append(c)
        prev = c
    return s, carbons


# --------------------------------------------------------------------------
# ac-001 — extraction: interior / boundary / entity_map + radius policy
# --------------------------------------------------------------------------


def test_from_is_an_atomistic_subclass():
    chain, carbons = _carbon_chain(5)
    region = AffectedRegion._from(
        chain, [carbons[2]], extract_radius=1, interior_reach=0
    )
    assert isinstance(region, AffectedRegion)
    assert isinstance(region, Atomistic)


def test_interior_is_the_touched_atoms():
    chain, carbons = _carbon_chain(5)
    region = AffectedRegion._from(
        chain, [carbons[2]], extract_radius=1, interior_reach=0
    )
    # exactly one seed -> one interior atom, mapping back to the parent center.
    assert len(region.interior) == 1
    assert region.entity_map[region.interior[0]] is carbons[2]


def test_boundary_atoms_have_a_neighbor_outside_the_ball():
    chain, carbons = _carbon_chain(5)
    # center C2, radius 1 -> ball is the carbons {C1, C2, C3} plus their H's.
    region = AffectedRegion._from(
        chain, [carbons[2]], extract_radius=1, interior_reach=0
    )
    # C1 (neighbor C0 outside) and C3 (neighbor C4 outside) are the carbon
    # boundary; every boundary atom must map to a parent with an outside neighbor.
    boundary_parents = {region.entity_map[b] for b in region.boundary}
    assert carbons[1] in boundary_parents
    assert carbons[3] in boundary_parents
    region_parents = set(region.entity_map.values())
    for b in region.boundary:
        parent = region.entity_map[b]
        neighbors = chain.get_neighbors(parent)
        assert any(nb not in region_parents for nb in neighbors)


def test_entity_map_round_trips_region_to_parent():
    chain, carbons = _carbon_chain(5)
    region = AffectedRegion._from(
        chain, [carbons[2]], extract_radius=1, interior_reach=0
    )
    # every region atom maps to a distinct parent atom of the same element.
    parents = [region.entity_map[a] for a in region.atoms]
    assert len(parents) == len(set(map(id, parents)))
    for region_atom in region.atoms:
        parent = region.entity_map[region_atom]
        assert region_atom.get("element") == parent.get("element")


def test_from_accepts_handles_as_well_as_atoms():
    chain, carbons = _carbon_chain(5)
    by_atom = AffectedRegion._from(
        chain, [carbons[2]], extract_radius=1, interior_reach=0
    )
    by_handle = AffectedRegion._from(
        chain, [carbons[2].handle], extract_radius=1, interior_reach=0
    )
    assert by_atom == by_handle


def test_extraction_radius_is_derived_from_reach_not_guessed():
    """No floor, no guessed radius: ``around`` derives both radii from ``reach``.

    ``region_radius()`` / ``_FLOOR`` / ``TypeScope`` are gone — see
    ``tests/test_typifier/test_region_radii.py`` for the arithmetic and the
    measured reach.
    """
    chain, carbons = _carbon_chain(12)
    region = AffectedRegion.around(chain, [carbons[6]], reach=2)

    assert region.extract_radius == 4
    assert region.interior_reach == 2
    # interior is the write-back ball, not "everything that is not boundary"
    inside = {h for h, _ in chain.topo_distances(carbons[6].handle, max_hops=2)}
    assert {region.entity_map[a].handle for a in region.interior} == inside


# --------------------------------------------------------------------------
# A ring is not divisible: a radius may not take half of one
# --------------------------------------------------------------------------


def _methylbenzene() -> tuple[Atomistic, list[Atom], Atom]:
    """Six-carbon ring with a methyl carbon on ring atom 0. Returns
    ``(graph, ring_atoms, methyl)``; ring index == hops from the methyl minus 1.
    """
    s = mp.Atomistic()
    ring = [s.def_atom(element="C", x=float(i), y=0.0, z=0.0) for i in range(6)]
    for i in range(6):
        s.def_bond(ring[i], ring[(i + 1) % 6], bond_type=4, bond_number=0)
    methyl = s.def_atom(element="C", x=0.0, y=1.5, z=0.0)
    s.def_bond(ring[0], methyl, bond_type=1, bond_number=1)
    return s, ring, methyl


def _rings_of(graph: Atomistic) -> list[set[int]]:
    return [set(ring) for ring in molrs.perceive.RingInfo(graph).rings()]


def test_region_never_splits_a_ring():
    # reach=1 gives extract_radius 3, which from the methyl carbon reaches the
    # meta atoms and stops one short of para — half a ring, without closure.
    graph, ring, methyl = _methylbenzene()
    region = AffectedRegion.around(graph, [methyl], reach=1)
    assert region.extract_radius == 3

    selected = {region.entity_map[a].handle for a in region.atoms}
    for whole_ring in _rings_of(graph):
        inside = whole_ring & selected
        assert inside in (set(), whole_ring), (
            f"ring split: {len(inside)} of {len(whole_ring)} atoms"
        )
    assert {a.handle for a in ring} <= selected
    assert methyl.handle in selected


def test_every_seed_keeps_every_ring_whole():
    graph, _, _ = _methylbenzene()
    rings = _rings_of(graph)
    for seed in list(graph.atoms):
        region = AffectedRegion.around(graph, [seed], reach=1)
        selected = {region.entity_map[a].handle for a in region.atoms}
        for whole_ring in rings:
            inside = whole_ring & selected
            assert inside in (set(), whole_ring), f"ring split at seed {seed.handle}"


def test_ring_closure_atoms_are_context_never_interior():
    """Closing the ring widens the view, not the write-back set.

    The atoms a ring drags in sit past ``extract_radius`` by construction, so
    ``interior`` stays exactly ``ball(touched, interior_reach)``.
    """
    graph, ring, methyl = _methylbenzene()
    region = AffectedRegion.around(graph, [methyl], reach=1)

    inside = {h for h, _ in graph.topo_distances(methyl.handle, max_hops=2)}
    assert {region.entity_map[a].handle for a in region.interior} == inside
    # para is four hops from the methyl: it is in the region, out of the interior.
    para = ring[3].handle
    assert para in {region.entity_map[a].handle for a in region.atoms}
    assert para not in {region.entity_map[a].handle for a in region.interior}


def test_a_fused_ring_system_comes_in_whole():
    # Naphthalene: ring A 0-1-2-3-4-5, ring B 3-4-6-7-8-9 share the 3-4 bond.
    s = mp.Atomistic()
    ids = [s.def_atom(element="C", x=float(i), y=0.0, z=0.0) for i in range(10)]
    for i in range(5):
        s.def_bond(ids[i], ids[i + 1], bond_type=4, bond_number=0)
    s.def_bond(ids[5], ids[0], bond_type=4, bond_number=0)
    for a, b in ((ids[3], ids[6]), (ids[6], ids[7]), (ids[7], ids[8])):
        s.def_bond(a, b, bond_type=4, bond_number=0)
    s.def_bond(ids[8], ids[9], bond_type=4, bond_number=0)
    s.def_bond(ids[9], ids[4], bond_type=4, bond_number=0)

    # A radius-1 ball around one atom of ring A cannot see ring B at all.
    region = AffectedRegion._from(s, [ids[0]], extract_radius=1, interior_reach=1)
    selected = {region.entity_map[a].handle for a in region.atoms}
    assert selected == {a.handle for a in ids}, (
        "a fused system is one unit: touching ring A brings ring B"
    )


def _macrocycle(m: int) -> tuple[Atomistic, list[Atom]]:
    """One `m`-membered carbocycle — a ring topologically, a chain locally."""
    s = mp.Atomistic()
    ring = [s.def_atom(element="C", x=float(i), y=0.0, z=0.0) for i in range(m)]
    for i in range(m):
        s.def_bond(ring[i], ring[(i + 1) % m], bond_type=1, bond_number=1)
    return s, ring


@pytest.mark.parametrize("size", [50, 500, 5000])
def test_a_macrocycle_does_not_drag_its_whole_loop_into_the_ball(size):
    """The bound is what keeps a region a region.

    Closure used to run on *any* ring, so a ball of nine atoms on a 5000-ring
    extracted all 5000 — and one crosslink joining two chains is all it takes to
    create that ring. The extracted ball must stay the size the radius asked
    for, and must not grow with the loop.
    """
    graph, ring = _macrocycle(size)
    region = AffectedRegion.around(graph, [ring[0]], reach=2)

    assert region.extract_radius == 4
    # A ball of radius 4 on a plain cycle: the seed plus four atoms each way.
    assert len(list(region.atoms)) == 9, (
        f"a {size}-membered ring pulled {len(list(region.atoms))} atoms into a "
        "ball that asked for 9"
    )
    assert len(region.interior) == 5


def test_the_bound_is_the_only_thing_separating_the_two_cases():
    """Same graph, same ball — only ``MAX_RING_SIZE`` decides.

    Guards against the ball staying small for some unrelated reason: raise the
    bound past the ring and the whole loop comes back.
    """
    graph, ring = _macrocycle(12)
    assert AffectedRegion.MAX_RING_SIZE == 8

    bounded, _ = graph.extract_subgraph(
        [ring[0]], 4, max_ring_size=AffectedRegion.MAX_RING_SIZE
    )
    assert len(list(bounded.atoms)) == 9

    generous, _ = graph.extract_subgraph([ring[0]], 4, max_ring_size=12)
    assert len(list(generous.atoms)) == 12


def test_a_small_ring_is_still_indivisible_under_the_bound():
    """The bound must not cost the behaviour it was added to preserve."""
    graph, ring, methyl = _methylbenzene()
    region = AffectedRegion.around(graph, [methyl], reach=1)
    selected = {region.entity_map[a].handle for a in region.atoms}
    assert {a.handle for a in ring} <= selected


# --------------------------------------------------------------------------
# ac-002 — structural __hash__ / __eq__ (dedup key); Entity identity preserved
# --------------------------------------------------------------------------


def test_identical_junctions_are_equal_and_hash_equal():
    chain_a, ca = _carbon_chain(5)
    chain_b, cb = _carbon_chain(5)
    region_a = AffectedRegion._from(
        chain_a, [ca[2]], extract_radius=2, interior_reach=0
    )
    region_b = AffectedRegion._from(
        chain_b, [cb[2]], extract_radius=2, interior_reach=0
    )
    assert region_a == region_b
    assert hash(region_a) == hash(region_b)


def test_different_junctions_are_not_equal():
    chain, carbons = _carbon_chain(6)
    small = AffectedRegion._from(
        chain, [carbons[2]], extract_radius=1, interior_reach=0
    )
    big = AffectedRegion._from(chain, [carbons[2]], extract_radius=3, interior_reach=0)
    assert small != big
    assert hash(small) != hash(big)


def test_same_graph_with_a_different_write_back_set_is_a_different_region():
    """Equality is the cache key, and the cache replays a *write-back set*.

    Two edits on one small molecule extract the same graph — the whole
    molecule — while disturbing opposite ends of it. A snapshot records only
    its own interior, keyed by canonical position, so replaying one onto the
    other would retype the first edit's atoms a second time and leave the
    second edit's atoms untouched. That is a silent wrong answer, not a slow
    one: nothing downstream can tell an unwritten type from an unchanged one.
    """
    chain, carbons = _carbon_chain(5)
    # An asymmetric terminus, so the two write-back sets cannot be related by an
    # automorphism of the chain — a symmetric molecule's two ends genuinely are
    # interchangeable, and a snapshot taken at one does apply at the other.
    chain.def_bond(carbons[0], chain.def_atom(element="O", x=0.0, y=-1.0, z=0.0))

    head = AffectedRegion._from(chain, [carbons[0]], extract_radius=9, interior_reach=1)
    tail = AffectedRegion._from(chain, [carbons[4]], extract_radius=9, interior_reach=1)

    def covered(region):
        return {atom.handle for atom in region.entity_map.values()}

    assert covered(head) == covered(tail)  # same extracted graph
    assert head != tail
    assert hash(head) != hash(tail)


def test_member_atoms_keep_identity_hashing():
    chain, carbons = _carbon_chain(5)
    region = AffectedRegion._from(
        chain, [carbons[2]], extract_radius=1, interior_reach=0
    )
    atom = region.interior[0]
    # region overrides hashing only at the region level; member atoms stay
    # identity-hashed (unchanged core contract).
    assert hash(atom) == id(atom)
    other = next(a for a in region.atoms if a is not atom)
    assert atom != other


def test_region_is_not_equal_to_plain_atomistic():
    chain, carbons = _carbon_chain(5)
    region = AffectedRegion._from(
        chain, [carbons[2]], extract_radius=1, interior_reach=0
    )
    assert region != chain


# --------------------------------------------------------------------------
# ac-003 — the region is a MolGraph, consumable by AmberTools unchanged
# --------------------------------------------------------------------------


def test_region_feeds_the_ambertools_pdb_bridge():
    chain, carbons = _carbon_chain(5)
    region = AffectedRegion._from(
        chain, [carbons[2]], extract_radius=2, interior_reach=0
    )

    # The antechamber input bridge is typed ``(path, atomistic: Atomistic)``;
    # the region satisfies that declared input type unchanged.
    assert isinstance(region, Atomistic)
    params = list(inspect.signature(write_antechamber_input_pdb).parameters.values())
    # (annotation is a string under ``from __future__ import annotations``)
    assert params[1].annotation in (Atomistic, "Atomistic")

    # ...and exposes the exact per-atom surface the PDB writer reads (element +
    # x/y/z on every atom), so the bridge consumes it like any Atomistic. (The
    # full write is not run here: it needs no antechamber, and the writer's
    # object-dtype string columns hit an unrelated molrs Block limitation that
    # affects every Atomistic equally, not the region.)
    atoms = list(region.atoms)
    assert atoms
    for a in atoms:
        assert a.get("element") is not None
        assert all(a.get(k) is not None for k in ("x", "y", "z"))


# --------------------------------------------------------------------------
# ac-004 — producers build the region
# --------------------------------------------------------------------------


class _ElementTypifier(Typifier[Atomistic]):
    """Types every atom by element; implements ``match`` and nothing else."""

    def match(self, graph: Atomistic) -> Match:
        return Match(nodes=tuple({"type": f"t_{a['element']}"} for a in graph.atoms))


def _nitrogen_oxygen_cloud(n: int = 3) -> Atomistic:
    cloud = mp.Atomistic()
    for i in range(n):
        cloud.def_atom(element="N", x=float(i), y=0.0, z=0.0)
        cloud.def_atom(element="O", x=float(i), y=1.0, z=0.0)
    return cloud


def test_assembler_types_every_region_it_builds():
    """One region per formed bond; each one's interior is written back."""
    from molpy.builder.assembly import ExhaustiveSelector, GraphAssembler

    out = GraphAssembler(
        mp.Reaction("[N:1].[O:2]>>[N:1][O:2]"), typifier=_ElementTypifier(), reach=2
    ).assemble(_nitrogen_oxygen_cloud(), ExhaustiveSelector(cutoff=2.0))

    assert isinstance(out, mp.Atomistic)
    assert len(list(out.bonds)) == 3
    # every atom the edits touched carries a type written back from its region
    assert all(atom.get("type") for atom in out.atoms)


def test_assembler_without_a_typifier_builds_no_region():
    """No typifier, no region, and no radius may be guessed.

    Pure-topology assembly is a named mode, not a region path with a fallback
    radius — ``_FLOOR = 4`` is gone.
    """
    from molpy.builder.assembly import ExhaustiveSelector, GraphAssembler

    out = GraphAssembler(mp.Reaction("[N:1].[O:2]>>[N:1][O:2]")).assemble(
        _nitrogen_oxygen_cloud(), ExhaustiveSelector(cutoff=2.0)
    )

    assert isinstance(out, mp.Atomistic)
    assert len(list(out.bonds)) == 3  # the edits still happened
    assert not any(atom.get("type") for atom in out.atoms)
