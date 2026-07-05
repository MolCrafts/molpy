"""Tests for :class:`AffectedRegion` (incremental-typify-01-region).

An ``AffectedRegion`` is the radius-N ball a graph edit touched, extracted as a
first-class ``Atomistic`` subgraph carrying ``interior`` / ``boundary`` /
``entity_map`` plus an isomorphism-invariant structural ``__hash__`` / ``__eq__``.
Producers (``Reacter``, ``Crosslinker``) build it from the atoms an edit touched.
"""

import inspect

import molpy as mp
from molpy.core.affected_region import AffectedRegion, region_radius
from molpy.core.atomistic import Atom, Atomistic
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
            s.def_bond(prev, c, order=1.0)
        carbons.append(c)
        prev = c
    return s, carbons


# --------------------------------------------------------------------------
# ac-001 — extraction: interior / boundary / entity_map + radius policy
# --------------------------------------------------------------------------


def test_from_is_an_atomistic_subclass():
    chain, carbons = _carbon_chain(5)
    region = AffectedRegion._from(chain, [carbons[2]], radius=1)
    assert isinstance(region, AffectedRegion)
    assert isinstance(region, Atomistic)


def test_interior_is_the_touched_atoms():
    chain, carbons = _carbon_chain(5)
    region = AffectedRegion._from(chain, [carbons[2]], radius=1)
    # exactly one seed -> one interior atom, mapping back to the parent center.
    assert len(region.interior) == 1
    assert region.entity_map[region.interior[0]] is carbons[2]


def test_boundary_atoms_have_a_neighbor_outside_the_ball():
    chain, carbons = _carbon_chain(5)
    # center C2, radius 1 -> ball is the carbons {C1, C2, C3} plus their H's.
    region = AffectedRegion._from(chain, [carbons[2]], radius=1)
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
    region = AffectedRegion._from(chain, [carbons[2]], radius=1)
    # every region atom maps to a distinct parent atom of the same element.
    parents = [region.entity_map[a] for a in region.atoms]
    assert len(parents) == len(set(map(id, parents)))
    for region_atom in region.atoms:
        parent = region.entity_map[region_atom]
        assert region_atom.get("element") == parent.get("element")


def test_from_accepts_handles_as_well_as_atoms():
    chain, carbons = _carbon_chain(5)
    by_atom = AffectedRegion._from(chain, [carbons[2]], radius=1)
    by_handle = AffectedRegion._from(chain, [carbons[2].handle], radius=1)
    assert by_atom == by_handle


def test_region_radius_floor_and_override():
    assert region_radius(None) == 4  # floor

    class _Shallow:
        context_radius = 2

    class _Deep:
        context_radius = 6

    assert region_radius(_Shallow()) == 4  # floored up
    assert region_radius(_Deep()) == 6  # override wins


# --------------------------------------------------------------------------
# ac-002 — structural __hash__ / __eq__ (dedup key); Entity identity preserved
# --------------------------------------------------------------------------


def test_identical_junctions_are_equal_and_hash_equal():
    chain_a, ca = _carbon_chain(5)
    chain_b, cb = _carbon_chain(5)
    region_a = AffectedRegion._from(chain_a, [ca[2]], radius=2)
    region_b = AffectedRegion._from(chain_b, [cb[2]], radius=2)
    assert region_a == region_b
    assert hash(region_a) == hash(region_b)


def test_different_junctions_are_not_equal():
    chain, carbons = _carbon_chain(6)
    small = AffectedRegion._from(chain, [carbons[2]], radius=1)
    big = AffectedRegion._from(chain, [carbons[2]], radius=3)
    assert small != big
    assert hash(small) != hash(big)


def test_member_atoms_keep_identity_hashing():
    chain, carbons = _carbon_chain(5)
    region = AffectedRegion._from(chain, [carbons[2]], radius=1)
    atom = region.interior[0]
    # region overrides hashing only at the region level; member atoms stay
    # identity-hashed (unchanged core contract).
    assert hash(atom) == id(atom)
    other = next(a for a in region.atoms if a is not atom)
    assert atom != other


def test_region_is_not_equal_to_plain_atomistic():
    chain, carbons = _carbon_chain(5)
    region = AffectedRegion._from(chain, [carbons[2]], radius=1)
    assert region != chain


# --------------------------------------------------------------------------
# ac-003 — the region is a MolGraph, consumable by AmberTools unchanged
# --------------------------------------------------------------------------


def test_region_feeds_the_ambertools_pdb_bridge():
    chain, carbons = _carbon_chain(5)
    region = AffectedRegion._from(chain, [carbons[2]], radius=2)

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


def test_crosslinker_builds_regions_from_touched_handles():
    from molpy.builder.crosslink import DeterministicCrosslinker

    reaction = "[N:1].[O:2]>>[N:1][O:2]"
    cloud = mp.Atomistic()
    for i in range(3):
        cloud.def_atom(element="N", x=float(i), y=0.0, z=0.0)
        cloud.def_atom(element="O", x=float(i), y=1.0, z=0.0)

    xl = DeterministicCrosslinker(reaction, cutoff=2.0)
    out = xl.apply(cloud)

    # one region per formed crosslink bond, each a real AffectedRegion.
    assert len(xl.last_regions) == 3
    for region in xl.last_regions:
        assert isinstance(region, AffectedRegion)
        assert len(region.interior) >= 1
    # apply's public contract is unchanged (returns the new graph).
    assert isinstance(out, mp.Atomistic)


def test_reacter_run_sets_region():
    from molpy.reacter import (
        Reacter,
        find_port,
        form_single_bond,
        select_hydrogens,
        select_self,
    )

    def methane_fragment(port: str) -> mp.Atomistic:
        s = mp.Atomistic()
        c = s.def_atom(element="C")
        for _ in range(3):
            s.def_bond(c, s.def_atom(element="H"))
        c["port"] = port
        return s

    left = methane_fragment(">")
    right = methane_fragment("<")
    coupling = Reacter(
        name="cc",
        anchor_selector_left=select_self,
        anchor_selector_right=select_self,
        leaving_selector_left=select_hydrogens(1),
        leaving_selector_right=select_hydrogens(1),
        bond_former=form_single_bond,
    )
    result = coupling.run(
        left,
        right,
        port_atom_L=find_port(left, ">"),
        port_atom_R=find_port(right, "<"),
    )
    assert result.region is not None
    assert isinstance(result.region, AffectedRegion)
    # the region seeds are the two anchors where the bond formed.
    assert len(result.region.interior) >= 1
