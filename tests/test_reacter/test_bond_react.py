#!/usr/bin/env python3
"""Unit tests for BondReactReacter class.

Tests cover:
- BondReactReacter initialization (with Reacter constructor parameters)
- run() template generation (BondReactResult.template)
- react_id assignment
- Pre/post template consistency
- BondReactTemplate structure
- Error handling
"""

import logging
from pathlib import Path

import pytest

from molpy.core.atomistic import Atom, Atomistic, Bond, Improper
from molpy.io import write_bond_react_map
from molpy.reacter import (
    form_single_bond,
    select_port,
    select_none,
    select_one_hydrogen,
)
from molpy.reacter.selectors import find_port_atom
from molpy.reacter.bond_react import (
    BondReactReacter,
    BondReactResult,
    BondReactTemplate,
)


# ── shared fixture builders / helpers ────────────────────────────────
#
# sp2-like reaction fixture: a left chain whose middle carbon (c1) has
# exactly 3 bonded neighbors and carries a pre-existing Improper, plus a
# right ethyl-like fragment.  C-C coupling between the port carbons each
# eliminates one hydrogen, so the anchors end with exactly 3 neighbors.


def _build_sp2_left() -> Atomistic:
    """Left reactant: h01-c0-c1(-h1)-c2(>)(h21)(h22), improper on c1."""
    struct = Atomistic()
    c0 = Atom(element="C", type="CT", charge=-0.18)
    h01 = Atom(element="H", type="HC", charge=0.06)
    c1 = Atom(element="C", type="CM", charge=-0.12)
    h1 = Atom(element="H", type="HC", charge=0.06)
    c2 = Atom(element="C", type="CT", charge=-0.18)
    h21 = Atom(element="H", type="HC", charge=0.06)
    h22 = Atom(element="H", type="HC", charge=0.06)
    struct.add_entity(c0, h01, c1, h1, c2, h21, h22)
    struct.add_link(
        Bond(c0, h01),
        Bond(c0, c1),
        Bond(c1, h1),
        Bond(c1, c2),
        Bond(c2, h21),
        Bond(c2, h22),
    )
    # c1 has exactly 3 bonded neighbors (c0, h1, c2): sp2-like center
    struct.add_link(Improper(c1, c0, c2, h1))
    c2["port"] = ">"
    return struct


def _build_sp2_right() -> Atomistic:
    """Right reactant: c3(<)(h31)(h32)-c4-h41."""
    struct = Atomistic()
    c3 = Atom(element="C", type="CT", charge=-0.18)
    h31 = Atom(element="H", type="HC", charge=0.06)
    h32 = Atom(element="H", type="HC", charge=0.06)
    c4 = Atom(element="C", type="CT", charge=-0.18)
    h41 = Atom(element="H", type="HC", charge=0.06)
    struct.add_entity(c3, h31, h32, c4, h41)
    struct.add_link(
        Bond(c3, h31),
        Bond(c3, h32),
        Bond(c3, c4),
        Bond(c4, h41),
    )
    c3["port"] = "<"
    return struct


def _make_sp2_reacter(radius: int) -> BondReactReacter:
    return BondReactReacter(
        name="sp2_growth",
        anchor_selector_left=select_port,
        anchor_selector_right=select_port,
        leaving_selector_left=select_one_hydrogen,
        leaving_selector_right=select_one_hydrogen,
        bond_former=form_single_bond,
        radius=radius,
    )


def _run_sp2_reaction(radius: int = 4) -> BondReactResult:
    """Run the sp2 fixture reaction on freshly built inputs."""
    left = _build_sp2_left()
    right = _build_sp2_right()
    reacter = _make_sp2_reacter(radius)
    port_l = find_port_atom(left, ">")
    port_r = find_port_atom(right, "<")
    return reacter.run(left, right, port_atom_L=port_l, port_atom_R=port_r)


def _make_manual_template_with_one_initiator() -> BondReactTemplate:
    """Hand-built template with only 1 initiator atom (invalid)."""
    pre = Atomistic()
    a_pre = pre.def_atom(element="C", type="CT", charge=0.0, react_id=1)
    b_pre = pre.def_atom(element="C", type="CT", charge=0.0, react_id=2)
    pre.def_bond(a_pre, b_pre, type="CT-CT")

    post = Atomistic()
    a_post = post.def_atom(element="C", type="CT", charge=0.0, react_id=1)
    b_post = post.def_atom(element="C", type="CT", charge=0.0, react_id=2)
    post.def_bond(a_post, b_post, type="CT-CT")

    return BondReactTemplate(
        pre=pre,
        post=post,
        initiator_atoms=[a_pre],
        edge_atoms=[],
        deleted_atoms=[],
        pre_react_id_to_atom={1: a_pre, 2: b_pre},
        post_react_id_to_atom={1: a_post, 2: b_post},
    )


def _improper_key(improper: Improper) -> tuple[int, frozenset[int]]:
    """Equivalence key: (center react_id, unordered neighbor react_ids)."""
    rids = [int(ep["react_id"]) for ep in improper.endpoints]
    return rids[0], frozenset(rids[1:])


def _parse_map_sections(map_path: Path) -> dict[str, list[list[int]]]:
    """Parse a fix bond/react ``.map`` file into {section: rows of ints}."""
    section_names = {"InitiatorIDs", "EdgeIDs", "DeleteIDs", "Equivalences"}
    sections: dict[str, list[list[int]]] = {}
    current: str | None = None
    for raw_line in map_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line in section_names:
            current = line
            sections[current] = []
            continue
        if current is not None:
            sections[current].append([int(token) for token in line.split()])
    return sections


class TestBondReactReacter:
    """Test BondReactReacter class."""

    def test_bond_react_reacter_initialization(self):
        """Test BondReactReacter initialization with Reacter constructor parameters."""
        bond_react_reacter = BondReactReacter(
            name="test_reaction",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
            radius=4,
        )

        # BondReactReacter IS a Reacter
        assert bond_react_reacter.name == "test_reaction"
        assert bond_react_reacter.radius == 4
        assert bond_react_reacter._react_id_counter == 0

    def test_bond_react_reacter_default_radius(self):
        """Test BondReactReacter with default radius."""
        bond_react_reacter = BondReactReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        assert bond_react_reacter.radius == 4  # default

    def test_run_with_template_basic(self):
        """Test basic run_with_template() execution."""
        # Create left structure: C-H
        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L = Atom(element="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        # Create right structure: C-H
        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        # Create BondReactReacter
        bond_react_reacter = BondReactReacter(
            name="C-C_coupling",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
            radius=2,
        )

        # Run reaction with template
        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = bond_react_reacter.run(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )
        template = result.template

        # Validate ReactionResult
        assert result is not None
        assert isinstance(result.product, Atomistic)
        assert len(list(result.product.atoms)) == 2  # 2 C atoms

        # Validate BondReactTemplate
        assert isinstance(template, BondReactTemplate)
        assert template.pre is not None
        assert template.post is not None
        assert len(template.initiator_atoms) == 2  # Two port atoms
        assert len(template.deleted_atoms) == 2  # Two H atoms removed

        # Check pre and post have same atoms (by react_id)
        pre_rids = {a["react_id"] for a in template.pre.atoms}
        post_rids = {a["react_id"] for a in template.post.atoms}
        assert pre_rids == post_rids

        # Check react_ids are assigned
        assert all("react_id" in a.data for a in template.pre.atoms)
        assert all("react_id" in a.data for a in template.post.atoms)

    def test_run_with_template_react_id_assignment(self):
        """Test that react_ids are assigned correctly."""
        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L = Atom(element="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        bond_react_reacter = BondReactReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")

        # Check react_ids are assigned before reaction
        bond_react_reacter._assign_react_ids(struct_L)
        bond_react_reacter._assign_react_ids(struct_R)

        assert "react_id" in c_L.data
        assert "react_id" in h_L.data
        assert "react_id" in c_R.data
        assert "react_id" in h_R.data

        # Check react_ids are unique
        all_rids = [c_L["react_id"], h_L["react_id"], c_R["react_id"], h_R["react_id"]]
        assert len(all_rids) == len(set(all_rids))

    def test_run_with_template_no_leaving_groups(self):
        """Test run_with_template() with no leaving groups."""
        struct_L = Atomistic()
        c_L = Atom(element="C")
        struct_L.add_entity(c_L)
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        struct_R.add_entity(c_R)
        c_R["port"] = "2"

        bond_react_reacter = BondReactReacter(
            name="addition",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = bond_react_reacter.run(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )
        template = result.template

        # No atoms removed
        assert len(template.deleted_atoms) == 0
        assert len(list(result.product.atoms)) == 2

        # Pre and post should have same atoms
        pre_rids = {a["react_id"] for a in template.pre.atoms}
        post_rids = {a["react_id"] for a in template.post.atoms}
        assert pre_rids == post_rids

    def test_run_with_template_compute_topology_false(self):
        """Test run_with_template() with compute_topology=False."""
        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L = Atom(element="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        bond_react_reacter = BondReactReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = bond_react_reacter.run(
            struct_L,
            struct_R,
            port_atom_L=port_atom_L,
            port_atom_R=port_atom_R,
            compute_topology=False,
        )
        template = result.template

        # Should still work
        assert result is not None
        assert template is not None
        assert len(list(result.product.atoms)) == 2

    def test_run_with_template_init_atoms(self):
        """Test that init_atoms (port atoms) are correctly identified."""
        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L = Atom(element="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        bond_react_reacter = BondReactReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = bond_react_reacter.run(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )
        template = result.template

        # Check init_atoms. Caller-owned atoms are never stamped with
        # react_id (mutation hygiene); the resolved internal copies are
        # exposed on the result.
        assert len(template.initiator_atoms) == 2
        init_rids = {a["react_id"] for a in template.initiator_atoms}
        assert result.port_atom_L["react_id"] in init_rids
        assert result.port_atom_R["react_id"] in init_rids
        assert "react_id" not in port_atom_L.data
        assert "react_id" not in port_atom_R.data

    def test_run_with_template_removed_atoms_tracked(self):
        """Test that removed atoms are correctly tracked in template."""
        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L1 = Atom(element="H")
        h_L2 = Atom(element="H")
        struct_L.add_entity(c_L, h_L1, h_L2)
        struct_L.add_link(Bond(c_L, h_L1), Bond(c_L, h_L2))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        bond_react_reacter = BondReactReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = bond_react_reacter.run(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )
        template = result.template

        # Check removed atoms
        assert len(template.deleted_atoms) == 2
        removed_rids = {a["react_id"] for a in template.deleted_atoms}

        # Removed atoms should be in pre but not in product
        pre_rids = {a["react_id"] for a in template.pre.atoms}
        product_rids = {a["react_id"] for a in result.product.atoms}

        # All removed atoms should be in pre
        assert removed_rids.issubset(pre_rids)
        # Removed atoms should not be in product
        assert removed_rids.isdisjoint(product_rids)
        # But removed atoms should be in post (for template consistency)
        post_rids = {a["react_id"] for a in template.post.atoms}
        assert removed_rids.issubset(post_rids)

    def test_run_with_template_react_id_mapping(self):
        """Test that react_id mappings are correct."""
        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L = Atom(element="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        bond_react_reacter = BondReactReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = bond_react_reacter.run(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )
        template = result.template

        # Check mappings exist
        assert template.pre_react_id_to_atom is not None
        assert template.post_react_id_to_atom is not None

        # Check all pre atoms are in mapping
        for atom in template.pre.atoms:
            rid = atom["react_id"]
            assert rid in template.pre_react_id_to_atom
            assert template.pre_react_id_to_atom[rid] is atom

        # Check all post atoms are in mapping
        for atom in template.post.atoms:
            rid = atom["react_id"]
            assert rid in template.post_react_id_to_atom
            assert template.post_react_id_to_atom[rid] is not None

        # Check same react_ids in both mappings
        assert set(template.pre_react_id_to_atom.keys()) == set(
            template.post_react_id_to_atom.keys()
        )

    def test_run_with_template_different_radius(self):
        """Test run_with_template() with different radius values."""
        # Create a larger structure to test radius effect
        struct_L = Atomistic()
        c1 = Atom(element="C")
        c2 = Atom(element="C")
        h1 = Atom(element="H")
        h2 = Atom(element="H")
        struct_L.add_entity(c1, c2, h1, h2)
        struct_L.add_link(Bond(c1, c2), Bond(c1, h1), Bond(c2, h2))
        c1["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        # Test with radius=1 (smaller subgraph)
        bond_react_reacter_1 = BondReactReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
            radius=1,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result1 = bond_react_reacter_1.run(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )
        template1 = result1.template

        # Test with radius=3 (larger subgraph)
        bond_react_reacter_3 = BondReactReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
            radius=3,
        )

        # Reset structures
        struct_L2 = struct_L.copy()
        struct_R2 = struct_R.copy()
        port_atom_L2 = find_port_atom(struct_L2, "1")
        port_atom_R2 = find_port_atom(struct_R2, "2")

        result2 = bond_react_reacter_3.run(
            struct_L2, struct_R2, port_atom_L=port_atom_L2, port_atom_R=port_atom_R2
        )
        template2 = result2.template

        # Larger radius should include more atoms (or same)
        assert len(list(template2.pre.atoms)) >= len(list(template1.pre.atoms))

    def test_assign_react_ids_preserves_existing(self):
        """Test that _assign_react_ids preserves existing react_ids."""
        struct = Atomistic()
        c1 = Atom(element="C")
        c2 = Atom(element="C")
        struct.add_entity(c1, c2)

        bond_react_reacter = BondReactReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        # Assign react_id to first atom manually
        c1["react_id"] = 100

        # Assign react_ids
        bond_react_reacter._assign_react_ids(struct)

        # First atom should keep its react_id
        assert c1["react_id"] == 100
        # Second atom should get a new react_id
        assert "react_id" in c2.data
        assert c2["react_id"] != 100

    def test_template_topology_consistency(self):
        """Test that pre and post have consistent topology."""
        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L = Atom(element="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"
        struct_L = struct_L.get_topo(gen_angle=True, gen_dihe=True)

        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"
        struct_R = struct_R.get_topo(gen_angle=True, gen_dihe=True)

        bond_react_reacter = BondReactReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = bond_react_reacter.run(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )
        template = result.template

        # Pre should have topology
        assert len(list(template.pre.bonds)) > 0

        # Post should not have bonds between deleted and non-deleted atoms
        removed_rids = {a["react_id"] for a in template.deleted_atoms}
        for bond in template.post.bonds:
            ep1_rid = bond.endpoints[0].get("react_id")
            ep2_rid = bond.endpoints[1].get("react_id")
            ep1_deleted = ep1_rid in removed_rids
            ep2_deleted = ep2_rid in removed_rids
            assert not (ep1_deleted != ep2_deleted), (
                f"Post has bond between deleted and non-deleted atom: {ep1_rid} - {ep2_rid}"
            )

    def test_template_map_export_via_io(self):
        """Map serialization goes through molpy.io.write_bond_react_map.

        Replaces the removed ``BondReactTemplate.write()`` path; the
        pre/post ``.mol`` outputs are covered by the golden-file test in
        tests/test_io/test_data/test_lammps_bond_react.py.
        """
        from pathlib import Path
        import tempfile

        from molpy.io import write_bond_react_map

        struct_L = Atomistic()
        c_L = Atom(element="C")
        h_L = Atom(element="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(element="C")
        h_R = Atom(element="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        bond_react_reacter = BondReactReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result = bond_react_reacter.run(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )
        template = result.template

        # Add dummy types for testing (normally done by typifier)
        for atom in template.pre.atoms:
            if "type" not in atom.data:
                atom["type"] = "dummy"
        for atom in template.post.atoms:
            if "type" not in atom.data:
                atom["type"] = "dummy"
        for bond in template.pre.bonds:
            if "type" not in bond.data:
                bond["type"] = "dummy"
        for bond in template.post.bonds:
            if "type" not in bond.data:
                bond["type"] = "dummy"

        # Write to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "test_rxn"
            write_bond_react_map(template, base_path)

            map_file = Path(f"{base_path}.map")
            assert map_file.exists()

            # Check map file format
            with open(map_file) as f:
                content = f.read()
                assert "InitiatorIDs" in content
                assert "EdgeIDs" in content
                assert "DeleteIDs" in content
                assert "Equivalences" in content

    def test_template_atom_ordering_consistency(self):
        """Test that pre and post templates have atoms in same order by react_id.

        This is critical for LAMMPS fix bond/react to correctly match atoms
        between pre and post templates.
        """
        struct_L = Atomistic()
        c1 = Atom(element="C")
        c2 = Atom(element="C")
        h1 = Atom(element="H")
        h2 = Atom(element="H")
        h3 = Atom(element="H")
        struct_L.add_entity(c1, c2, h1, h2, h3)
        struct_L.add_link(Bond(c1, c2))
        struct_L.add_link(Bond(c1, h1))
        struct_L.add_link(Bond(c2, h2))
        struct_L.add_link(Bond(c2, h3))
        c2["port"] = ">"

        struct_R = Atomistic()
        c3 = Atom(element="C")
        c4 = Atom(element="C")
        h4 = Atom(element="H")
        h5 = Atom(element="H")
        h6 = Atom(element="H")
        struct_R.add_entity(c3, c4, h4, h5, h6)
        struct_R.add_link(Bond(c3, c4))
        struct_R.add_link(Bond(c3, h4))
        struct_R.add_link(Bond(c4, h5))
        struct_R.add_link(Bond(c4, h6))
        c3["port"] = "<"

        bond_react_reacter = BondReactReacter(
            name="test_ordering",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
            radius=3,
        )

        port_L = find_port_atom(struct_L, ">")
        port_R = find_port_atom(struct_R, "<")
        result = bond_react_reacter.run(struct_L, struct_R, port_L, port_R)
        template = result.template

        # Pre and post must have same number of atoms
        pre_atoms = list(template.pre.atoms)
        post_atoms = list(template.post.atoms)
        assert len(pre_atoms) == len(post_atoms)

        # React_ids must be in same order
        pre_rids = [a.get("react_id") for a in pre_atoms]
        post_rids = [a.get("react_id") for a in post_atoms]
        assert pre_rids == post_rids, (
            f"Pre and post atom ordering mismatch! Pre: {pre_rids}, Post: {post_rids}"
        )

    def test_template_unified_type_mapping(self):
        """Test that pre and post use same type-to-ID mapping after write.

        When types are converted to integer IDs, both pre and post must use
        the same mapping so equivalent atoms have the same type ID.
        """
        import tempfile
        from pathlib import Path

        struct_L = Atomistic()
        c1 = Atom(element="C", type="type_A")
        c2 = Atom(element="C", type="type_B")
        struct_L.add_entity(c1, c2)
        struct_L.add_link(Bond(c1, c2, type="bond_X"))
        c2["port"] = ">"

        struct_R = Atomistic()
        c3 = Atom(element="C", type="type_A")
        c4 = Atom(element="C", type="type_B")
        struct_R.add_entity(c3, c4)
        struct_R.add_link(Bond(c3, c4, type="bond_X"))
        c3["port"] = "<"

        bond_react_reacter = BondReactReacter(
            name="test_types",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
            radius=2,
        )

        port_L = find_port_atom(struct_L, ">")
        port_R = find_port_atom(struct_R, "<")
        result = bond_react_reacter.run(struct_L, struct_R, port_L, port_R)
        template = result.template

        # Add types for testing
        for atom in template.pre.atoms:
            if "type" not in atom.data:
                atom["type"] = f"type_{atom.get('element')}"
        for atom in template.post.atoms:
            if "type" not in atom.data:
                atom["type"] = f"type_{atom.get('element')}"
        for bond in template.pre.bonds:
            if "type" not in bond.data:
                bond["type"] = "bond_type"
        for bond in template.post.bonds:
            if "type" not in bond.data:
                bond["type"] = "bond_type"
        for angle in template.pre.angles:
            if "type" not in angle.data:
                angle["type"] = "angle_type"
        for angle in template.post.angles:
            if "type" not in angle.data:
                angle["type"] = "angle_type"
        for dihedral in template.pre.dihedrals:
            if "type" not in dihedral.data:
                dihedral["type"] = "dihedral_type"
        for dihedral in template.post.dihedrals:
            if "type" not in dihedral.data:
                dihedral["type"] = "dihedral_type"

        from molpy.io import write_lammps_molecule
        from molpy.io.data.lammps_bond_react import apply_type_maps, collect_type_maps

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "test_types"

            # Serialize via the io layer: unified string-type → ID maps
            # across pre and post, applied to both frames.
            template.assign_atom_ids()
            pre_frame = template.pre.to_frame()
            post_frame = template.post.to_frame()
            _, type_maps = collect_type_maps([pre_frame, post_frame])
            for tpl_frame in (pre_frame, post_frame):
                apply_type_maps(tpl_frame, type_maps)

            pre_mol = Path(f"{base_path}_pre.mol")
            post_mol = Path(f"{base_path}_post.mol")
            write_lammps_molecule(pre_mol, pre_frame)
            write_lammps_molecule(post_mol, post_frame)

            with open(pre_mol) as f:
                pre_content = f.read()
            with open(post_mol) as f:
                post_content = f.read()

            # Extract types section
            import re

            pre_match = re.search(r"Types\n\n(.*?)\n\n", pre_content, re.DOTALL)
            post_match = re.search(r"Types\n\n(.*?)\n\n", post_content, re.DOTALL)

            assert pre_match and post_match, "Types section not found in mol files"

            # Parse types
            pre_types = {}
            for line in pre_match.group(1).strip().split("\n"):
                parts = line.split()
                if len(parts) >= 2:
                    pre_types[int(parts[0])] = int(parts[1])

            post_types = {}
            for line in post_match.group(1).strip().split("\n"):
                parts = line.split()
                if len(parts) >= 2:
                    post_types[int(parts[0])] = int(parts[1])

            # Equivalences should have same types (for non-chemically-changed atoms)
            # At minimum, atoms at same position should have same type
            for atom_id in pre_types:
                if atom_id in post_types:
                    assert pre_types[atom_id] == post_types[atom_id], (
                        f"Type mismatch at position {atom_id}: "
                        f"pre={pre_types[atom_id]}, post={post_types[atom_id]}"
                    )

    def test_template_edge_atoms_type_preserved(self):
        """Test that edge atoms have consistent types in pre and post.

        LAMMPS fix bond/react requires edge atoms to have unchanged types.
        """
        struct_L = Atomistic()
        c1 = Atom(element="C", type="edge_type")
        c2 = Atom(element="C", type="center_type")
        h1 = Atom(element="H", type="leaving_type")
        struct_L.add_entity(c1, c2, h1)
        struct_L.add_link(Bond(c1, c2))
        struct_L.add_link(Bond(c2, h1))
        c2["port"] = ">"

        struct_R = Atomistic()
        c3 = Atom(element="C", type="edge_type")
        c4 = Atom(element="C", type="center_type")
        h2 = Atom(element="H", type="leaving_type")
        struct_R.add_entity(c3, c4, h2)
        struct_R.add_link(Bond(c3, c4))
        struct_R.add_link(Bond(c4, h2))
        c4["port"] = "<"

        bond_react_reacter = BondReactReacter(
            name="test_edge",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
            radius=2,
        )

        port_L = find_port_atom(struct_L, ">")
        port_R = find_port_atom(struct_R, "<")
        result = bond_react_reacter.run(struct_L, struct_R, port_L, port_R)
        template = result.template

        # Get edge atom react_ids
        edge_rids = {a.get("react_id") for a in template.edge_atoms}

        # Build type lookup by react_id
        pre_types = {a.get("react_id"): a.get("type") for a in template.pre.atoms}
        post_types = {a.get("react_id"): a.get("type") for a in template.post.atoms}

        # Edge atoms must have same types in pre and post
        for rid in edge_rids:
            pre_type = pre_types.get(rid)
            post_type = post_types.get(rid)
            assert pre_type == post_type, (
                f"Edge atom (react_id={rid}) type changed: "
                f"pre={pre_type}, post={post_type}"
            )

    def test_template_coordinates_match_for_equivalences(self):
        """Test that equivalent atoms have matching coordinates in pre and post.

        LAMMPS uses coordinates for molecular geometry verification.
        """
        struct_L = Atomistic()
        c1 = Atom(element="C", x=0.0, y=0.0, z=0.0)
        c2 = Atom(element="C", x=1.5, y=0.0, z=0.0)
        h1 = Atom(element="H", x=2.0, y=1.0, z=0.0)
        struct_L.add_entity(c1, c2, h1)
        struct_L.add_link(Bond(c1, c2))
        struct_L.add_link(Bond(c2, h1))
        c2["port"] = ">"

        struct_R = Atomistic()
        c3 = Atom(element="C", x=0.0, y=0.0, z=3.0)
        c4 = Atom(element="C", x=1.5, y=0.0, z=3.0)
        h2 = Atom(element="H", x=2.0, y=1.0, z=3.0)
        struct_R.add_entity(c3, c4, h2)
        struct_R.add_link(Bond(c3, c4))
        struct_R.add_link(Bond(c4, h2))
        c4["port"] = "<"

        bond_react_reacter = BondReactReacter(
            name="test_coords",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
            radius=2,
        )

        port_L = find_port_atom(struct_L, ">")
        port_R = find_port_atom(struct_R, "<")
        result = bond_react_reacter.run(struct_L, struct_R, port_L, port_R)
        template = result.template

        # Build coord lookup by react_id
        pre_coords = {}
        for a in template.pre.atoms:
            rid = a.get("react_id")
            pre_coords[rid] = (a.get("x", 0), a.get("y", 0), a.get("z", 0))

        post_coords = {}
        for a in template.post.atoms:
            rid = a.get("react_id")
            post_coords[rid] = (a.get("x", 0), a.get("y", 0), a.get("z", 0))

        # All atoms must have matching coordinates
        for rid in pre_coords:
            assert rid in post_coords, f"react_id {rid} not in post"
            pre_c = pre_coords[rid]
            post_c = post_coords[rid]
            for i, (p, q) in enumerate(zip(pre_c, post_c)):
                assert abs(p - q) < 1e-6, (
                    f"Coordinate mismatch for react_id {rid}: "
                    f"pre={pre_c}, post={post_c}"
                )


class TestImproperPropagation:
    """Impropers cloned into pre must survive into the post template."""

    def test_untouched_improper_present_in_post(self) -> None:
        """Every pre improper not touching deleted atoms has a post match."""
        result = _run_sp2_reaction(radius=4)
        template = result.template
        assert template is not None

        deleted_rids = {a["react_id"] for a in template.deleted_atoms}
        untouched_pre = [
            imp
            for imp in template.pre.impropers
            if not any(ep["react_id"] in deleted_rids for ep in imp.endpoints)
        ]
        # Fixture sanity: the c1 improper survives (no deleted endpoint)
        assert len(untouched_pre) >= 1, (
            "Fixture broken: expected at least one untouched improper in pre"
        )

        post_keys = {_improper_key(imp) for imp in template.post.impropers}
        for imp in untouched_pre:
            assert _improper_key(imp) in post_keys, (
                f"Pre improper {_improper_key(imp)} lost in post template "
                f"(post has {sorted(post_keys)})"
            )

    def test_untouched_improper_count_equal_pre_vs_post(self) -> None:
        """Count of untouched impropers matches between pre and post."""
        result = _run_sp2_reaction(radius=4)
        template = result.template
        assert template is not None

        deleted_rids = {a["react_id"] for a in template.deleted_atoms}
        untouched_pre_keys = {
            _improper_key(imp)
            for imp in template.pre.impropers
            if not any(ep["react_id"] in deleted_rids for ep in imp.endpoints)
        }
        matched_post = [
            imp
            for imp in template.post.impropers
            if _improper_key(imp) in untouched_pre_keys
        ]
        assert len(matched_post) == len(untouched_pre_keys), (
            f"Untouched improper count mismatch: pre={len(untouched_pre_keys)}, "
            f"post matches={len(matched_post)}"
        )


class TestMapDeterminism:
    """The written .map must be deterministic, with ordered InitiatorIDs."""

    def test_map_file_deterministic_across_runs(self, tmp_path: Path) -> None:
        """Two equivalent runs write byte-identical .map files."""
        result1 = _run_sp2_reaction(radius=2)
        result2 = _run_sp2_reaction(radius=2)
        assert result1.template is not None
        assert result2.template is not None

        write_bond_react_map(result1.template, tmp_path / "run1")
        write_bond_react_map(result2.template, tmp_path / "run2")

        bytes1 = (tmp_path / "run1.map").read_bytes()
        bytes2 = (tmp_path / "run2.map").read_bytes()
        assert bytes1 == bytes2, "Map files differ between equivalent runs"

    def test_map_determinism_initiator_order_left_anchor_first(
        self, tmp_path: Path
    ) -> None:
        """InitiatorIDs has exactly 2 entries; first is the LEFT anchor."""
        result = _run_sp2_reaction(radius=2)
        template = result.template
        assert template is not None

        write_bond_react_map(template, tmp_path / "rxn")
        sections = _parse_map_sections(tmp_path / "rxn.map")

        initiator_ids = [row[0] for row in sections["InitiatorIDs"]]
        assert len(initiator_ids) == 2

        rid_to_idx = {
            a["react_id"]: i for i, a in enumerate(template.pre.atoms, start=1)
        }
        left_anchor_idx = rid_to_idx[template.initiator_atoms[0]["react_id"]]
        assert initiator_ids[0] == left_anchor_idx, (
            f"InitiatorIDs not in template.initiator_atoms order: "
            f"first written={initiator_ids[0]}, left anchor={left_anchor_idx}"
        )


class TestInitiatorValidation:
    """Templates must have exactly 2 initiators, never on the boundary."""

    def test_initiator_on_template_boundary_raises_radius_error(self) -> None:
        """Anchor sitting on the template edge (radius too small) raises."""
        left = _build_sp2_left()
        right = _build_sp2_right()
        port_l = find_port_atom(left, ">")
        port_r = find_port_atom(right, "<")

        with pytest.raises(ValueError, match="radius"):
            reacter = _make_sp2_reacter(radius=0)
            reacter.run(left, right, port_atom_L=port_l, port_atom_R=port_r)

    def test_initiator_count_must_be_two_in_validation(self) -> None:
        """validate_bond_react_template rejects a 1-initiator template."""
        from molpy.reacter.bond_react import validate_bond_react_template

        template = _make_manual_template_with_one_initiator()
        with pytest.raises(ValueError, match="(?i)initiator|anchor"):
            validate_bond_react_template(template)

    def test_initiator_count_must_be_two_in_map_writer(self, tmp_path: Path) -> None:
        """write_bond_react_map rejects a 1-initiator template."""
        template = _make_manual_template_with_one_initiator()
        with pytest.raises(ValueError):
            write_bond_react_map(template, tmp_path / "bad")


class TestEdgeConsistency:
    """Edge atoms must keep identical type and charge in pre and post."""

    def test_edge_consistent_template_passes_validation(self) -> None:
        """A real-flow template with untouched edge atoms validates cleanly."""
        from molpy.reacter.bond_react import validate_bond_react_template

        result = _run_sp2_reaction(radius=2)
        template = result.template
        assert template is not None
        # Fixture sanity: radius=2 leaves a genuine boundary (c0 is an edge)
        assert len(template.edge_atoms) >= 1

        assert validate_bond_react_template(template) is None

    def test_edge_atom_type_mismatch_raises(self) -> None:
        """Mutated post edge atom type -> ValueError naming both values."""
        from molpy.reacter.bond_react import validate_bond_react_template

        result = _run_sp2_reaction(radius=2)
        template = result.template
        assert template is not None
        assert len(template.edge_atoms) >= 1

        edge_rid = template.edge_atoms[0]["react_id"]
        pre_type = str(template.pre_react_id_to_atom[edge_rid]["type"])
        post_atom = template.post_react_id_to_atom[edge_rid]
        post_atom["type"] = "XX"

        with pytest.raises(ValueError, match="radius") as excinfo:
            validate_bond_react_template(template)
        message = str(excinfo.value)
        assert pre_type in message
        assert "XX" in message

    def test_edge_atom_charge_mismatch_raises(self) -> None:
        """Mutated post edge atom charge -> ValueError naming both values."""
        from molpy.reacter.bond_react import validate_bond_react_template

        result = _run_sp2_reaction(radius=2)
        template = result.template
        assert template is not None
        assert len(template.edge_atoms) >= 1

        edge_rid = template.edge_atoms[0]["react_id"]
        post_atom = template.post_react_id_to_atom[edge_rid]
        post_atom["charge"] = 0.75  # pre charge is -0.18

        with pytest.raises(ValueError, match="radius") as excinfo:
            validate_bond_react_template(template)
        message = str(excinfo.value)
        assert "0.18" in message
        assert "0.75" in message


class TestMutationHygiene:
    """run() must not mutate the caller's left/right structures."""

    def test_run_does_not_mutate_caller_atoms_with_react_id(self) -> None:
        """No caller atom carries react_id after run()."""
        left = _build_sp2_left()
        right = _build_sp2_right()
        reacter = _make_sp2_reacter(radius=4)
        port_l = find_port_atom(left, ">")
        port_r = find_port_atom(right, "<")

        reacter.run(left, right, port_atom_L=port_l, port_atom_R=port_r)

        stamped_left = [a for a in left.atoms if "react_id" in a.data]
        stamped_right = [a for a in right.atoms if "react_id" in a.data]
        assert stamped_left == [], (
            f"run() mutated caller's left: {len(stamped_left)} atoms stamped "
            f"with react_id"
        )
        assert stamped_right == [], (
            f"run() mutated caller's right: {len(stamped_right)} atoms stamped "
            f"with react_id"
        )

    def test_run_does_not_mutate_caller_atom_counts(self) -> None:
        """Caller atom counts are unchanged after run()."""
        left = _build_sp2_left()
        right = _build_sp2_right()
        left_count = len(list(left.atoms))
        right_count = len(list(right.atoms))
        reacter = _make_sp2_reacter(radius=4)
        port_l = find_port_atom(left, ">")
        port_r = find_port_atom(right, "<")

        reacter.run(left, right, port_atom_L=port_l, port_atom_R=port_r)

        assert len(list(left.atoms)) == left_count
        assert len(list(right.atoms)) == right_count


class TestChargeConservation:
    """Template generation must check total charge pre vs post."""

    def test_charge_conservation_tolerance_constant(self) -> None:
        """CHARGE_CONSERVATION_TOL is exported and equals 1e-6."""
        from molpy.reacter.bond_react import CHARGE_CONSERVATION_TOL

        assert CHARGE_CONSERVATION_TOL == 1e-6

    def test_charge_imbalance_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Post total charge off by 1.0 -> WARNING on the module logger."""
        from molpy.reacter.bond_react import validate_bond_react_template

        result = _run_sp2_reaction(radius=4)
        template = result.template
        assert template is not None

        # Perturb a non-edge atom (the left anchor) in post only
        anchor_rid = template.initiator_atoms[0]["react_id"]
        post_atom = template.post_react_id_to_atom[anchor_rid]
        post_atom["charge"] = float(post_atom["charge"]) + 1.0

        with caplog.at_level(logging.WARNING, logger="molpy.reacter.bond_react"):
            validate_bond_react_template(template)

        warning_records = [
            r
            for r in caplog.records
            if r.name == "molpy.reacter.bond_react" and r.levelno >= logging.WARNING
        ]
        assert len(warning_records) >= 1, (
            "Expected a charge-conservation WARNING on logger "
            "'molpy.reacter.bond_react'"
        )

    def test_charge_conserving_template_logs_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Untouched real-flow template logs no charge warning."""
        from molpy.reacter.bond_react import validate_bond_react_template

        result = _run_sp2_reaction(radius=4)
        template = result.template
        assert template is not None

        with caplog.at_level(logging.WARNING, logger="molpy.reacter.bond_react"):
            validate_bond_react_template(template)

        warning_records = [
            r
            for r in caplog.records
            if r.name == "molpy.reacter.bond_react" and r.levelno >= logging.WARNING
        ]
        assert warning_records == []


class TestReacterMapInvariants:
    """Structural invariants of the written .map file."""

    def test_map_invariant_equivalences_are_bijection(self, tmp_path: Path) -> None:
        """Equivalences: unique pre IDs, unique post IDs, count == atom count."""
        result = _run_sp2_reaction(radius=2)
        template = result.template
        assert template is not None

        write_bond_react_map(template, tmp_path / "rxn")
        sections = _parse_map_sections(tmp_path / "rxn.map")

        equivalences = sections["Equivalences"]
        pre_ids = [row[0] for row in equivalences]
        post_ids = [row[1] for row in equivalences]
        n_atoms = len(list(template.pre.atoms))

        assert len(equivalences) == n_atoms
        assert len(set(pre_ids)) == n_atoms, "Duplicate pre IDs in Equivalences"
        assert len(set(post_ids)) == n_atoms, "Duplicate post IDs in Equivalences"

    def test_map_invariant_initiators_and_edges_disjoint(self, tmp_path: Path) -> None:
        """Exactly 2 InitiatorIDs and EdgeIDs do not overlap them."""
        result = _run_sp2_reaction(radius=2)
        template = result.template
        assert template is not None

        write_bond_react_map(template, tmp_path / "rxn")
        sections = _parse_map_sections(tmp_path / "rxn.map")

        initiator_ids = {row[0] for row in sections["InitiatorIDs"]}
        edge_ids = {row[0] for row in sections["EdgeIDs"]}

        assert len(initiator_ids) == 2
        assert initiator_ids & edge_ids == set()

    def test_map_invariant_deleted_atoms_in_delete_ids(self, tmp_path: Path) -> None:
        """Every deleted atom's pre index appears in DeleteIDs."""
        result = _run_sp2_reaction(radius=2)
        template = result.template
        assert template is not None

        write_bond_react_map(template, tmp_path / "rxn")
        sections = _parse_map_sections(tmp_path / "rxn.map")

        rid_to_idx = {
            a["react_id"]: i for i, a in enumerate(template.pre.atoms, start=1)
        }
        expected = {rid_to_idx[a["react_id"]] for a in template.deleted_atoms}
        delete_ids = {row[0] for row in sections["DeleteIDs"]}

        # Fixture sanity: one hydrogen leaves from each side
        assert len(expected) == 2
        assert expected <= delete_ids
