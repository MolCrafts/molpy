#!/usr/bin/env python3
"""Unit tests for TemplateReacter class.

Tests cover:
- TemplateReacter initialization (with Reacter constructor parameters)
- run_with_template() method
- react_id assignment
- Pre/post template consistency
- TemplateResult structure
- Error handling
"""

import pytest

from molpy.core.atomistic import Atom, Atomistic, Bond
from molpy.reacter import (
    form_single_bond,
    select_port,
    select_none,
    select_one_hydrogen,
)
from molpy.reacter.selectors import find_port_atom
from molpy.reacter.template import TemplateReacter, TemplateResult, write_template_files


class TestTemplateReacter:
    """Test TemplateReacter class."""

    def test_template_reacter_initialization(self):
        """Test TemplateReacter initialization with Reacter constructor parameters."""
        template_reacter = TemplateReacter(
            name="test_reaction",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
            radius=4,
        )

        # Check internal reacter is created
        assert template_reacter.reacter is not None
        assert template_reacter.reacter.name == "test_reaction"
        assert template_reacter.radius == 4
        assert template_reacter._react_id_counter == 0

    def test_template_reacter_default_radius(self):
        """Test TemplateReacter with default radius."""
        template_reacter = TemplateReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        assert template_reacter.radius == 4  # default

    def test_run_with_template_basic(self):
        """Test basic run_with_template() execution."""
        # Create left structure: C-H
        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        # Create right structure: C-H
        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        # Create TemplateReacter
        template_reacter = TemplateReacter(
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
        result, template = template_reacter.run_with_template(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )

        # Validate ReactionResult
        assert result is not None
        assert isinstance(result.product_info.product, Atomistic)
        assert len(list(result.product_info.product.atoms)) == 2  # 2 C atoms

        # Validate TemplateResult
        assert isinstance(template, TemplateResult)
        assert template.pre is not None
        assert template.post is not None
        assert len(template.init_atoms) == 2  # Two port atoms
        assert len(template.removed_atoms) == 2  # Two H atoms removed

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
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        template_reacter = TemplateReacter(
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
        template_reacter._assign_react_ids(struct_L)
        template_reacter._assign_react_ids(struct_R)

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
        c_L = Atom(symbol="C")
        struct_L.add_entity(c_L)
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        struct_R.add_entity(c_R)
        c_R["port"] = "2"

        template_reacter = TemplateReacter(
            name="addition",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_none,
            leaving_selector_right=select_none,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result, template = template_reacter.run_with_template(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )

        # No atoms removed
        assert len(template.removed_atoms) == 0
        assert len(list(result.product_info.product.atoms)) == 2

        # Pre and post should have same atoms
        pre_rids = {a["react_id"] for a in template.pre.atoms}
        post_rids = {a["react_id"] for a in template.post.atoms}
        assert pre_rids == post_rids

    def test_run_with_template_compute_topology_false(self):
        """Test run_with_template() with compute_topology=False."""
        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        template_reacter = TemplateReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result, template = template_reacter.run_with_template(
            struct_L,
            struct_R,
            port_atom_L=port_atom_L,
            port_atom_R=port_atom_R,
            compute_topology=False,
        )

        # Should still work
        assert result is not None
        assert template is not None
        assert len(list(result.product_info.product.atoms)) == 2

    def test_run_with_template_init_atoms(self):
        """Test that init_atoms (port atoms) are correctly identified."""
        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        template_reacter = TemplateReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result, template = template_reacter.run_with_template(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )

        # Check init_atoms
        assert len(template.init_atoms) == 2
        init_rids = {a["react_id"] for a in template.init_atoms}
        assert port_atom_L["react_id"] in init_rids
        assert port_atom_R["react_id"] in init_rids

    def test_run_with_template_removed_atoms_tracked(self):
        """Test that removed atoms are correctly tracked in template."""
        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L1 = Atom(symbol="H")
        h_L2 = Atom(symbol="H")
        struct_L.add_entity(c_L, h_L1, h_L2)
        struct_L.add_link(Bond(c_L, h_L1), Bond(c_L, h_L2))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        template_reacter = TemplateReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result, template = template_reacter.run_with_template(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )

        # Check removed atoms
        assert len(template.removed_atoms) == 2
        removed_rids = {a["react_id"] for a in template.removed_atoms}

        # Removed atoms should be in pre but not in product
        pre_rids = {a["react_id"] for a in template.pre.atoms}
        product_rids = {a["react_id"] for a in result.product_info.product.atoms}

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
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        template_reacter = TemplateReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result, template = template_reacter.run_with_template(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )

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
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")
        struct_L.add_entity(c1, c2, h1, h2)
        struct_L.add_link(Bond(c1, c2), Bond(c1, h1), Bond(c2, h2))
        c1["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        # Test with radius=1 (smaller subgraph)
        template_reacter_1 = TemplateReacter(
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
        result1, template1 = template_reacter_1.run_with_template(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )

        # Test with radius=3 (larger subgraph)
        template_reacter_3 = TemplateReacter(
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

        result2, template2 = template_reacter_3.run_with_template(
            struct_L2, struct_R2, port_atom_L=port_atom_L2, port_atom_R=port_atom_R2
        )

        # Larger radius should include more atoms (or same)
        assert len(list(template2.pre.atoms)) >= len(list(template1.pre.atoms))

    def test_assign_react_ids_preserves_existing(self):
        """Test that _assign_react_ids preserves existing react_ids."""
        struct = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        struct.add_entity(c1, c2)

        template_reacter = TemplateReacter(
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
        template_reacter._assign_react_ids(struct)

        # First atom should keep its react_id
        assert c1["react_id"] == 100
        # Second atom should get a new react_id
        assert "react_id" in c2.data
        assert c2["react_id"] != 100

    def test_template_topology_consistency(self):
        """Test that pre and post have consistent topology."""
        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        struct_L.get_topo(gen_angle=True, gen_dihe=True)
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        struct_R.get_topo(gen_angle=True, gen_dihe=True)
        c_R["port"] = "2"

        template_reacter = TemplateReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result, template = template_reacter.run_with_template(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )

        # Pre should have topology
        assert len(list(template.pre.bonds)) > 0

        # Post should not have bonds between deleted and non-deleted atoms
        removed_rids = {a["react_id"] for a in template.removed_atoms}
        for bond in template.post.bonds:
            ep1_rid = bond.endpoints[0].get("react_id")
            ep2_rid = bond.endpoints[1].get("react_id")
            ep1_deleted = ep1_rid in removed_rids
            ep2_deleted = ep2_rid in removed_rids
            assert not (
                ep1_deleted != ep2_deleted
            ), f"Post has bond between deleted and non-deleted atom: {ep1_rid} - {ep2_rid}"

    def test_write_template_files(self):
        """Test write_template_files function."""
        from molpy.reacter.template import write_template_files
        from pathlib import Path
        import tempfile

        struct_L = Atomistic()
        c_L = Atom(symbol="C")
        h_L = Atom(symbol="H")
        struct_L.add_entity(c_L, h_L)
        struct_L.add_link(Bond(c_L, h_L))
        c_L["port"] = "1"

        struct_R = Atomistic()
        c_R = Atom(symbol="C")
        h_R = Atom(symbol="H")
        struct_R.add_entity(c_R, h_R)
        struct_R.add_link(Bond(c_R, h_R))
        c_R["port"] = "2"

        template_reacter = TemplateReacter(
            name="test",
            anchor_selector_left=select_port,
            anchor_selector_right=select_port,
            leaving_selector_left=select_one_hydrogen,
            leaving_selector_right=select_one_hydrogen,
            bond_former=form_single_bond,
        )

        port_atom_L = find_port_atom(struct_L, "1")
        port_atom_R = find_port_atom(struct_R, "2")
        result, template = template_reacter.run_with_template(
            struct_L, struct_R, port_atom_L=port_atom_L, port_atom_R=port_atom_R
        )

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
            write_template_files(base_path, template, typifier=None)

            # Check files exist
            pre_mol = Path(f"{base_path}_pre.mol")
            post_mol = Path(f"{base_path}_post.mol")
            map_file = Path(f"{base_path}.map")

            assert pre_mol.exists()
            assert post_mol.exists()
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
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        h1 = Atom(symbol="H")
        h2 = Atom(symbol="H")
        h3 = Atom(symbol="H")
        struct_L.add_entity(c1, c2, h1, h2, h3)
        struct_L.add_link(Bond(c1, c2))
        struct_L.add_link(Bond(c1, h1))
        struct_L.add_link(Bond(c2, h2))
        struct_L.add_link(Bond(c2, h3))
        c2["port"] = ">"

        struct_R = Atomistic()
        c3 = Atom(symbol="C")
        c4 = Atom(symbol="C")
        h4 = Atom(symbol="H")
        h5 = Atom(symbol="H")
        h6 = Atom(symbol="H")
        struct_R.add_entity(c3, c4, h4, h5, h6)
        struct_R.add_link(Bond(c3, c4))
        struct_R.add_link(Bond(c3, h4))
        struct_R.add_link(Bond(c4, h5))
        struct_R.add_link(Bond(c4, h6))
        c3["port"] = "<"

        template_reacter = TemplateReacter(
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
        result, template = template_reacter.run_with_template(
            struct_L, struct_R, port_L, port_R
        )

        # Pre and post must have same number of atoms
        pre_atoms = list(template.pre.atoms)
        post_atoms = list(template.post.atoms)
        assert len(pre_atoms) == len(post_atoms)

        # React_ids must be in same order
        pre_rids = [a.get("react_id") for a in pre_atoms]
        post_rids = [a.get("react_id") for a in post_atoms]
        assert pre_rids == post_rids, (
            f"Pre and post atom ordering mismatch! "
            f"Pre: {pre_rids}, Post: {post_rids}"
        )

    def test_template_unified_type_mapping(self):
        """Test that pre and post use same type-to-ID mapping after write.

        When types are converted to integer IDs, both pre and post must use
        the same mapping so equivalent atoms have the same type ID.
        """
        import tempfile
        from pathlib import Path

        struct_L = Atomistic()
        c1 = Atom(symbol="C", type="type_A")
        c2 = Atom(symbol="C", type="type_B")
        struct_L.add_entity(c1, c2)
        struct_L.add_link(Bond(c1, c2, type="bond_X"))
        c2["port"] = ">"

        struct_R = Atomistic()
        c3 = Atom(symbol="C", type="type_A")
        c4 = Atom(symbol="C", type="type_B")
        struct_R.add_entity(c3, c4)
        struct_R.add_link(Bond(c3, c4, type="bond_X"))
        c3["port"] = "<"

        template_reacter = TemplateReacter(
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
        result, template = template_reacter.run_with_template(
            struct_L, struct_R, port_L, port_R
        )

        # Add types for testing
        for atom in template.pre.atoms:
            if "type" not in atom.data:
                atom["type"] = f"type_{atom.get('symbol')}"
        for atom in template.post.atoms:
            if "type" not in atom.data:
                atom["type"] = f"type_{atom.get('symbol')}"
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

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "test_types"
            write_template_files(base_path, template, typifier=None)

            # Read back pre and post mol files
            pre_mol = Path(f"{base_path}_pre.mol")
            post_mol = Path(f"{base_path}_post.mol")

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
        c1 = Atom(symbol="C", type="edge_type")
        c2 = Atom(symbol="C", type="center_type")
        h1 = Atom(symbol="H", type="leaving_type")
        struct_L.add_entity(c1, c2, h1)
        struct_L.add_link(Bond(c1, c2))
        struct_L.add_link(Bond(c2, h1))
        c2["port"] = ">"

        struct_R = Atomistic()
        c3 = Atom(symbol="C", type="edge_type")
        c4 = Atom(symbol="C", type="center_type")
        h2 = Atom(symbol="H", type="leaving_type")
        struct_R.add_entity(c3, c4, h2)
        struct_R.add_link(Bond(c3, c4))
        struct_R.add_link(Bond(c4, h2))
        c4["port"] = "<"

        template_reacter = TemplateReacter(
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
        result, template = template_reacter.run_with_template(
            struct_L, struct_R, port_L, port_R
        )

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
        c1 = Atom(symbol="C", x=0.0, y=0.0, z=0.0)
        c2 = Atom(symbol="C", x=1.5, y=0.0, z=0.0)
        h1 = Atom(symbol="H", x=2.0, y=1.0, z=0.0)
        struct_L.add_entity(c1, c2, h1)
        struct_L.add_link(Bond(c1, c2))
        struct_L.add_link(Bond(c2, h1))
        c2["port"] = ">"

        struct_R = Atomistic()
        c3 = Atom(symbol="C", x=0.0, y=0.0, z=3.0)
        c4 = Atom(symbol="C", x=1.5, y=0.0, z=3.0)
        h2 = Atom(symbol="H", x=2.0, y=1.0, z=3.0)
        struct_R.add_entity(c3, c4, h2)
        struct_R.add_link(Bond(c3, c4))
        struct_R.add_link(Bond(c4, h2))
        c4["port"] = "<"

        template_reacter = TemplateReacter(
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
        result, template = template_reacter.run_with_template(
            struct_L, struct_R, port_L, port_R
        )

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
