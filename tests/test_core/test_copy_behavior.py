"""
Tests for Atomistic copy behavior.

Focus on verifying that copy operations correctly duplicate all entities and bonds,
and that no orphan references remain.
"""

from molpy.core.atomistic import Atom, Atomistic, Bond


class TestAtomisticCopy:
    """Test Atomistic.copy() behavior."""

    def test_copy_preserves_all_atoms(self):
        """Test that copy duplicates all atoms."""
        asm = Atomistic()
        a1 = Atom(symbol="C")
        a2 = Atom(symbol="H")
        a3 = Atom(symbol="O")

        asm.entities.add(a1)
        asm.entities.add(a2)
        asm.entities.add(a3)

        # Copy
        asm_copy = asm.copy()

        atoms_orig = list(asm.atoms)
        atoms_copy = list(asm_copy.atoms)

        assert len(atoms_copy) == len(atoms_orig) == 3
        assert all(a.get("symbol") in ["C", "H", "O"] for a in atoms_copy)

    def test_copy_preserves_all_bonds(self):
        """Test that copy duplicates all bonds."""
        asm = Atomistic()
        a1 = Atom(symbol="C")
        a2 = Atom(symbol="H")
        a3 = Atom(symbol="O")

        asm.entities.add(a1)
        asm.entities.add(a2)
        asm.entities.add(a3)

        b1 = Bond(a1, a2)
        b2 = Bond(a1, a3)

        asm.links.add(b1)
        asm.links.add(b2)

        # Copy
        asm_copy = asm.copy()

        bonds_orig = list(asm.bonds)
        bonds_copy = list(asm_copy.bonds)

        assert len(bonds_copy) == len(bonds_orig) == 2

    def test_copy_bonds_reference_copied_atoms(self):
        """Test that copied bonds reference copied atoms, not original atoms."""
        asm = Atomistic()
        a1 = Atom(symbol="C")
        a2 = Atom(symbol="H")

        asm.entities.add(a1)
        asm.entities.add(a2)

        b1 = Bond(a1, a2)
        asm.links.add(b1)

        # Copy
        asm_copy = asm.copy()

        atoms_copy = list(asm_copy.atoms)
        bonds_copy = list(asm_copy.bonds)

        assert len(bonds_copy) == 1
        bond = bonds_copy[0]

        # Check that bond endpoints are in the copied atoms
        assert bond.endpoints[0] in atoms_copy
        assert bond.endpoints[1] in atoms_copy

        # Check that bond endpoints are NOT the original atoms
        assert bond.endpoints[0] is not a1
        assert bond.endpoints[1] is not a2

    def test_copy_no_orphan_bonds(self):
        """Test that copy doesn't create orphan bonds (bonds with missing endpoints)."""
        asm = Atomistic()
        a1 = Atom(symbol="C")
        a2 = Atom(symbol="H")
        a3 = Atom(symbol="O")

        asm.entities.add(a1)
        asm.entities.add(a2)
        asm.entities.add(a3)

        b1 = Bond(a1, a2)
        b2 = Bond(a1, a3)
        b3 = Bond(a2, a3)

        asm.links.add(b1)
        asm.links.add(b2)
        asm.links.add(b3)

        # Copy
        asm_copy = asm.copy()

        # Get all entities in a set
        entities_set = set()
        for entity_type in asm_copy.entities.classes():
            for entity in asm_copy.entities.bucket(entity_type):
                entities_set.add(entity)

        # Check that all bond endpoints are in entities
        orphan_bonds = []
        for bond in asm_copy.bonds:
            for ep in bond.endpoints:
                if ep not in entities_set:
                    orphan_bonds.append(bond)
                    break

        assert (
            len(orphan_bonds) == 0
        ), f"Found {len(orphan_bonds)} orphan bonds after copy"

    def test_copy_independence(self):
        """Test that modifications to copy don't affect original."""
        asm = Atomistic()
        a1 = Atom(symbol="C")
        a2 = Atom(symbol="H")

        asm.entities.add(a1)
        asm.entities.add(a2)

        b1 = Bond(a1, a2)
        asm.links.add(b1)

        # Copy
        asm_copy = asm.copy()

        # Modify copy
        a3 = Atom(symbol="O")
        asm_copy.entities.add(a3)

        # Check original is unchanged
        assert len(list(asm.atoms)) == 2
        assert len(list(asm_copy.atoms)) == 3


class TestAtomisticCopyWithPorts:
    """Test Atomistic.copy() behavior with port markers."""

    def test_copy_preserves_structure(self):
        """Test that structure copy preserves all atoms and bonds."""
        # Create structure
        struct = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        h1 = Atom(symbol="H")

        struct.entities.add(c1)
        struct.entities.add(c2)
        struct.entities.add(h1)

        b1 = Bond(c1, c2)
        b2 = Bond(c1, h1)

        struct.links.add(b1)
        struct.links.add(b2)

        # Copy
        struct_copy = struct.copy()

        # Check structure
        atoms_copy = list(struct_copy.atoms)
        bonds_copy = list(struct_copy.bonds)

        assert len(atoms_copy) == 3
        assert len(bonds_copy) == 2

    def test_copy_no_orphan_bonds(self):
        """Test that structure copy doesn't create orphan bonds."""
        # Create structure
        struct = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        o1 = Atom(symbol="O")
        h1 = Atom(symbol="H")

        struct.entities.add(c1)
        struct.entities.add(c2)
        struct.entities.add(o1)
        struct.entities.add(h1)

        b1 = Bond(c1, c2)
        b2 = Bond(c1, o1)
        b3 = Bond(o1, h1)

        struct.links.add(b1)
        struct.links.add(b2)
        struct.links.add(b3)

        # Mark port on atom
        c1["port"] = "port_1"

        # Copy
        struct_copy = struct.copy()

        # Check for orphan bonds
        entities_set = set()
        for entity_type in struct_copy.entities.classes():
            for entity in struct_copy.entities.bucket(entity_type):
                entities_set.add(entity)

        orphan_bonds = []
        for bond in struct_copy.bonds:
            for ep in bond.endpoints:
                if ep not in entities_set:
                    orphan_bonds.append(bond)
                    break

        assert (
            len(orphan_bonds) == 0
        ), f"Found {len(orphan_bonds)} orphan bonds in structure copy"

    def test_copy_ports_remapped(self):
        """Test that port markers are correctly preserved on copied atoms."""
        # Create structure
        struct = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")

        struct.entities.add(c1)
        struct.entities.add(c2)

        b1 = Bond(c1, c2)
        struct.links.add(b1)

        # Mark port on atom
        c1["port"] = "port_1"

        # Copy
        struct_copy = struct.copy()

        # Check port marker is preserved
        atoms_copy = list(struct_copy.atoms)
        port_atom = None
        for atom in atoms_copy:
            if atom.get("port") == "port_1":
                port_atom = atom
                break

        assert port_atom is not None, "Port marker should be preserved in copy"
        assert port_atom is not c1  # Should be a copied atom

    def test_multiple_copies_independent(self):
        """Test that multiple copies are independent."""
        # Create structure
        struct = Atomistic()
        c1 = Atom(symbol="C")
        h1 = Atom(symbol="H")

        struct.entities.add(c1)
        struct.entities.add(h1)

        b1 = Bond(c1, h1)
        struct.links.add(b1)

        # Create multiple copies
        copy1 = struct.copy()
        copy2 = struct.copy()
        copy3 = struct.copy()

        # Modify each copy
        copy1.entities.add(Atom(symbol="O"))
        copy2.entities.add(Atom(symbol="N"))

        # Check independence
        assert len(list(struct.atoms)) == 2
        assert len(list(copy1.atoms)) == 3
        assert len(list(copy2.atoms)) == 3
        assert len(list(copy3.atoms)) == 2
