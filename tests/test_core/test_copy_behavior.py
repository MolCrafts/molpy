"""
Tests for Atomistic and Monomer copy behavior.

Focus on verifying that copy operations correctly duplicate all entities and bonds,
and that no orphan references remain.
"""

from molpy import Atom, Atomistic, Bond
from molpy.core.wrappers.monomer import Monomer


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

        assert len(orphan_bonds) == 0, (
            f"Found {len(orphan_bonds)} orphan bonds after copy"
        )

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


class TestMonomerCopy:
    """Test Monomer.copy() behavior."""

    def test_copy_preserves_structure(self):
        """Test that monomer copy preserves all atoms and bonds."""
        # Create atomistic structure
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        h1 = Atom(symbol="H")

        asm.entities.add(c1)
        asm.entities.add(c2)
        asm.entities.add(h1)

        b1 = Bond(c1, c2)
        b2 = Bond(c1, h1)

        asm.links.add(b1)
        asm.links.add(b2)

        # Create monomer
        monomer = Monomer(asm)

        # Copy
        monomer_copy = monomer.copy()

        # Check structure
        asm_copy = monomer_copy.unwrap()
        atoms_copy = list(asm_copy.atoms)
        bonds_copy = list(asm_copy.bonds)

        assert len(atoms_copy) == 3
        assert len(bonds_copy) == 2

    def test_copy_no_orphan_bonds(self):
        """Test that monomer copy doesn't create orphan bonds."""
        # Create atomistic structure
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")
        o1 = Atom(symbol="O")
        h1 = Atom(symbol="H")

        asm.entities.add(c1)
        asm.entities.add(c2)
        asm.entities.add(o1)
        asm.entities.add(h1)

        b1 = Bond(c1, c2)
        b2 = Bond(c1, o1)
        b3 = Bond(o1, h1)

        asm.links.add(b1)
        asm.links.add(b2)
        asm.links.add(b3)

        # Create monomer with port
        monomer = Monomer(asm)
        monomer.set_port("port_1", c1, role="reactive")

        # Copy
        monomer_copy = monomer.copy()

        # Check for orphan bonds
        asm_copy = monomer_copy.unwrap()

        entities_set = set()
        for entity_type in asm_copy.entities.classes():
            for entity in asm_copy.entities.bucket(entity_type):
                entities_set.add(entity)

        orphan_bonds = []
        for bond in asm_copy.bonds:
            for ep in bond.endpoints:
                if ep not in entities_set:
                    orphan_bonds.append(bond)
                    break

        assert len(orphan_bonds) == 0, (
            f"Found {len(orphan_bonds)} orphan bonds in monomer copy"
        )

    def test_copy_ports_remapped(self):
        """Test that ports are correctly remapped to copied atoms."""
        # Create atomistic structure
        asm = Atomistic()
        c1 = Atom(symbol="C")
        c2 = Atom(symbol="C")

        asm.entities.add(c1)
        asm.entities.add(c2)

        b1 = Bond(c1, c2)
        asm.links.add(b1)

        # Create monomer with port
        monomer = Monomer(asm)
        monomer.set_port("port_1", c1, role="reactive")

        # Copy
        monomer_copy = monomer.copy()

        # Check port is remapped
        port_copy = monomer_copy.get_port("port_1")
        assert port_copy is not None
        assert port_copy.target is not c1  # Should point to copied atom

        # Check port target is in copied atoms
        atoms_copy = list(monomer_copy.unwrap().atoms)
        assert port_copy.target in atoms_copy

    def test_multiple_copies_independent(self):
        """Test that multiple copies are independent."""
        # Create atomistic structure
        asm = Atomistic()
        c1 = Atom(symbol="C")
        h1 = Atom(symbol="H")

        asm.entities.add(c1)
        asm.entities.add(h1)

        b1 = Bond(c1, h1)
        asm.links.add(b1)

        # Create monomer
        monomer = Monomer(asm)

        # Create multiple copies
        copy1 = monomer.copy()
        copy2 = monomer.copy()
        copy3 = monomer.copy()

        # Modify each copy
        copy1.unwrap().entities.add(Atom(symbol="O"))
        copy2.unwrap().entities.add(Atom(symbol="N"))

        # Check independence
        assert len(list(monomer.unwrap().atoms)) == 2
        assert len(list(copy1.unwrap().atoms)) == 3
        assert len(list(copy2.unwrap().atoms)) == 3
        assert len(list(copy3.unwrap().atoms)) == 2
