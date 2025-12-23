"""
Test Atomistic class API for creating and adding entities.

Tests the enhanced API that distinguishes between:
- def_* methods: factory functions that create and add entities
- add_* methods: add existing entity objects
- Plural forms: batch operations
"""

import numpy as np
import pytest

from molpy import Angle, Atom, Atomistic, Bond, Dihedral


class TestAtomisticFactoryMethods:
    """Test def_* factory methods that create and add entities."""

    def test_def_atom_creates_and_adds(self):
        """Test def_atom creates an Atom and adds it to the structure."""
        struct = Atomistic()
        atom = struct.def_atom(symbol="C", xyz=[0, 0, 0])

        assert isinstance(atom, Atom)
        assert atom.get("symbol") == "C"
        assert len(struct.atoms) == 1
        assert next(iter(struct.atoms)) is atom

    def test_def_bond_creates_and_adds(self):
        """Test def_bond creates a Bond between two atoms."""
        struct = Atomistic()
        a1 = struct.def_atom(symbol="C", xyz=[0, 0, 0])
        a2 = struct.def_atom(symbol="H", xyz=[1, 0, 0])

        bond = struct.def_bond(a1, a2, order=1)

        assert isinstance(bond, Bond)
        assert bond.itom is a1
        assert bond.jtom is a2
        assert bond.get("order") == 1
        assert len(struct.bonds) == 1

    def test_def_angle_creates_and_adds(self):
        """Test def_angle creates an Angle between three atoms."""
        struct = Atomistic()
        a1 = struct.def_atom(symbol="H")
        a2 = struct.def_atom(symbol="C")
        a3 = struct.def_atom(symbol="H")

        angle = struct.def_angle(a1, a2, a3, theta=109.5)

        assert isinstance(angle, Angle)
        assert angle.itom is a1
        assert angle.jtom is a2
        assert angle.ktom is a3
        assert angle.get("theta") == 109.5
        assert len(struct.angles) == 1

    def test_def_dihedral_creates_and_adds(self):
        """Test def_dihedral creates a Dihedral between four atoms."""
        struct = Atomistic()
        a1 = struct.def_atom(symbol="H")
        a2 = struct.def_atom(symbol="C")
        a3 = struct.def_atom(symbol="C")
        a4 = struct.def_atom(symbol="H")

        dihe = struct.def_dihedral(a1, a2, a3, a4, phi=180.0)

        assert isinstance(dihe, Dihedral)
        assert dihe.itom is a1
        assert dihe.jtom is a2
        assert dihe.ktom is a3
        assert dihe.ltom is a4
        assert dihe.get("phi") == 180.0
        assert len(struct.dihedrals) == 1


class TestAtomisticAddMethods:
    """Test add_* methods that add existing entity objects."""

    def test_add_atom_adds_existing(self):
        """Test add_atom adds an already created Atom object."""
        struct = Atomistic()
        atom = Atom(symbol="C", xyz=[0, 0, 0])

        result = struct.add_atom(atom)

        assert result is atom
        assert len(struct.atoms) == 1
        assert next(iter(struct.atoms)) is atom

    def test_add_bond_adds_existing(self):
        """Test add_bond adds an already created Bond object."""
        struct = Atomistic()
        a1 = struct.def_atom(symbol="C")
        a2 = struct.def_atom(symbol="H")
        bond = Bond(a1, a2, order=1)

        result = struct.add_bond(bond)

        assert result is bond
        assert len(struct.bonds) == 1
        assert next(iter(struct.bonds)) is bond

    def test_add_angle_adds_existing(self):
        """Test add_angle adds an already created Angle object."""
        struct = Atomistic()
        a1 = struct.def_atom(symbol="H")
        a2 = struct.def_atom(symbol="C")
        a3 = struct.def_atom(symbol="H")
        angle = Angle(a1, a2, a3, theta=109.5)

        result = struct.add_angle(angle)

        assert result is angle
        assert len(struct.angles) == 1
        assert next(iter(struct.angles)) is angle

    def test_add_dihedral_adds_existing(self):
        """Test add_dihedral adds an already created Dihedral object."""
        struct = Atomistic()
        a1 = struct.def_atom(symbol="H")
        a2 = struct.def_atom(symbol="C")
        a3 = struct.def_atom(symbol="C")
        a4 = struct.def_atom(symbol="H")
        dihe = Dihedral(a1, a2, a3, a4, phi=180.0)

        result = struct.add_dihedral(dihe)

        assert result is dihe
        assert len(struct.dihedrals) == 1
        assert next(iter(struct.dihedrals)) is dihe


class TestAtomisticBatchFactoryMethods:
    """Test def_*s plural factory methods for batch creation."""

    def test_def_atoms_batch_create(self):
        """Test def_atoms creates multiple atoms at once."""
        struct = Atomistic()

        atoms = struct.def_atoms(
            [
                {"symbol": "C", "xyz": [0, 0, 0]},
                {"symbol": "H", "xyz": [1, 0, 0]},
                {"symbol": "H", "xyz": [0, 1, 0]},
            ]
        )

        assert len(atoms) == 3
        assert all(isinstance(a, Atom) for a in atoms)
        assert len(struct.atoms) == 3
        assert np.array_equal(struct.atoms["symbol"], ["C", "H", "H"])

    def test_def_bonds_batch_create(self):
        """Test def_bonds creates multiple bonds at once."""
        struct = Atomistic()
        a1 = struct.def_atom(symbol="C")
        a2 = struct.def_atom(symbol="H")
        a3 = struct.def_atom(symbol="H")

        bonds = struct.def_bonds(
            [
                (a1, a2, {"order": 1}),
                (a1, a3, {"order": 1}),
            ]
        )

        assert len(bonds) == 2
        assert all(isinstance(b, Bond) for b in bonds)
        assert len(struct.bonds) == 2

    def test_def_angles_batch_create(self):
        """Test def_angles creates multiple angles at once."""
        struct = Atomistic()
        a1 = struct.def_atom(symbol="H")
        a2 = struct.def_atom(symbol="C")
        a3 = struct.def_atom(symbol="H")
        a4 = struct.def_atom(symbol="H")

        angles = struct.def_angles(
            [
                (a1, a2, a3, {"theta": 109.5}),
                (a1, a2, a4, {"theta": 109.5}),
            ]
        )

        assert len(angles) == 2
        assert all(isinstance(ang, Angle) for ang in angles)
        assert len(struct.angles) == 2

    def test_def_dihedrals_batch_create(self):
        """Test def_dihedrals creates multiple dihedrals at once."""
        struct = Atomistic()
        atoms = struct.def_atoms([{"symbol": "H"} for _ in range(5)])

        dihedrals = struct.def_dihedrals(
            [
                (atoms[0], atoms[1], atoms[2], atoms[3], {"phi": 0.0}),
                (atoms[1], atoms[2], atoms[3], atoms[4], {"phi": 180.0}),
            ]
        )

        assert len(dihedrals) == 2
        assert all(isinstance(d, Dihedral) for d in dihedrals)
        assert len(struct.dihedrals) == 2


class TestAtomisticBatchAddMethods:
    """Test add_*s plural methods for batch addition of existing entities."""

    def test_add_atoms_batch_add(self):
        """Test add_atoms adds multiple existing Atom objects."""
        struct = Atomistic()
        atoms = [
            Atom(symbol="C", xyz=[0, 0, 0]),
            Atom(symbol="H", xyz=[1, 0, 0]),
            Atom(symbol="H", xyz=[0, 1, 0]),
        ]

        result = struct.add_atoms(atoms)

        assert result == atoms
        assert len(struct.atoms) == 3
        for atom in atoms:
            assert atom in struct.atoms

    def test_add_bonds_batch_add(self):
        """Test add_bonds adds multiple existing Bond objects."""
        struct = Atomistic()
        a1 = struct.def_atom(symbol="C")
        a2 = struct.def_atom(symbol="H")
        a3 = struct.def_atom(symbol="H")

        bonds = [
            Bond(a1, a2, order=1),
            Bond(a1, a3, order=1),
        ]

        result = struct.add_bonds(bonds)

        assert result == bonds
        assert len(struct.bonds) == 2

    def test_add_angles_batch_add(self):
        """Test add_angles adds multiple existing Angle objects."""
        struct = Atomistic()
        atoms = struct.def_atoms([{"symbol": "H"} for _ in range(4)])

        angles = [
            Angle(atoms[0], atoms[1], atoms[2]),
            Angle(atoms[1], atoms[2], atoms[3]),
        ]

        result = struct.add_angles(angles)

        assert result == angles
        assert len(struct.angles) == 2

    def test_add_dihedrals_batch_add(self):
        """Test add_dihedrals adds multiple existing Dihedral objects."""
        struct = Atomistic()
        atoms = struct.def_atoms([{"symbol": "H"} for _ in range(5)])

        dihedrals = [
            Dihedral(atoms[0], atoms[1], atoms[2], atoms[3]),
            Dihedral(atoms[1], atoms[2], atoms[3], atoms[4]),
        ]

        result = struct.add_dihedrals(dihedrals)

        assert result == dihedrals
        assert len(struct.dihedrals) == 2


class TestAtomisticSemanticClarity:
    """Test that the new API provides semantic clarity."""

    def test_def_vs_add_atom_distinction(self):
        """Test clear distinction between factory and add methods."""
        struct = Atomistic()

        # Factory: creates new atom
        atom1 = struct.def_atom(symbol="C")
        assert atom1 in struct.atoms

        # Add: adds existing atom
        atom2 = Atom(symbol="N")
        struct.add_atom(atom2)
        assert atom2 in struct.atoms

        assert len(struct.atoms) == 2

    def test_singular_vs_plural_distinction(self):
        """Test clear distinction between singular and plural forms."""
        struct = Atomistic()

        # Singular: one at a time
        struct.def_atom(symbol="C")

        # Plural: batch operation
        atoms = struct.def_atoms(
            [
                {"symbol": "H"},
                {"symbol": "H"},
            ]
        )

        assert len(struct.atoms) == 3
        assert len(atoms) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
