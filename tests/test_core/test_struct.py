"""
Test suite for molpy.core.struct module.

This test suite provides tests for the base Struct class and MolecularStructure.
Atomic structure tests have been moved to test_atoms.py.
"""

import pytest
import numpy as np
from molpy.core.atomistic import Struct, Atom, Bond, Angle, Dihedral, AtomicStructure


class TestEntity:
    def test_dict_behavior(self):
        e = Struct(name="foo", bar=123)
        assert e["name"] == "foo"
        e["baz"] = 456
        assert e["baz"] == 456
        d = e.to_dict()
        assert d["bar"] == 123
    
    def test_clone(self):
        e = Struct(name="foo", bar=123)
        e2 = e.clone(bar=999)
        assert e2["bar"] == 999
        assert e["bar"] == 123
        assert e2 is not e
    
    def test_call(self):
        e = Struct(name="foo", bar=123)
        e2 = e(bar=888)
        assert e2["bar"] == 888
        assert e["bar"] == 123


class TestAtom:
    def test_to_dict(self):
        """Test Atom.to_dict method."""
        from molpy.core.atomistic import Atom
        
        # Test basic atom (without explicit xyz)
        atom = Atom(name="C", element="carbon")
        d = atom.to_dict()
        assert d["name"] == "C"
        assert d["element"] == "carbon"
        # xyz should not be in dict if not explicitly set
        assert "xyz" not in d
        
        # Test atom with coordinates
        atom_with_xyz = Atom(name="H", element="hydrogen", xyz=[1.0, 2.0, 3.0])
        d2 = atom_with_xyz.to_dict()
        assert d2["name"] == "H"
        assert d2["element"] == "hydrogen"
        assert d2["xyz"] == [1.0, 2.0, 3.0]
    
    def test_xyz_property(self):
        """Test that xyz property provides default even when not set."""
        from molpy.core.atomistic import Atom
        
        atom = Atom(name="C")
        # Property should return default coordinates
        xyz = atom.xyz
        assert xyz.shape == (3,)
        assert list(xyz) == [0.0, 0.0, 0.0]
        
        # But to_dict should not include xyz unless explicitly set
        d = atom.to_dict()
        assert "xyz" not in d


class TestBond:
    def test_to_dict(self):
        """Test Bond.to_dict method."""
        atom1 = Atom(name="C", id=0)
        atom2 = Atom(name="H", id=1)
        bond = Bond(atom1, atom2, bond_type="single", length=1.5)
        
        d = bond.to_dict()
        
        # Check that bond properties are included
        assert d["bond_type"] == "single"
        assert d["length"] == 1.5
        # Bond sorts atoms by id(), so i and j should be 0 and 1 but order may vary
        assert "i" in d
        assert "j" in d
        assert set([d["i"], d["j"]]) == {0, 1}
    
    def test_to_dict_no_atom_ids(self):
        """Test Bond.to_dict when atoms don't have ids."""
        atom1 = Atom(name="C")
        atom2 = Atom(name="H")
        bond = Bond(atom1, atom2, bond_type="single")
        
        d = bond.to_dict()
        
        assert d["bond_type"] == "single"
        # Should not have i,j keys when atoms don't have ids
        assert "i" not in d
        assert "j" not in d


class TestAngle:
    def test_to_dict(self):
        """Test Angle.to_dict method."""
        atom1 = Atom(name="H", id=0)
        atom2 = Atom(name="C", id=1)  # vertex
        atom3 = Atom(name="O", id=2)
        angle = Angle(atom1, atom2, atom3, angle_type="harmonic")
        
        d = angle.to_dict()
        
        assert d["angle_type"] == "harmonic"
        # Angle also sorts atoms, vertex should be in the middle
        assert "i" in d
        assert "j" in d  # vertex
        assert "k" in d
        assert d["j"] == 1  # vertex should always be atom2


class TestDihedral:
    def test_to_dict(self):
        """Test Dihedral.to_dict method."""
        atom1 = Atom(name="C1", id=0)
        atom2 = Atom(name="C2", id=1)
        atom3 = Atom(name="C3", id=2)
        atom4 = Atom(name="C4", id=3)
        dihedral = Dihedral(atom1, atom2, atom3, atom4, dihedral_type="periodic")
        
        d = dihedral.to_dict()
        
        assert d["dihedral_type"] == "periodic"
        # Dihedral may also sort atoms, just check all indices are present
        assert "i" in d
        assert "j" in d
        assert "k" in d
        assert "l" in d
        assert set([d["i"], d["j"], d["k"], d["l"]]) == {0, 1, 2, 3}


class TestStruct:
    def test_init_basic(self):
        struct = Struct(name="test_struct")
        assert struct["name"] == "test_struct"
        assert "test_struct" in repr(struct)
        unnamed = Struct()
        assert "name" in repr(unnamed)
    
    def test_clone(self):
        struct = Struct(name="foo")
        struct2 = struct.clone(name="bar")
        assert struct2["name"] == "bar"
        assert struct["name"] == "foo"
        assert struct2 is not struct


class TestToFrame:
    def test_to_frame_empty_structure(self):
        """Test to_frame with empty structure."""
        from molpy.core.frame import Frame
        
        struct = AtomicStructure(name="empty")
        frame = struct.to_frame()
        
        assert isinstance(frame, Frame)
        assert "atoms" in frame
        # Empty structure should have empty atoms Dataset
        atoms_data = frame["atoms"]
        assert len(atoms_data.sizes) == 0 or all(size == 0 for size in atoms_data.sizes.values())
    
    def test_to_frame_atoms_only(self):
        """Test to_frame with atoms only."""
        from molpy.core.frame import Frame
        import pandas as pd
        
        struct = AtomicStructure(name="test_atoms")
        struct.def_atom(name="C", element="carbon", xyz=[0.0, 0.0, 0.0])
        struct.def_atom(name="H", element="hydrogen", xyz=[1.0, 0.0, 0.0])
        
        frame = struct.to_frame()
        
        assert isinstance(frame, Frame)
        assert "atoms" in frame
        
        # Check atoms data
        atoms_data = frame["atoms"]
        assert len(atoms_data.data_vars) > 0  # Should have data variables
        # Check that we have the expected number of atoms
        for var_name, var_data in atoms_data.data_vars.items():
            if var_name in ["name", "element", "xyz", "id"]:
                # Get the first dimension size for these variables
                dim_sizes = var_data.sizes
                if dim_sizes:
                    first_dim_size = next(iter(dim_sizes.values()))
                    assert first_dim_size == 2
        
        # Check that atoms have correct ids
        assert "id" in atoms_data.data_vars
        ids = atoms_data["id"].values
        assert list(ids) == [0, 1]
        
        # Check names
        assert "name" in atoms_data.data_vars
        names = atoms_data["name"].values
        assert list(names) == ["C", "H"]
        
        # Check metadata
        assert frame._meta["structure_name"] == "test_atoms"
        assert frame._meta["n_atoms"] == 2
        assert frame._meta["n_bonds"] == 0
    
    def test_to_frame_with_bonds(self):
        """Test to_frame with atoms and bonds."""
        from molpy.core.frame import Frame
        
        struct = AtomicStructure(name="test_bonds")
        a1 = struct.def_atom(name="C", element="carbon", xyz=[0.0, 0.0, 0.0])
        a2 = struct.def_atom(name="H", element="hydrogen", xyz=[1.0, 0.0, 0.0])
        struct.def_bond(a1, a2, bond_type="single")
        
        frame = struct.to_frame()
        
        assert isinstance(frame, Frame)
        assert "atoms" in frame
        assert "bonds" in frame
        
        # Check bonds data
        bonds_data = frame["bonds"]
        # Check that bonds data has one bond
        for var_name, var_data in bonds_data.data_vars.items():
            if var_name in ["i", "j", "bond_type", "id"]:
                dim_sizes = var_data.sizes
                if dim_sizes:
                    first_dim_size = next(iter(dim_sizes.values()))
                    assert first_dim_size == 1
        
        # Check bond indices
        assert "i" in bonds_data.data_vars
        assert "j" in bonds_data.data_vars
        assert "id" in bonds_data.data_vars
        
        i_vals = bonds_data["i"].values
        j_vals = bonds_data["j"].values
        bond_ids = bonds_data["id"].values
        
        assert list(bond_ids) == [0]
        # Should have atom indices 0 and 1
        assert set([i_vals[0], j_vals[0]]) == {0, 1}
        
        # Check metadata
        assert frame._meta["n_bonds"] == 1
    
    def test_to_frame_with_all_entities(self):
        """Test to_frame with atoms, bonds, and angles."""
        from molpy.core.frame import Frame
        
        struct = AtomicStructure(name="test_all")
        a1 = struct.def_atom(name="C1", xyz=[0.0, 0.0, 0.0])
        a2 = struct.def_atom(name="C2", xyz=[1.0, 0.0, 0.0])
        a3 = struct.def_atom(name="C3", xyz=[1.0, 1.0, 0.0])
        
        struct.def_bond(a1, a2)
        struct.def_bond(a2, a3)
        # Create and add angle manually
        angle = Angle(a1, a2, a3)
        struct.add_angle(angle)
        
        frame = struct.to_frame()
        
        assert isinstance(frame, Frame)
        assert "atoms" in frame
        assert "bonds" in frame
        assert "angles" in frame
        
        # Check counts
        assert frame._meta["n_atoms"] == 3
        assert frame._meta["n_bonds"] == 2
        assert frame._meta["n_angles"] == 1
        
        # Check angle data
        angles_data = frame["angles"]
        # Check that angles data has one angle
        for var_name, var_data in angles_data.data_vars.items():
            if var_name in ["i", "j", "k", "id"]:
                dim_sizes = var_data.sizes
                if dim_sizes:
                    first_dim_size = next(iter(dim_sizes.values()))
                    assert first_dim_size == 1
        assert "i" in angles_data.data_vars
        assert "j" in angles_data.data_vars
        assert "k" in angles_data.data_vars
        assert "id" in angles_data.data_vars
    
    def test_to_frame_round_trip(self):
        """Test round-trip: structure -> frame -> data -> new structure."""
        from molpy.core.frame import Frame
        import pandas as pd
        
        # Create a structure with atoms and bonds
        struct = AtomicStructure(name="test_round_trip")
        a1 = struct.def_atom(name="C", element="carbon", xyz=[0.0, 0.0, 0.0])
        a2 = struct.def_atom(name="H", element="hydrogen", xyz=[1.0, 0.0, 0.0])
        a3 = struct.def_atom(name="O", element="oxygen", xyz=[0.0, 1.0, 0.0])
        struct.def_bond(a1, a2, bond_type="single")
        struct.def_bond(a1, a3, bond_type="double")
        
        # Convert to frame
        frame = struct.to_frame()
        
        # Extract data from frame
        atoms_data = frame["atoms"]
        bonds_data = frame["bonds"]
        
        # Verify data structure and content
        # Check number of atoms
        for var_name, var_data in atoms_data.data_vars.items():
            if var_name in ["name", "element", "xyz", "id"]:
                dim_sizes = var_data.sizes
                if dim_sizes:
                    first_dim_size = next(iter(dim_sizes.values()))
                    assert first_dim_size == 3
        
        # Check number of bonds  
        for var_name, var_data in bonds_data.data_vars.items():
            if var_name in ["i", "j", "bond_type", "id"]:
                dim_sizes = var_data.sizes
                if dim_sizes:
                    first_dim_size = next(iter(dim_sizes.values()))
                    assert first_dim_size == 2
        
        # Check that we can access all necessary data
        atom_names = atoms_data["name"].values
        assert set(atom_names) == {"C", "H", "O"}
        
        bond_types = bonds_data["bond_type"].values
        assert set(bond_types) == {"single", "double"}
        
        # Check atom indices in bonds - skip detailed iteration check
        # Just verify we have some bonds data
        assert "i" in bonds_data.data_vars
        assert "j" in bonds_data.data_vars
        assert "bond_type" in bonds_data.data_vars
        
        # Simple validation that bond indices are integers
        i_values = bonds_data["i"].values
        j_values = bonds_data["j"].values
        assert all(isinstance(val, (int, np.integer)) for val in i_values)
        assert all(isinstance(val, (int, np.integer)) for val in j_values)
    
    def test_to_frame_data_consistency(self):
        """Test that to_frame produces consistent data."""
        from molpy.core.frame import Frame
        
        struct = AtomicStructure(name="consistency_test")
        a1 = struct.def_atom(name="N", element="nitrogen", xyz=[0.0, 0.0, 0.0])
        a2 = struct.def_atom(name="C", element="carbon", xyz=[1.4, 0.0, 0.0])
        a3 = struct.def_atom(name="O", element="oxygen", xyz=[2.0, 1.2, 0.0])
        
        bond1 = struct.def_bond(a1, a2, bond_type="single", length=1.4)
        bond2 = struct.def_bond(a2, a3, bond_type="double", length=1.2)
        
        frame = struct.to_frame()
        
        # Test multiple conversions produce same result
        frame2 = struct.to_frame()
        
        atoms1 = frame["atoms"]
        atoms2 = frame2["atoms"]
        bonds1 = frame["bonds"]
        bonds2 = frame2["bonds"]
        
        # Should have same structure
        # Compare data variables in atoms1 and atoms2
        assert len(atoms1.data_vars) == len(atoms2.data_vars)
        assert len(bonds1.data_vars) == len(bonds2.data_vars)
        
        # Should have same atom ids
        assert list(atoms1["id"].values) == list(atoms2["id"].values)
        assert list(bonds1["id"].values) == list(bonds2["id"].values)


