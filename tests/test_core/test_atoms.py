"""
Test suite for molpy.core.atoms module.

This test suite provides comprehensive coverage for atomic structure classes
including Atom, Bond, Angle, Dihedral, Improper, and AtomicStructure.
"""

import pytest
import numpy as np
from molpy.core.struct import Atom, Bond, Angle, Dihedral, Improper, AtomicStructure, Entities


class TestAtom:
    def test_init(self):
        atom = Atom(name="C", xyz=[1,2,3])
        assert atom.name == "C"
        assert np.allclose(atom.xyz, [1,2,3])
    
    def test_xyz_setter(self):
        atom = Atom(name="H")
        atom.xyz = [4,5,6]
        assert np.allclose(atom.xyz, [4,5,6])
    
    def test_move_and_distance(self):
        a1 = Atom(xyz=[0,0,0])
        a2 = Atom(xyz=[1,0,0])
        assert abs(a1.distance_to(a2) - 1.0) < 1e-10
        a1.move([1,2,3])
        assert np.allclose(a1.xyz, [1,2,3])
    
    def test_clone(self):
        atom = Atom(name="C", xyz=[1,2,3])
        atom2 = atom.clone(name="N")
        assert atom2.name == "N"
        assert np.allclose(atom2.xyz, [1,2,3])
        assert atom2 is not atom


class TestBond:
    def test_init_basic(self):
        atom1 = Atom(name="C1", xyz=[0, 0, 0])
        atom2 = Atom(name="C2", xyz=[1.5, 0, 0])
        bond = Bond(atom1, atom2)
        assert bond.atom1 in [atom1, atom2]
        assert bond.atom2 in [atom1, atom2]
        assert bond.atom1 is not bond.atom2
    
    def test_length(self):
        a1 = Atom(xyz=[0,0,0])
        a2 = Atom(xyz=[3,4,0])
        bond = Bond(a1, a2)
        assert abs(bond.length - 5.0) < 1e-10
    
    def test_clone(self):
        a1 = Atom(xyz=[0,0,0])
        a2 = Atom(xyz=[1,0,0])
        bond = Bond(a1, a2)
        bond2 = bond.clone()
        assert isinstance(bond2, Bond)
        assert len(bond2.atoms) == 2
        for orig, clone in zip(bond.atoms, bond2.atoms):
            assert np.allclose(orig.xyz, clone.xyz)
            assert orig is not clone


class TestAngle:
    def test_init_basic(self):
        a1 = Atom(xyz=[0,0,0])
        v = Atom(xyz=[1,0,0])
        a2 = Atom(xyz=[2,0,0])
        angle = Angle(a1, v, a2)
        assert angle.atom1 is not None
        assert angle.vertex is not None
        assert angle.atom2 is not None
    
    def test_value(self):
        a1 = Atom(xyz=[0,0,0])
        v = Atom(xyz=[1,0,0])
        a2 = Atom(xyz=[1,1,0])
        angle = Angle(a1, v, a2)
        assert abs(np.degrees(angle.value) - 90) < 1e-6
    
    def test_clone(self):
        a1 = Atom(xyz=[0,0,0])
        v = Atom(xyz=[1,0,0])
        a2 = Atom(xyz=[2,0,0])
        angle = Angle(a1, v, a2)
        angle2 = angle.clone()
        assert isinstance(angle2, Angle)
        assert len(angle2.atoms) == 3


class TestDihedral:
    def test_init_basic(self):
        a1 = Atom(xyz=[0, 0, 0])
        a2 = Atom(xyz=[1, 0, 0])
        a3 = Atom(xyz=[1, 1, 0])
        a4 = Atom(xyz=[2, 1, 0])
        dihedral = Dihedral(a1, a2, a3, a4)
        assert dihedral.atom1 is not None
        assert dihedral.atom2 is not None
        assert dihedral.atom3 is not None
        assert dihedral.atom4 is not None
    
    def test_value(self):
        a1 = Atom(xyz=[0, 0, 0])
        a2 = Atom(xyz=[1, 0, 0])
        a3 = Atom(xyz=[2, 0, 0])
        a4 = Atom(xyz=[2, 1, 0])
        dihedral = Dihedral(a1, a2, a3, a4)
        value = dihedral.value
        assert isinstance(value, float)
        assert -np.pi <= value <= np.pi
    
    def test_clone(self):
        a1 = Atom(xyz=[0, 0, 0])
        a2 = Atom(xyz=[1, 0, 0])
        a3 = Atom(xyz=[1, 1, 0])
        a4 = Atom(xyz=[2, 1, 0])
        dihedral = Dihedral(a1, a2, a3, a4)
        dihedral2 = dihedral.clone()
        assert isinstance(dihedral2, Dihedral)
        assert len(dihedral2.atoms) == 4


class TestImproper:
    def test_init_basic(self):
        a1 = Atom(xyz=[0, 0, 0])
        a2 = Atom(xyz=[1, 0, 0])
        a3 = Atom(xyz=[1, 1, 0])
        a4 = Atom(xyz=[2, 1, 0])
        improper = Improper(a1, a2, a3, a4)
        assert improper.atom1 is not None
        assert len(improper.atoms) == 4


class TestEntities:
    def test_add_get_remove(self):
        ent = Entities()
        atom = Atom(name="C")
        ent.add(atom)
        assert atom in ent
        assert len(ent) == 1
        ent.remove(atom)
        assert atom not in ent
        assert len(ent) == 0
    
    def test_get_by_condition(self):
        ent = Entities()
        a1 = Atom(name="C")
        a2 = Atom(name="H")
        ent.add(a1)
        ent.add(a2)
        carbon = ent.get_by(lambda x: x.name == "C")
        assert carbon is a1


class TestAtomicStructure:
    def test_init(self):
        s = AtomicStructure()
        assert len(s.atoms) == 0
        assert len(s.bonds) == 0
        assert len(s.angles) == 0
        assert len(s.dihedrals) == 0
    
    def test_add_atom_bond_angle_dihedral(self):
        s = AtomicStructure()
        a1 = s.def_atom(name="C", xyz=[0,0,0])
        a2 = s.def_atom(name="H", xyz=[1,0,0])
        b = s.def_bond(a1, a2)
        ang = Angle(a1, a2, a1.clone(xyz=[2,0,0]))
        dih = Dihedral(a1, a2, a1.clone(xyz=[2,0,0]), a1.clone(xyz=[2,1,0]))
        s.add_angle(ang)
        s.add_dihedral(dih)
        assert a1 in s.atoms
        assert a2 in s.atoms
        assert b in s.bonds
        assert ang in s.angles
        assert dih in s.dihedrals
    
    def test_remove_atom_bond(self):
        s = AtomicStructure()
        a1 = s.def_atom(name="C", xyz=[0,0,0])
        a2 = s.def_atom(name="H", xyz=[1,0,0])
        b = s.def_bond(a1, a2)
        s.remove_bond(b)
        assert b not in s.bonds
        s.remove_atom(a1)
        assert a1 not in s.atoms
    
    def test_move_xyz(self):
        s = AtomicStructure()
        a1 = s.def_atom(name="C", xyz=[0,0,0])
        a2 = s.def_atom(name="H", xyz=[1,0,0])
        s.move([1,2,3])
        assert np.allclose(a1.xyz, [1,2,3])
        assert np.allclose(a2.xyz, [2,2,3])
        # xyz setter
        s.xyz = np.array([[7,8,9],[10,11,12]])
        assert np.allclose(a1.xyz, [7,8,9])
        assert np.allclose(a2.xyz, [10,11,12])
    
    def test_add_atoms_bonds_batch(self):
        s = AtomicStructure()
        atoms = [Atom(name=f"C{i}", xyz=[i,0,0]) for i in range(3)]
        s.add_atoms(atoms)
        assert all(a in s.atoms for a in atoms)
        bonds = [Bond(atoms[0], atoms[1]), Bond(atoms[1], atoms[2])]
        s.add_bonds(bonds)
        assert all(b in s.bonds for b in bonds)
    
    def test_hierarchy(self):
        parent = AtomicStructure(name="parent")
        child = AtomicStructure(name="child")
        parent.add_child(child)
        assert child.parent is parent
        assert child in parent.children
        assert parent.is_root
        assert child.depth == 1
    
    def test_add_struct_and_concat(self):
        s1 = AtomicStructure(name="s1")
        s2 = AtomicStructure(name="s2")
        a1 = s1.def_atom(name="C", xyz=[0,0,0])
        a2 = s2.def_atom(name="H", xyz=[1,0,0])
        s1.def_bond(a1, a1.clone(xyz=[0,1,0]))
        s2.def_bond(a2, a2.clone(xyz=[1,1,0]))
        s1.add_struct(s2)
        assert a2 in s1.atoms
        assert len(s1.children) == 1
        s3 = AtomicStructure.concat("combo", [s1, s2])
        assert s3["name"] == "combo"
        assert a1 in s3.atoms and a2 in s3.atoms
    
    def test_get_topology(self):
        s = AtomicStructure()
        a1 = s.def_atom(name="C", xyz=[0,0,0], id=1)
        a2 = s.def_atom(name="H", xyz=[1,0,0], id=2)
        s.def_bond(a1, a2)
        topo = s.get_topology(attrs=["id"])
        assert hasattr(topo, "add_atoms")
        assert hasattr(topo, "add_bonds")
    
    def test_to_frame(self):
        """Test conversion of AtomicStructure to Frame."""
        from molpy.core.frame import Frame
        
        # Test empty structure
        s = AtomicStructure()
        frame = s.to_frame()
        assert isinstance(frame, Frame)
        
        # Test structure with atoms
        s = AtomicStructure(name="test_structure")
        a1 = s.def_atom(name="C", element="C", xyz=[0.0, 0.0, 0.0])
        a2 = s.def_atom(name="H", element="H", xyz=[1.0, 0.0, 0.0])
        a3 = s.def_atom(name="O", element="O", xyz=[0.0, 1.0, 0.0])
        
        frame = s.to_frame()
        assert isinstance(frame, Frame)
        
        # Check that atoms data is in the frame
        assert 'atoms' in frame._data
        atoms_ds = frame._data['atoms']
        
        # Check coordinates
        assert 'xyz' in atoms_ds.data_vars
        xyz_values = atoms_ds['xyz'].values
        assert xyz_values.shape == (3, 3)  # 3 atoms, 3 coordinates each
        np.testing.assert_array_equal(xyz_values[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(xyz_values[1], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(xyz_values[2], [0.0, 1.0, 0.0])
        
        # Check other properties
        assert 'name' in atoms_ds.data_vars
        names = atoms_ds['name'].values
        assert list(names) == ["C", "H", "O"]
        
        assert 'element' in atoms_ds.data_vars
        elements = atoms_ds['element'].values
        assert list(elements) == ["C", "H", "O"]
        
        # Check metadata
        assert frame._meta['structure_name'] == "test_structure"
    
    def test_to_frame_no_coordinates(self):
        """Test to_frame with atoms that have no coordinates."""
        from molpy.core.frame import Frame
        
        s = AtomicStructure()
        # Add atom without explicit coordinates
        a1 = s.def_atom(name="C", element="C")
        
        frame = s.to_frame()
        assert isinstance(frame, Frame)
        
        # Should get default coordinates [0, 0, 0]
        atoms_ds = frame._data['atoms']
        xyz_values = atoms_ds['xyz'].values
        np.testing.assert_array_equal(xyz_values[0], [0.0, 0.0, 0.0])
