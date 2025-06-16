"""
Complete test suite for molpy.core.struct module.

This test suite provides comprehensive coverage for all classes and methods
in the struct module, including edge cases and error conditions.
"""

import pytest
import numpy as np
from molpy.core.struct import Struct, AtomicStructure, MolecularStructure, Atom, Bond, Angle, Dihedral, Improper, Entities

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
        for orig, clone in zip(angle.atoms, angle2.atoms):
            assert np.allclose(orig.xyz, clone.xyz)
            assert orig is not clone

class TestDihedral:
    def test_init_basic(self):
        a1 = Atom(xyz=[0,0,0])
        a2 = Atom(xyz=[1,0,0])
        a3 = Atom(xyz=[2,0,0])
        a4 = Atom(xyz=[2,1,0])
        dih = Dihedral(a1, a2, a3, a4)
        assert dih.atom1 is not None
        assert dih.atom2 is not None
        assert dih.atom3 is not None
        assert dih.atom4 is not None
    def test_value(self):
        # 90度二面角
        a1 = Atom(xyz=[0,0,0])
        a2 = Atom(xyz=[1,0,0])
        a3 = Atom(xyz=[1,1,0])
        a4 = Atom(xyz=[1,1,1])
        dih = Dihedral(a1, a2, a3, a4)
        assert abs(abs(np.degrees(dih.value)) - 90) < 1e-6
    def test_clone(self):
        a1 = Atom(xyz=[0,0,0])
        a2 = Atom(xyz=[1,0,0])
        a3 = Atom(xyz=[2,0,0])
        a4 = Atom(xyz=[2,1,0])
        dih = Dihedral(a1, a2, a3, a4)
        dih2 = dih.clone()
        assert isinstance(dih2, Dihedral)
        assert len(dih2.atoms) == 4
        for orig, clone in zip(dih.atoms, dih2.atoms):
            assert np.allclose(orig.xyz, clone.xyz)
            assert orig is not clone

class TestImproper:
    def test_init_and_repr(self):
        c = Atom(xyz=[0,0,0])
        a1 = Atom(xyz=[1,0,0])
        a2 = Atom(xyz=[0,1,0])
        a3 = Atom(xyz=[0,0,1])
        imp = Improper(c, a1, a2, a3)
        assert imp.atom1 is c
        assert imp.atom2 in [a1, a2, a3]
        assert "Improper" in repr(imp)

class TestEntities:
    def test_add_remove(self):
        e = Entities()
        a = Atom(name="C")
        e.add(a)
        assert len(e) == 1
        e.remove(a)
        assert len(e) == 0
    def test_indexing(self):
        atoms = [Atom(name=f"C{i}") for i in range(3)]
        e = Entities(atoms)
        assert e[0] is atoms[0]
        assert e["C1"] is atoms[1]
        assert e[0, "C2"] == [atoms[0], atoms[2]]
    def test_get_by(self):
        atoms = [Atom(name="C"), Atom(name="H")]
        e = Entities(atoms)
        found = e.get_by(lambda x: x.name == "H")
        assert found.name == "H"

class TestStruct:
    def test_init_basic(self):
        struct = Struct(name="test_struct")
        assert struct["name"] == "test_struct"
        assert repr(struct) == "<Struct: test_struct>"
        unnamed = Struct()
        assert repr(unnamed) == "<Struct: >"
    def test_clone(self):
        struct = Struct(name="foo")
        struct2 = struct.clone(name="bar")
        assert struct2["name"] == "bar"
        assert struct["name"] == "foo"
        assert struct2 is not struct

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

class TestMolecularStructure:
    def test_repr(self):
        mol = MolecularStructure()
        mol.def_atom(name="C1")
        mol.def_bond(mol.def_atom(name="C2"), 0)
        assert "MolecularStructure:" in repr(mol)
        assert "2 atoms" in repr(mol)
        assert "1 bonds" in repr(mol)

