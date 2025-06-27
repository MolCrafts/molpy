import pytest
import numpy as np
import molpy as mp

# --- 通用断言工具 ---
def assert_atoms_deepcopied(orig_atoms, copy_atoms):
    assert len(orig_atoms) == len(copy_atoms)
    for a, b in zip(orig_atoms, copy_atoms):
        assert a is not b
        assert a.get("name") == b.get("name")
        assert a.get("type") == b.get("type")
        assert a.get("q") == b.get("q")
        if "xyz" in a:
            assert np.allclose(a["xyz"], b["xyz"])

def assert_bonds_deepcopied(orig_bonds, copy_bonds, orig_atoms, copy_atoms):
    assert len(orig_bonds) == len(copy_bonds)
    for ob, cb in zip(orig_bonds, copy_bonds):
        assert ob is not cb
        assert cb.itom in copy_atoms and cb.jtom in copy_atoms
        assert ob.itom not in copy_atoms and ob.jtom not in copy_atoms
        assert {ob.itom.get("name"), ob.jtom.get("name")} == {cb.itom.get("name"), cb.jtom.get("name")}

def assert_angles_deepcopied(orig_angles, copy_angles, orig_atoms, copy_atoms):
    assert len(orig_angles) == len(copy_angles)
    for oa, ca in zip(orig_angles, copy_angles):
        assert oa is not ca
        assert ca.itom in copy_atoms and ca.jtom in copy_atoms and ca.ktom in copy_atoms
        assert oa.itom not in copy_atoms and oa.jtom not in copy_atoms and oa.ktom not in copy_atoms
        assert {oa.itom.get("name"), oa.jtom.get("name"), oa.ktom.get("name")} == \
               {ca.itom.get("name"), ca.jtom.get("name"), ca.ktom.get("name")}
        assert oa.jtom.get("name") == ca.jtom.get("name")

def assert_dihedrals_deepcopied(orig_dihedrals, copy_dihedrals, orig_atoms, copy_atoms):
    assert len(orig_dihedrals) == len(copy_dihedrals)
    for od, cd in zip(orig_dihedrals, copy_dihedrals):
        assert od is not cd
        for atom in [cd.itom, cd.jtom, cd.ktom, cd.ltom]:
            assert atom in copy_atoms
        for atom in [od.itom, od.jtom, od.ktom, od.ltom]:
            assert atom not in copy_atoms
        assert {od.itom.get("name"), od.jtom.get("name"), od.ktom.get("name"), od.ltom.get("name")} == \
               {cd.itom.get("name"), cd.jtom.get("name"), cd.ktom.get("name"), cd.ltom.get("name")}

class TestAtomicStructDeepCopy:
    """更正交、更全面的 AtomicStruct 深拷贝测试（不含 improper）"""

    def test_atom_bond_angle_dihedral_deepcopy(self):
        struct = mp.AtomicStruct(name="all_topo")
        a1 = struct.def_atom(name="A1", type="A", xyz=[0,0,0])
        a2 = struct.def_atom(name="A2", type="A", xyz=[1,0,0])
        a3 = struct.def_atom(name="A3", type="A", xyz=[0,1,0])
        a4 = struct.def_atom(name="A4", type="A", xyz=[0,0,1])
        struct.def_bond(a1, a2)
        struct.def_bond(a2, a3)
        struct.add_angle(mp.Angle(a1, a2, a3))
        struct.add_dihedral(mp.Dihedral(a1, a2, a3, a4))
        copy = struct()
        assert_atoms_deepcopied(struct.atoms, copy.atoms)
        assert_bonds_deepcopied(struct.bonds, copy.bonds, struct.atoms, copy.atoms)
        assert_angles_deepcopied(struct.angles, copy.angles, struct.atoms, copy.atoms)
        assert_dihedrals_deepcopied(struct.dihedrals, copy.dihedrals, struct.atoms, copy.atoms)

    def test_deepcopy_empty_and_no_topology(self):
        struct = mp.AtomicStruct(name="empty")
        copy = struct()
        assert len(copy.atoms) == 0
        assert len(copy.bonds) == 0
        assert len(copy.angles) == 0
        assert len(copy.dihedrals) == 0
        struct2 = mp.AtomicStruct(name="no_topo")
        struct2.def_atom(name="A", type="A", xyz=[0,0,0])
        struct2.def_atom(name="B", type="B", xyz=[1,0,0])
        copy2 = struct2()
        assert_atoms_deepcopied(struct2.atoms, copy2.atoms)
        assert len(copy2.bonds) == 0
        assert len(copy2.angles) == 0
        assert len(copy2.dihedrals) == 0

    def test_deepcopy_with_modifications(self):
        struct = mp.AtomicStruct(name="mod")
        o = struct.def_atom(name="O", type="O", q=-0.8, xyz=[0,0,0])
        h = struct.def_atom(name="H", type="H", q=0.4, xyz=[1,0,0])
        struct.def_bond(o, h)
        # 手动实现属性覆盖
        copy = struct()
        for atom in copy.atoms:
            atom["q"] = 0.0
        assert all(atom.get("q") == 0.0 for atom in copy.atoms)
        assert struct.atoms[0].get("q") == -0.8
        assert struct.atoms[1].get("q") == 0.4

    def test_deepcopy_independence(self):
        struct = mp.AtomicStruct(name="indep")
        o = struct.def_atom(name="O", type="O", q=-0.8, xyz=[0,0,0])
        h = struct.def_atom(name="H", type="H", q=0.4, xyz=[1,0,0])
        struct.def_bond(o, h)
        copy = struct()
        copy.atoms[0]["q"] = -1.0
        copy.atoms[0]["xyz"] = [10,10,10]
        copy.def_atom(name="N", type="N", q=-0.5)
        assert struct.atoms[0].get("q") == -0.8
        assert np.allclose(struct.atoms[0]["xyz"], [0,0,0])
        assert len(struct.atoms) == 2
        assert copy.atoms[0].get("q") == -1.0
        assert np.allclose(copy.atoms[0]["xyz"], [10,10,10])
        assert len(copy.atoms) == 3

    def test_deepcopy_preserves_custom_properties(self):
        struct = mp.AtomicStruct(name="custom")
        struct["custom_list"] = [1,2,3]
        struct["custom_dict"] = {"a":1, "b":2}
        struct["custom_value"] = 42
        import copy as pycopy
        copy_struct = pycopy.deepcopy(struct)
        assert copy_struct["custom_list"] == [1,2,3]
        assert copy_struct["custom_dict"] == {"a":1, "b":2}
        assert copy_struct["custom_value"] == 42
        copy_struct["custom_list"].append(4)
        copy_struct["custom_dict"]["c"] = 3
        copy_struct["custom_value"] = 100
        assert struct["custom_list"] == [1,2,3]
        assert struct["custom_dict"] == {"a":1, "b":2}
        assert struct["custom_value"] == 42

    def test_multiple_independent_copies(self):
        struct = mp.AtomicStruct(name="multi")
        o = struct.def_atom(name="O", xyz=[0,0,0])
        h = struct.def_atom(name="H", xyz=[1,0,0])
        struct.def_bond(o, h)
        copies = [struct(molid=i+1) for i in range(5)]
        for i, copy in enumerate(copies):
            assert len(copy.atoms) == 2
            assert len(copy.bonds) == 1
            unique_coord = [i+10, i+10, i+10]
            copy.atoms[0]["xyz"] = unique_coord
            for j, other_copy in enumerate(copies):
                if i != j:
                    assert not np.allclose(other_copy.atoms[0]["xyz"], unique_coord)

    def test_bond_angle_references_after_copy(self):
        struct = mp.AtomicStruct(name="ref")
        o = struct.def_atom(name="O", xyz=[0,0,0])
        h1 = struct.def_atom(name="H1", xyz=[1,0,0])
        h2 = struct.def_atom(name="H2", xyz=[0,1,0])
        struct.def_bond(o, h1)
        struct.def_bond(o, h2)
        struct.add_angle(mp.Angle(h1, o, h2))
        copy = struct()
        for bond in copy.bonds:
            assert bond.itom in copy.atoms
            assert bond.jtom in copy.atoms
            assert bond.itom not in struct.atoms
            assert bond.jtom not in struct.atoms
        for angle in copy.angles:
            assert angle.itom in copy.atoms
            assert angle.jtom in copy.atoms
            assert angle.ktom in copy.atoms

    def test_deepcopy_with_large_number_of_atoms(self):
        struct = mp.AtomicStruct(name="large")
        n = 100
        atoms = [struct.def_atom(name=f"A{i}", type="A", xyz=[i,0,0]) for i in range(n)]
        for i in range(n-1):
            struct.def_bond(atoms[i], atoms[i+1])
        copy = struct()
        assert_atoms_deepcopied(struct.atoms, copy.atoms)
        assert_bonds_deepcopied(struct.bonds, copy.bonds, struct.atoms, copy.atoms)
        # 修改 copy 不影响原始
        copy.atoms[0]["xyz"] = [999,999,999]
        assert not np.allclose(struct.atoms[0]["xyz"], [999,999,999])

    def test_deepcopy_preserves_order(self):
        struct = mp.AtomicStruct(name="order")
        names = [f"A{i}" for i in range(10)]
        atoms = [struct.def_atom(name=n, type="A", xyz=[i,0,0]) for i, n in enumerate(names)]
        copy = struct()
        for a, b in zip(struct.atoms, copy.atoms):
            assert a.get("name") == b.get("name")

    def test_deepcopy_with_no_atoms(self):
        struct = mp.AtomicStruct(name="noatom")
        copy = struct()
        assert len(copy.atoms) == 0
        assert len(copy.bonds) == 0
        assert len(copy.angles) == 0
        assert len(copy.dihedrals) == 0
