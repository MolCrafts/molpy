import pytest
import numpy as np
import molpy as mp

from molpy.core import Atom, Bond, Angle, Dihedral

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
        assert {ob.itom.get("name"), ob.jtom.get("name")} == {
            cb.itom.get("name"),
            cb.jtom.get("name"),
        }


def assert_angles_deepcopied(orig_angles, copy_angles, orig_atoms, copy_atoms):
    assert len(orig_angles) == len(copy_angles)
    for oa, ca in zip(orig_angles, copy_angles):
        assert oa is not ca
        assert ca.itom in copy_atoms and ca.jtom in copy_atoms and ca.ktom in copy_atoms
        assert (
            oa.itom not in copy_atoms
            and oa.jtom not in copy_atoms
            and oa.ktom not in copy_atoms
        )
        assert {oa.itom.get("name"), oa.jtom.get("name"), oa.ktom.get("name")} == {
            ca.itom.get("name"),
            ca.jtom.get("name"),
            ca.ktom.get("name"),
        }
        assert oa.jtom.get("name") == ca.jtom.get("name")


def assert_dihedrals_deepcopied(orig_dihedrals, copy_dihedrals, orig_atoms, copy_atoms):
    assert len(orig_dihedrals) == len(copy_dihedrals)
    for od, cd in zip(orig_dihedrals, copy_dihedrals):
        assert od is not cd
        for atom in [cd.itom, cd.jtom, cd.ktom, cd.ltom]:
            assert atom in copy_atoms
        for atom in [od.itom, od.jtom, od.ktom, od.ltom]:
            assert atom not in copy_atoms
        assert {
            od.itom.get("name"),
            od.jtom.get("name"),
            od.ktom.get("name"),
            od.ltom.get("name"),
        } == {
            cd.itom.get("name"),
            cd.jtom.get("name"),
            cd.ktom.get("name"),
            cd.ltom.get("name"),
        }


class TestAtomisticDeepCopy:
    """更正交、更全面的 Atomistic 深拷贝测试（不含 improper）"""

    def test_atom_bond_angle_dihedral_deepcopy(self):
        struct = mp.Atomistic(name="all_topo")
        a1 = struct.def_atom(name="A1", type="A", xyz=[0, 0, 0])
        a2 = struct.def_atom(name="A2", type="A", xyz=[1, 0, 0])
        a3 = struct.def_atom(name="A3", type="A", xyz=[0, 1, 0])
        a4 = struct.def_atom(name="A4", type="A", xyz=[0, 0, 1])
        struct.def_bond(a1, a2)
        struct.def_bond(a2, a3)
        struct.add_angle(mp.Angle(a1, a2, a3))
        struct.add_dihedral(mp.Dihedral(a1, a2, a3, a4))
        copy = struct()
        assert_atoms_deepcopied(struct.atoms, copy.atoms)
        assert_bonds_deepcopied(struct.bonds, copy.bonds, struct.atoms, copy.atoms)
        assert_angles_deepcopied(struct.angles, copy.angles, struct.atoms, copy.atoms)
        assert_dihedrals_deepcopied(
            struct.dihedrals, copy.dihedrals, struct.atoms, copy.atoms
        )

    def test_deepcopy_empty_and_no_topology(self):
        struct = mp.Atomistic(name="empty")
        copy = struct()
        assert len(copy.atoms) == 0
        assert len(copy.bonds) == 0
        assert len(copy.angles) == 0
        assert len(copy.dihedrals) == 0
        struct2 = mp.Atomistic(name="no_topo")
        struct2.def_atom(name="A", type="A", xyz=[0, 0, 0])
        struct2.def_atom(name="B", type="B", xyz=[1, 0, 0])
        copy2 = struct2()
        assert_atoms_deepcopied(struct2.atoms, copy2.atoms)
        assert len(copy2.bonds) == 0
        assert len(copy2.angles) == 0
        assert len(copy2.dihedrals) == 0

    def test_deepcopy_with_modifications(self):
        struct = mp.Atomistic(name="mod")
        o = struct.def_atom(name="O", type="O", q=-0.8, xyz=[0, 0, 0])
        h = struct.def_atom(name="H", type="H", q=0.4, xyz=[1, 0, 0])
        struct.def_bond(o, h)
        # 手动实现属性覆盖
        copy = struct()
        for atom in copy.atoms:
            atom["q"] = 0.0
        assert all(atom.get("q") == 0.0 for atom in copy.atoms)
        assert struct.atoms[0].get("q") == -0.8
        assert struct.atoms[1].get("q") == 0.4

    def test_deepcopy_independence(self):
        struct = mp.Atomistic(name="indep")
        o = struct.def_atom(name="O", type="O", q=-0.8, xyz=[0, 0, 0])
        h = struct.def_atom(name="H", type="H", q=0.4, xyz=[1, 0, 0])
        struct.def_bond(o, h)
        copy = struct()
        copy.atoms[0]["q"] = -1.0
        copy.atoms[0]["xyz"] = [10, 10, 10]
        copy.def_atom(name="N", type="N", q=-0.5)
        assert struct.atoms[0].get("q") == -0.8
        assert np.allclose(struct.atoms[0]["xyz"], [0, 0, 0])
        assert len(struct.atoms) == 2
        assert copy.atoms[0].get("q") == -1.0
        assert np.allclose(copy.atoms[0]["xyz"], [10, 10, 10])
        assert len(copy.atoms) == 3

    def test_deepcopy_preserves_custom_properties(self):
        struct = mp.Atomistic(name="custom")
        struct["custom_list"] = [1, 2, 3]
        struct["custom_dict"] = {"a": 1, "b": 2}
        struct["custom_value"] = 42
        import copy as pycopy

        copy_struct = pycopy.deepcopy(struct)
        assert copy_struct["custom_list"] == [1, 2, 3]
        assert copy_struct["custom_dict"] == {"a": 1, "b": 2}
        assert copy_struct["custom_value"] == 42
        copy_struct["custom_list"].append(4)
        copy_struct["custom_dict"]["c"] = 3
        copy_struct["custom_value"] = 100
        assert struct["custom_list"] == [1, 2, 3]
        assert struct["custom_dict"] == {"a": 1, "b": 2}
        assert struct["custom_value"] == 42

    def test_multiple_independent_copies(self):
        struct = mp.Atomistic(name="multi")
        o = struct.def_atom(name="O", xyz=[0, 0, 0])
        h = struct.def_atom(name="H", xyz=[1, 0, 0])
        struct.def_bond(o, h)
        copies = [struct(molid=i + 1) for i in range(5)]
        for i, copy in enumerate(copies):
            assert len(copy.atoms) == 2
            assert len(copy.bonds) == 1
            unique_coord = [i + 10, i + 10, i + 10]
            copy.atoms[0]["xyz"] = unique_coord
            for j, other_copy in enumerate(copies):
                if i != j:
                    assert not np.allclose(other_copy.atoms[0]["xyz"], unique_coord)

    def test_bond_angle_references_after_copy(self):
        struct = mp.Atomistic(name="ref")
        o = struct.def_atom(name="O", xyz=[0, 0, 0])
        h1 = struct.def_atom(name="H1", xyz=[1, 0, 0])
        h2 = struct.def_atom(name="H2", xyz=[0, 1, 0])
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
        struct = mp.Atomistic(name="large")
        n = 100
        atoms = [
            struct.def_atom(name=f"A{i}", type="A", xyz=[i, 0, 0]) for i in range(n)
        ]
        for i in range(n - 1):
            struct.def_bond(atoms[i], atoms[i + 1])
        copy = struct()
        assert_atoms_deepcopied(struct.atoms, copy.atoms)
        assert_bonds_deepcopied(struct.bonds, copy.bonds, struct.atoms, copy.atoms)
        # 修改 copy 不影响原始
        copy.atoms[0]["xyz"] = [999, 999, 999]
        assert not np.allclose(struct.atoms[0]["xyz"], [999, 999, 999])

    def test_deepcopy_preserves_order(self):
        struct = mp.Atomistic(name="order")
        names = [f"A{i}" for i in range(10)]
        atoms = [
            struct.def_atom(name=n, type="A", xyz=[i, 0, 0])
            for i, n in enumerate(names)
        ]
        copy = struct()
        for a, b in zip(struct.atoms, copy.atoms):
            assert a.get("name") == b.get("name")

    def test_deepcopy_with_no_atoms(self):
        struct = mp.Atomistic(name="noatom")
        copy = struct()
        assert len(copy.atoms) == 0
        assert len(copy.bonds) == 0
        assert len(copy.angles) == 0
        assert len(copy.dihedrals) == 0


class TestAtomisticSerialization:
    """Unit tests for Atomistic.to_frame and from_frame (Frame/Block API)."""

    @pytest.fixture(scope="class", name="simple_atomistic")
    def make_simple_atomistic(self):
        """Create a simple Atomistic structure for roundtrip testing."""
        atoms = [
            mp.Atom(name="H", xyz=[0.0, 0.0, 0.0]),
            mp.Atom(name="O", xyz=[1.0, 0.0, 0.0]),
            mp.Atom(name="H", xyz=[0.0, 1.0, 0.0]),
            mp.Atom(name="C", xyz=[0.0, 0.0, 1.0]),
        ]
        struct = mp.Atomistic(name="water")
        struct.add_atoms(atoms)
        struct.add_bond(mp.Bond(atoms[0], atoms[1]))
        struct.add_bond(mp.Bond(atoms[1], atoms[2]))
        struct.add_angle(mp.Angle(atoms[0], atoms[1], atoms[2]))
        struct.add_dihedral(mp.Dihedral(atoms[0], atoms[1], atoms[2], atoms[3]))
        return struct

    def test_to_frame_and_from_frame_roundtrip(self, simple_atomistic):
        """Test Atomistic.to_frame and from_frame roundtrip with all entity types."""
        struct = simple_atomistic
        frame = struct.to_frame()
        struct2 = mp.Atomistic.from_frame(frame)

        # Check atom count and names
        assert len(struct2.atoms) == 4
        assert [a["name"] for a in struct2.atoms] == [a["name"] for a in struct.atoms]
        # Check coordinates
        np.testing.assert_allclose(
            [a["xyz"] for a in struct2.atoms], [a["xyz"] for a in struct.atoms]
        )
        # Check bonds
        assert len(struct2.bonds) == 2
        # Check angles
        assert len(struct2.angles) == 1
        # Check dihedrals
        assert len(struct2.dihedrals) == 1
        # Check metadata
        assert frame.metadata["name"] == "water"

    def test_to_frame(self):
        """Test Atomistic.to_frame produces correct Frame/Block structure and metadata."""
        atoms = [
            mp.Atom(name="H", xyz=[0.0, 0.0, 0.0]),
            mp.Atom(name="O", xyz=[1.0, 0.0, 0.0]),
            mp.Atom(name="H", xyz=[0.0, 1.0, 0.0]),
            mp.Atom(name="C", xyz=[0.0, 0.0, 1.0]),
        ]
        struct = mp.Atomistic(name="water")
        struct.add_atoms(atoms)
        struct.add_bond(mp.Bond(atoms[0], atoms[1]))
        struct.add_bond(mp.Bond(atoms[1], atoms[2]))
        struct.add_angle(mp.Angle(atoms[0], atoms[1], atoms[2]))
        struct.add_dihedral(mp.Dihedral(atoms[0], atoms[1], atoms[2], atoms[3]))
        # struct.impropers = [mp.Improper(atoms[1], atoms[0], atoms[2], atoms[3])]
        frame = struct.to_frame()
        # Check atom block
        assert "atoms" in frame
        assert frame["atoms"].nrows == 4
        assert set(frame["atoms"].keys()) >= {"name", "xyz"}
        # Check bond/angle/dihedral/improper blocks
        assert "bonds" in frame and frame["bonds"].nrows == 2
        assert "angles" in frame and frame["angles"].nrows == 1
        assert "dihedrals" in frame and frame["dihedrals"].nrows == 1
        # assert "impropers" in frame and frame["impropers"].nrows == 1
        # Check metadata
        assert frame.metadata["name"] == "water"
        assert frame.metadata["n_atoms"] == 4
        assert frame.metadata["n_bonds"] == 2
        assert frame.metadata["n_angles"] == 1
        assert frame.metadata["n_dihedrals"] == 1

    def test_from_frame(self):
        """Test Atomistic.from_frame reconstructs structure from Frame/Block."""
        from molpy.core.frame import Frame, Block

        # Build a minimal frame for water
        atoms_block = Block(
            {
                "name": ["H", "O", "H", "C"],
                "xyz": np.array(
                    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
                ),
            }
        )
        bonds_block = Block({"i": [0, 1], "j": [1, 2]})
        angles_block = Block({"i": [0], "j": [1], "k": [2]})
        dihedrals_block = Block({"i": [0], "j": [1], "k": [2], "l": [3]})
        frame = Frame()
        frame["atoms"] = atoms_block
        frame["bonds"] = bonds_block
        frame["angles"] = angles_block
        frame["dihedrals"] = dihedrals_block
        frame.metadata["name"] = "water"
        struct = mp.Atomistic.from_frame(frame)
        # Check atom count and names
        assert len(struct.atoms) == 4
        assert [a["name"] for a in struct.atoms] == ["H", "O", "H", "C"]
        # Check coordinates
        np.testing.assert_allclose(
            [a["xyz"] for a in struct.atoms],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
        # Check bonds
        assert len(struct.bonds) == 2
        # Check angles
        assert len(struct.angles) == 1
        # Check dihedrals
        assert len(struct.dihedrals) == 1
        # Check metadata
        assert frame.metadata["name"] == "water"
