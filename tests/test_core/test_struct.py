import pytest
import molpy as mp

class TestAtom:

    @pytest.fixture
    def atom(self):
        return mp.Atom(name='C')

    def test_copy(self, atom):
        atom_copy = atom.copy()
        assert atom_copy == atom
        assert atom_copy is not atom
        atom_copy_again = atom_copy.copy()
        assert atom_copy_again == atom
        assert atom_copy_again is not atom

class TestBond:

    @pytest.fixture
    def bond(self):
        return mp.Bond(
            itom=mp.Atom(name='C'),
            jtom=mp.Atom(name='H'),
            order=1
        )
    
    def test_copy(self, bond):
        bond_copy = bond.copy()
        assert bond_copy == bond
        assert bond_copy is not bond
        bond_copy_again = bond_copy.copy()
        assert bond_copy_again == bond
        assert bond_copy_again is not bond

class TestStruct:

    @pytest.fixture
    def struct(self):
        atoms = [mp.Atom(name='C'), mp.Atom(name='H')]
        struct = mp.Struct(["atoms", "bonds"])
        struct['atoms'] = atoms
        struct['bonds'] = [mp.Bond(itom=atoms[0], jtom=atoms[1], order=1)]
        return struct
    
    def test_copy(self, struct):
        struct_copy = struct.copy()
        # assert struct_copy == struct
        assert struct_copy['bonds'][0].itom is struct_copy['atoms'][0]
        assert struct_copy['bonds'][0].jtom is struct_copy['atoms'][1]