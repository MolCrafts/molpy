# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from copy import deepcopy
import numpy as np
import pytest
import molpy as mp
from molpy.atom import Atom

class TestAtom:
    
    @pytest.fixture(scope='class')
    def H2O(self):
        O = mp.Atom('O', key='value')
        H1 = Atom('H1')
        H2 = Atom('H2')
        assert O.key == 'value'
        assert O.properties['key'] == 'value'
        O.bondto(H1)
        O.bondto(H2)
        yield O, H1, H2
        
    def test_init(self):
        atomType = mp.AtomType('test', key2='value')
        particle = mp.Atom('particle', key='value', position=np.array([1., 2., 3]), atomType=atomType)
        assert particle.atomType
        assert particle.key == 'value'
        assert np.array_equal(particle.position, np.array([1., 2., 3]))
        assert particle.key2 == 'value'
        with pytest.raises(AttributeError):
            assert particle.key3
        
    def test_element(self, H2O):
        O, H1, H2 = H2O
        O.element = 'O'
        assert O.element.mass.magnitude == 15.99943
        assert O.getSymbol() == 'O'
        with pytest.raises(KeyError):
            O.element = 'UNK_ElEMENT'
        
    def test_bond(self, H2O):
        O, H1, H2 = H2O
        assert O.getBond(H1).atom == O
        assert O.getBond(H1) == H1.getBond(O)
        assert O.getBond(H2) == H2.getBond(O)
        
    def test_removeBond(self, H2O):
        O, H1, H2 = H2O
        O.removeBond(H1)
        assert len(O.bondedAtoms) == 1
        assert H1 not in O.bondedAtoms
            
    def test_copy(self, particle):
        
        p = particle()
        assert p.position is not None
        assert p._position is not None
        assert p.uuid != particle.uuid
        assert p.name == particle.name
        
class TestAtomGeometry:
    
    def test_move(self, particle):
        vec = np.arange(3)
        p0 = particle
        p1 = particle()
        p0.move(vec)
        print(id(p1.position), id(p0.position))
        assert not np.array_equal(p1.position, p0.position)
        p2 = particle()
            
class TestAtomCopy:
    
    def test_atom_copy(self):
        
        a = mp.Atom('a')
        a = deepcopy(a)
        assert a.name == 'a'

    def test_atomType_copy(self):
        
        a = mp.Atom('a')
        at = mp.AtomType('A', key='value')
        a.atomType = at
        acopy = deepcopy(a)
        assert acopy.atomType.key == 'value'
        assert acopy.atomType.uuid == at.uuid
        assert id(acopy.atomType) == id(at)
        
    def test_bond_copy(self):
        
        a = mp.Atom('a')
        b = mp.Atom('b')
        bond = a.bondto(b)
        acopy = deepcopy(a)
        print(a.bonds)
        assert id(acopy.bondedAtoms[0] != b)
        assert id(acopy.bonds[0].atom) == id(acopy)
        assert id(acopy.bonds[0]) != id(bond)
        
    def test_position_copy(self):
        
        a = mp.Atom('a')
        a.position = [1, 2, 3]
        acopy = deepcopy(a)
        assert np.array_equal(acopy.position, a.position)
        acopy.position = [2,3,4]
        assert not np.array_equal(acopy.position, a.position)