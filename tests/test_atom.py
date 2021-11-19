# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

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
        particle = mp.Atom('particle', key='value', position=np.array([1., 2., 3]))
        assert particle.key == 'value'
        assert np.array_equal(particle.position, np.array([1., 2., 3]))
        
    def test_element(self, H2O):
        O, H1, H2 = H2O
        O.element = 'O'
        assert O.element.mass.magnitude == 15.99943
        
    def test_bond(self, H2O):
        O, H1, H2 = H2O
        assert O.bonds[H1] == H1.bonds[O]
        assert O.bonds[H2] == H2.bonds[O]
        
    def test_removeBond(self, H2O):
        O, H1, H2 = H2O
        O.removeBond(H1)
        assert len(O.bondedAtoms) == 1
        assert H1 not in O.bonds
            
    def test_copy(self, H2O):
        O, H1, H1 = H2O
        O.test = 'test'
        Onew = O.copy()
        assert Onew != O
        assert Onew.name == O.name
        assert Onew.element == O.element
        assert Onew._bondInfo == {}
        assert Onew.test == O.test
        assert Onew.uuid != O.uuid
        
class TestAtomGeometry:
    
    def test_move(self, particle):
        opos = np.array(particle.position) 
        vec = np.array([1, 2, 3])
        particle.move(vec)
        npos = particle.position
        assert np.array_equal(opos+vec, npos)