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
        O = mp.Atom('O')
        H1 = Atom('H1')
        H2 = Atom('H2')
        
        O.bondto(H1)
        O.bondto(H2)
        yield O, H1, H2
        
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
        with pytest.raises(KeyError):
            assert O.bonds[H1]
            
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