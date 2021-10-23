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
        
    def test_serialize(self, H2O):
        O, H1, H2 = H2O
        O.tensorx = np.zeros((3, 3))
        op = O.serialize()
        assert op['_name'] == 'O'
        assert op['_itemType'] == 'Atom'
        assert op['tensorx'] == np.zeros((3, 3)).tolist()
                
    def test_deserialize(self, H2O):
        O, H1, H2 = H2O
        op = O.serialize()
        oprime = Atom('').deserialize(op)
        assert oprime.name == 'O'
        assert oprime.itemType == 'Atom'
        assert (oprime.tensorx == np.zeros((3, 3))).all()
        
    
    def test_move(self, H2O):
        O, H1, H2 = H2O
        O.position = np.zeros((3, ))
        O.moveTo(np.array([1,2,3]))
        assert (O.position == np.array([1,2,3])).all()
        O.moveBy(np.array([1,1,1]))
        assert (O.position == np.array([2,3,4])).all()
        
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