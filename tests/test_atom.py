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
    
    def test_move(self, H2O):
        O, H1, H2 = H2O
        O.position = np.zeros((3, ))
        O.moveTo(np.array([1,2,3]))
        assert (O.position == np.array([1,2,3])).all()
        O.moveBy(np.array([1,1,1]))
        assert (O.position == np.array([2,3,4])).all()