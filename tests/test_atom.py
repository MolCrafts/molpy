# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

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
    