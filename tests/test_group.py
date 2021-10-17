# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np

class TestGroup:
    
    @pytest.fixture(scope='class')
    def CH4(self):
        CH4 = mp.Group('CH4')
        C = mp.Atom('C')
        Hs = [mp.Atom(f'H{i}') for i in range(4)]
        CH4.add(C)
        for H in Hs:
            CH4.add(H)
        yield CH4
            
    def test_topoByCovalentMap(self, CH4):
        covalentMap = np.zeros((CH4.natoms, CH4.natoms), dtype=int)
        covalentMap[0, 1:] = covalentMap[1:, 0] = 1
        CH4.setTopoByCovalentMap(covalentMap)
        assert len(CH4['C'].bondedAtoms) == 4
        assert CH4['H0'].bondedAtoms[0] == CH4['C']