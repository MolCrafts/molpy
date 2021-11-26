# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np

class TestSystem:
    
    @pytest.fixture(scope='class')
    def system(self):
        
        system = mp.System('test')
        
        yield system
        
    def test_init_system(self, system, C6):
        assert C6.natoms == 12
        assert C6.nbonds == 12
        assert len(C6._bondList) == 12
        assert len(C6._bonds) == 12
        C6.searchAngles()
        C6.searchDihedrals()
        system.addMolecule(C6)
        assert system.natoms == 12
        assert system.nbonds == 12
        assert system.nangles == 18
        assert system.ndihedrals == 24


