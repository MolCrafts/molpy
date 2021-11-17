# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

import pytest
import molpy as mp

class System:
    
    @pytest.fixture(scope='class')
    def system():
        
        system = mp.System('test')
        
        yield system
        
    def test_init_system(self, system, C6):
        system.addGroup(C6)
        C6.searchAngles()
        C6.searchDihedrals()
        assert system.natoms == 12
        assert system.nbonds == 12
        assert system.nangles
        assert system.ndihedrals