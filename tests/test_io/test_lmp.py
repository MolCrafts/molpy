# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np
from molpy.factory import toLAMMPS
    
class TestLAMMPS:
    
    def test_write_ch4(self, CH4):
        
        system = mp.System('ch4')
        cell = mp.Cell(3, 'ppp', xlo=-2, xhi=2, ylo=-2, yhi=2, zlo=-1, zhi=1)
        system.cell = cell
        system.addMolecule(CH4)
        
        # toLAMMPS('test.lmp', system)
        