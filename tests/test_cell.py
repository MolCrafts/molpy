# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

import pytest
import molpy as mp

class TestCell:
    
    @pytest.fixture(scope='class')
    def cell3D(self):
        cell = mp.Cell(3, 'ppp', xlo=0, xhi=1, ylo=0, yhi=1, zlo=0, zhi=1)
        yield cell
        
    def test_cell3D(self, cell3D):
        assert cell3D.xlo == 0
        assert cell3D.x_boundary_condition == 'p'
        assert cell3D.lx == 1