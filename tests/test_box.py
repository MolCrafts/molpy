# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

import pytest
import molpy as mp

class Testbox:
    
    @pytest.fixture(scope='class')
    def box3D(self):
        box = mp.Box(3, 'ppp', xlo=0, xhi=1, ylo=0, yhi=1, zlo=0, zhi=1)
        yield box
        
    def test_box3D(self, box3D):
        assert box3D.xlo == 0
        assert box3D.x_boundary_condition == 'p'
        assert box3D.lx == 1