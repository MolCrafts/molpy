# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-10
# version: 0.0.1

from molpy.convert import toJAXMD
import numpy as np

class TestJAXMD:
    
    def test_toJAXMD(self, C6):
        
        c6 = toJAXMD(C6)
        self.c6 = c6
        
        assert np.array_equal(c6['positions'].shape, (12, 3))
        assert len(c6['atomTypes']) == 12
        assert len(c6['elements']) == 12
        assert len(c6['bonds']) == 12
        