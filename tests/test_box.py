# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

import numpy as np
import pytest
import molpy as mp

class TestBox:
    
    @pytest.fixture(scope='class')
    def orthogonalBox3D(self):
        box = mp.Box('ppp')
        box.defByEdge(0, 3, 0, 4, 0, 5)
        yield box
        
    def test_wrap(self, orthogonalBox3D):
        
        position = np.array([1, 5, -1])
        
        wrapped = orthogonalBox3D.wrap(position)
        
        assert np.array_equal(wrapped, np.array([1, 1, 4]))
        
        positions = np.array([[1, 5, -1],
                              [0, 0, 0],
                              [3, 4, 5],
                              [1.1, 5.1, -0.9]])
        
        wrapped = orthogonalBox3D.wrap(positions)

        assert np.allclose(wrapped, np.array([[1, 1, 4],
                                                 [0, 0, 0],
                                                 [0, 0, 0],
                                                 [1.1, 1.1, 4.1]]), )