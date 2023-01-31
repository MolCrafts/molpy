# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-31
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np
import numpy.testing as npt

class TestBox:

    def test_wrap_single_particle(self):
        box = mp.Box(2, 2, 2, 0, 0, 0, 0, 0, 0)

        points = [0, -1, -1]

        npt.assert_allclose(box.wrap(points), np.array([0, 1, 1]), rtol=1e-6)

        with pytest.raises(ValueError):
            box.wrap([1, 2])

        triclinic_box = mp.Box(2, 2, 2, 0, 0, 0, 1, 0, 0)
        npt.assert_allclose(triclinic_box.wrap(points), np.array([1, 1, 1]), rtol=1e-6)

    def test_wrap_multiple_particles(self):
        box = mp.Box(1, 1, 1, 0, 0, 0, 1, 1, 1)

        points = [
             [0,  0,  0],
             [0,  0,  1],
             [0,  1,  0], 
             [0,  1,  1],
             [1,  0,  0],
             [1,  0,  1],
             [1,  1,  0],
             [1,  1,  1],            
        ]
        npt.assert_allclose(box.wrap(points), np.zeros((8, 3)), rtol=1e-6)



