# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-12
# version: 0.0.1

import numpy as np
import molpy as mp
import numpy.testing as npt

class TestSimpleRandomWalk:

    def test_linear(self):

        system = mp.System()
        system.set_box(10, 10, 10)

        rw = mp.modeller.SimpleRandomWalk(system.box)
        linear_chain1 = rw.linear(10, 1, start_point=np.array([0, 0, 0]))

        assert linear_chain1.shape == (10, 3)
