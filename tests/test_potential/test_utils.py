# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-04
# version: 0.0.1

import jax.numpy as np
from molpy.potential.utils import make_pbc
import numpy.testing as npt

class TestUtils:

    def test_pbc(self):

        xyz = np.array([[1, 0, 0], [6, 0, 0]])
        box = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])

        shifted_xyz = make_pbc(xyz, box)
        npt.assert_allclose(shifted_xyz, np.array([[1, 0, 0], [-4, 0, 0]]))