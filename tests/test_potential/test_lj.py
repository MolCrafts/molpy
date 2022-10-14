# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-04
# version: 0.0.1

import jax.numpy as np
from molpy.potential.lj import LJCut
from molpy.core.forcefield import Params
import pytest

class TestLennardJones:

    def test_lj126(self):
        
        pytest.importorskip("jax")

        xyz = np.array([
            [0.000,   1.000,   0.000],
            [1.000,   0.000,   1.000],
            [2.700,   1.600,   0.200],
            [3.700,   0.600,   1.200],
        ])

        # connect 1 2
        # connect 3 4

        pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])

        epsilon = np.array([1] * len(pairs))
        sigma = np.array([0.2] * len(pairs))
        params = Params(epsilon=epsilon, sigma=sigma)

        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

        lj126 = LJCut(r_cutoff=4.0, is_pbc=True)

        E = lj126.energy(xyz, pairs, box, params)
        F = lj126.force(xyz, pairs, box, params)