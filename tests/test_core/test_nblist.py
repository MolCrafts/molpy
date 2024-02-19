import numpy as np
import molpy as mp
import numpy.testing as npt
from molpy.core.neighborlist import NeighborList

class TestNeighborList:

    def test_calc_cell_idx(self):

        nblist = NeighborList(1.0)
        xyz = np.array([
            [0.0, 0.0, 0.0], 
            [1.5, 1.5, 1.5]
        ])
        # npt.assert_allclose(nblist._calc_cell_idx(xyz, 1.0), np.array([[0, 0, 0], [1, 1, 1]]) )
        npt.assert_allclose(nblist._calc_cell_idx(xyz, 1.0), np.array([0, 7]))

    def test_build(self):

        nblist = NeighborList(1.0)
        xyz = np.array([
            [0.0, 0.0, 0.0], 
            [1.5, 1.5, 1.5]
        ])
        box = mp.Box.cube(2.0)
        pairs = nblist.build(xyz, box)
        print(pairs)
        assert False