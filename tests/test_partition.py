# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-12
# version: 0.0.1

import molpy as mp
import numpy as np
import numpy.testing as npt

class TestPartition:

    def test_calc_cell_dimensions(self):

        # create parallelpiped box
        box = np.array([[10, 3, 3], [0, 10, 3], [0, 0, 10]])
        r_cutoff = 3.0
        cell_per_side = mp.core.partition.calc_cell_dimensions(box, r_cutoff)

        npt.assert_equal(cell_per_side, np.array([3, 3, 3]))
        
    def test_create_cellList(self):

        # create orthogonal box
        box = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        r_cutoff = 3.0
        xyz = np.array([[1, 1, 0], [3, 4, 0], [4, 3, 0]])
        cellList = mp.core.partition.create_cellList(box, xyz, r_cutoff)

        # create parallelpiped box
        box = np.array([[10, 3, 3], [0, 10, 3], [0, 0, 10]])
        r_cutoff = 3.0
        xyz = np.array([[1, 1, 0], [3, 4, 0], [4, 1, 0]])
        cellList = mp.core.partition.create_cellList(box, xyz, r_cutoff)

        assert cellList