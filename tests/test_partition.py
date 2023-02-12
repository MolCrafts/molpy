# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-12
# version: 0.0.1

import molpy as mp
import numpy as np
import numpy.testing as npt

class TestCellList:

    def test_cellList_utils(self):

        cellList = mp.core.partition.CellList(np.array([]), np.array([3,3,3]))
        npt.assert_equal(cellList.get_shell_cell_indices(np.array([0, 0, 0])), np.array([[0, 0, 0], [0, 0, 2], [0, 2, 0], [0, 2, 2], [2, 0, 0], [2, 0, 2], [2, 2, 0], [2, 2, 2]]))

        npt.assert_equal(cellList.get_shell_cell_indices(np.array([2, 2, 2])), np.array([[2, 2, 2], [2, 2, 1], [2, 1, 2], [2, 1, 1], [1, 2, 2], [1, 2, 1], [1, 1, 2], [1, 1, 1]]))

    def test_calc_cell_dimensions(self):

        # create parallelpiped box
        box = np.array([[10, 3, 3], [0, 10, 3], [0, 0, 10]])
        r_cutoff = 3.0
        cell_lattice, cell_per_side = mp.core.partition.calc_cell_dimensions(box, r_cutoff)

        npt.assert_equal(cell_lattice, np.array([[10/3, 1, 1], [0, 10/3, 1], [0, 0, 10/3]]))
        npt.assert_equal(cell_per_side, np.array([3, 3, 3]))

        # create othogonal box
        box = np.diag([10, 10, 10])
        r_cutoff = 1.5
        cell_lattice, cell_per_side = mp.core.partition.calc_cell_dimensions(box, r_cutoff)

        npt.assert_equal(cell_per_side, np.array([6, 6, 6]))
        npt.assert_equal(cell_lattice, box / cell_per_side)


    def test_create_cellList(self):

        # create orthogonal box
        box = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        r_cutoff = 1.5
        xyz = np.array([[1, 1, 0], [2, 0, 0], [2, 2, 0]])
        cellList = mp.core.partition.create_cellList(box, xyz, r_cutoff)

        assert cellList

        # create parallelpiped box
        box = np.array([[10, 3, 3], [0, 10, 3], [0, 0, 10]])
        r_cutoff = 3.0
        xyz = np.array([[1, 1, 0], [3, 4, 0], [4, 1, 0]])
        cellList = mp.core.partition.create_cellList(box, xyz, r_cutoff)

        assert cellList

class TestNeighborList:

    def test_create_neighborList(self):

        box = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        r_cutoff = 3
        # create a set of xyz on the site
        # xyz = np.array([[0,0,0], [1, 0, 0], [2,0,0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [0, 2, 0], [1, 2, 0], [2, 2, 0]])
        xyz = np.array([[0, 0, 0], [1, 0, 0]])

        debugNblist = mp.core.partition.create_neighborList(box, xyz, r_cutoff, False)

        nblist = mp.core.partition.create_neighborList(box, xyz, r_cutoff, True)

        npt.assert_equal(debugNblist.indices, nblist.indices)
        