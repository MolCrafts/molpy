import freud
import numpy as np
import molpy as mp
import numpy.testing as npt
from molpy.core.neighborlist import NeighborList


class TestNeighborList:

    def test_init_cell(self):
        cutoff = 1.0
        nl = NeighborList(cutoff)
        assert nl.cutoff == cutoff
        _cell_shape, _cell_offset, _all_cell_coords = nl._init_cell(
            np.array(list(np.ndindex(3, 3, 3))), cutoff
        )
        npt.assert_equal(_cell_shape, np.array([3, 3, 3]))
        npt.assert_equal(_cell_offset, np.array([1, 3, 9]))
        npt.assert_equal(
            _all_cell_coords,
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 1, 2],
                    [0, 2, 0],
                    [0, 2, 1],
                    [0, 2, 2],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 0, 2],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 2],
                    [1, 2, 0],
                    [1, 2, 1],
                    [1, 2, 2],
                    [2, 0, 0],
                    [2, 0, 1],
                    [2, 0, 2],
                    [2, 1, 0],
                    [2, 1, 1],
                    [2, 1, 2],
                    [2, 2, 0],
                    [2, 2, 1],
                    [2, 2, 2],
                ]
            ),
        )

    def test_add_to_cell(self):
        cutoff = 1.0
        xyz = np.array(list(np.ndindex(3, 3, 3)))

        nl = NeighborList(cutoff)
        _cell_shape, cell_offset, _all_cell_coords = nl._init_cell(
            xyz, cutoff
        )
        _xyz_cell_coords, _xyz_cell_idx = nl._add_to_cell(xyz, cell_offset)
        assert len(_xyz_cell_coords) == len(xyz)
        assert len(_xyz_cell_idx) == len(xyz)

    def test_find_atoms_in_cell(self):

        cutoff = 1.2
        xyz = np.array(list(np.ndindex(3, 3, 3)))

        nl = NeighborList(cutoff)
        _cell_shape, cell_offset, _all_cell_coords = nl._init_cell(
            xyz, cutoff
        )
        _xyz_cell_coords, _xyz_cell_idx = nl._add_to_cell(
            xyz, cell_offset
        )
        center_xyz, center_id = nl._find_atoms_in_cell(
            xyz, _xyz_cell_idx, np.array([0, 0, 0])
        )
        assert len(center_xyz) == 1
        assert len(center_id) == 1

    def test_find_neighbor_cell(self):

        cutoff = 1.2
        xyz = np.array(list(np.ndindex(3, 3, 3)))

        nl = NeighborList(cutoff)
        _cell_shape, cell_offset, _all_cell_coords = nl._init_cell(
            xyz, cutoff
        )
        _xyz_cell_coords, _xyz_cell_idx = nl._add_to_cell(
            xyz, cell_offset
        )
        nbor_cell = nl._find_neighbor_cell(_cell_shape, np.array([0, 0, 0]))
        assert len(nbor_cell) == 14  # (3**3 - 1) / 2 + 1
