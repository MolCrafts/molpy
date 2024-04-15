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
        npt.assert_equal(_cell_shape, np.array([2, 2, 2]))
        npt.assert_equal(_cell_offset, np.array([4, 2, 1]))
        npt.assert_equal(
            _all_cell_coords,
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
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
        assert len(center_xyz) == 8
        assert len(center_id) == 8

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
        nbor_cell = nl._find_neighbor_cell(_cell_shape, np.array([2, 2, 2]))
        assert len(nbor_cell) == 13  # (3**3 - 1) / 2 
        assert np.all(nbor_cell <= 2)
        assert np.all(nbor_cell >= 0)

    def test_find_neighbor_cell_2(self):

        cutoff = 2.0
        xyz = np.array([
            [0, 0, 9],
            [0, 0, 2],
            [0, 0, 4],
            [0, 0, 0],
        ])
        box = mp.Box.cube(10)
        nl = NeighborList(cutoff)
        _cell_shape, cell_offset, _all_cell_coords = nl._init_cell(
            xyz, cutoff
        )
        _xyz_cell_coords, _xyz_cell_idx = nl._add_to_cell(
            xyz, cell_offset
        )
        nbor_cell = nl._find_neighbor_cell(_cell_shape, np.array([0, 0, 0]))
        assert len(nbor_cell) == 13  # (3**3 - 1) / 2 + 1
        assert np.all(nbor_cell <= 2)
        assert np.all(nbor_cell >= 0)

    # def test_find_all_pairs(self, test_data_path):

    #     frame = mp.io.load_frame(test_data_path / "lammps-data/molid.lmp", format="LAMMPS Data")
    #     nl = NeighborList(5.0)
    #     nl.build(frame.positions)
    #     nl.update(frame.positions, frame.box)
    #     pairs = nl.find_all_pairs(frame.box)
    #     assert len(pairs) == 0
        
    def test_find_debug(self):

        cutoff = 2.0
        xyz = np.array([
            [0, 0, 9],
            [0, 0, 2],
            [0, 0, 4],
            [0, 0, 0],
        ])
        box = mp.Box.cube(10)
        nl = NeighborList(cutoff)
        nl.build(xyz)
        pairs = nl.find_all_pairs(box)
        assert len(pairs) == 3
        
    def test_find_random_pair(self):

        n_atoms = 10
        cutoff = 2
        xyz = np.random.rand(n_atoms, 3) * 10
        box = mp.Box(10, 10, 10)
        rij = box.self_diff(xyz)
        dij = np.linalg.norm(rij, axis=1)
        mask = dij < cutoff
        less_cutoff = np.where(mask)[0]

        nl = NeighborList(cutoff)
        nl.build(xyz)
        pairs = nl.find_all_pairs(box)
        assert len(pairs) == len(less_cutoff)
