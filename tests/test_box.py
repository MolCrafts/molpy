# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

import numpy as np
import pytest
import molpy as mp
from molpy.box import NeighborList


@pytest.fixture(scope="module")
def orthogonalBox3D():
    box = mp.Box("ppp")
    box.defByEdge(30, 40, 35)
    yield box


class TestBox:
    def test_init(self):
        """test for FCC lattice"""
        lbox = 2.757716
        degree = 60
        box0 = mp.Box(
            "ppp", lz=lbox, ly=lbox, lx=lbox, alpha=degree, beta=degree, gamma=degree
        )
        box1 = mp.Box(
            "ppp",
            zhi=box0.zhi,
            yhi=box0.yhi,
            xhi=box0.xhi,
            xy=box0.xy,
            xz=box0.xz,
            yz=box0.yz,
        )
        box2 = mp.Box("ppp", a1=box1.x_vec, a2=box1.y_vec, a3=box1.z_vec)

        assert np.all(np.isclose(box0.lengths(), lbox))
        assert np.all(np.isclose(box0.angles(), degree))
        assert np.all(np.isclose(box0.basis, box1.basis))
        assert np.all(np.isclose(box0.basis, box2.basis))
        xyz = np.asarray(
            [[0.55922094, 0.75597713, 0.95963365], [0.0084922, 0.55692665, 0.5926623]]
        )
        xyz_scaled = xyz @ box0.basis
        output_xyz_scaled = np.asarray(
            [[3.90775648, 2.56941297, 2.16077404], [1.60853919, 1.80189021, 1.3344773]]
        )
        assert np.all(np.isclose(xyz_scaled, output_xyz_scaled))

    def test_wrap(self, orthogonalBox3D):

        position = np.array([10, 50, -10])

        wrapped = orthogonalBox3D.wrap(position)

        assert np.array_equal(wrapped, np.array([10, 10, 25]))

        positions = np.array(
            [[10, 50, -10], [0, 0, 0], [30, 40, 35], [1.1, 40.1, -0.9]]
        )

        wrapped = orthogonalBox3D.wrap(positions)

        assert np.allclose(
            wrapped,
            np.array([[10, 10, 25], [0, 0, 0], [0, 0, 0], [1.1, 0.1, 34.1]]),
        )


@pytest.fixture(scope="module")
def orthogonalCell3D(orthogonalBox3D):
    cell = mp.CellList(orthogonalBox3D, 4)
    yield cell


class TestCellList:
    def test_build(self, orthogonalCell3D):

        orthogonalCell3D.build()
        assert np.array_equal(orthogonalCell3D.ncell, np.prod([7, 10, 8]))
        assert np.allclose(
            orthogonalCell3D.cell_size, np.diag([4.285, 4, 4.375]), atol=1e-2
        )

    def test_update(self, orthogonalCell3D):

        positions = np.array([[3, 4, 5], [13, 14, 11], [23, 24, 21]])

        g = mp.full("test", [f"{pos}" for pos in positions], position=positions)

        orthogonalCell3D.update(g.atoms)
        assert orthogonalCell3D.cell_list[0, 1, 1]
        assert orthogonalCell3D.cell_list[3, 3, 2]
        assert orthogonalCell3D.cell_list[5, 6, 4]

    def test_getAdjCell(self, orthogonalCell3D):

        adj_cells = orthogonalCell3D.getAdjCell(np.array([3, 3, 2]))

        assert adj_cells.shape == (26,)
        assert isinstance(adj_cells[0], list)

        adj_atoms = []
        adj_atoms.extend(adj_cells)

    def test_cell3D_lxyz(self):
        box = mp.Box("ppp", lx=1.0, ly=3.0, lz=4.0)
        assert box.lx == 1
        assert box.volume == pytest.approx(12, 1e-8)


@pytest.fixture(scope="module")
def orthogonalNBL3D(orthogonalCell3D):
    orthogonalCell3D.reset()  # erase atoms
    yield NeighborList(orthogonalCell3D)


import time


class TestNeighborList:
    def test_update(self, orthogonalNBL3D):

        #

        box = orthogonalNBL3D.box
        n_atoms = 200  # 10000
        scale_pos = np.random.random((n_atoms, 3))
        positions = scale_pos @ orthogonalNBL3D.box.basis
        g = mp.full("test", [f"{pos}" for pos in positions], position=positions)
        # start = time.time()
        orthogonalNBL3D.update(g.atoms)
        # end = time.time()

        # flags
        Wrong_interacting_pairs = 0
        Missing_interacting_pairs = 0
        Duplication_in_neighbour_list = 0
        Distances_mismatch = 0

        nbl = orthogonalNBL3D.neighbor_list  # atom: {
        for catom, neiInfo in nbl.items():  #  'neighbors': nei_atoms,
            patoms = neiInfo["neighbors"]  #  'dr': dr,
            for patom in patoms:  #  'distance': distance
                dr = box.distance(catom, patom)  #        }
                distance = np.linalg.norm(dr)
                if distance > orthogonalNBL3D.rcutoff:
                    Wrong_interacting_pairs += 1
                    print("ERROR: Wrong interacting pair:", catom, patom, distance)
                    continue
                if distance <= orthogonalNBL3D.rcutoff:
                    index = np.where(patoms == catom)
                    if len(index) == 0:
                        Missing_interacting_pairs += 1
                        print(
                            "ERROR: Missing interacting pair:", catom, patom, distance
                        )
                    elif len(index) > 1:
                        print(
                            "ERROR: Duplication in neighbour list:",
                            catom,
                            patom,
                            distance,
                        )
                        Duplication_in_neighbour_list += 1
                    else:
                        if np.allclose(distance, neiInfo["distance"]):
                            print("ERROR: Distances mismatch:", catom)
                            Distances_mismatch += 1

        assert not Wrong_interacting_pairs
        assert not Missing_interacting_pairs
        assert not Duplication_in_neighbour_list
        assert not Distances_mismatch
        # cost = end-start
        # assert cost == 0, TimeoutError(f'build neigh cost too mush time {cost=}')
