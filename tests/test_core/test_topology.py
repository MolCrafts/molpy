import pytest
import molpy as mp
import numpy as np
import numpy.testing as npt

class TestTopology:

    @pytest.fixture(scope="class")
    def topo(self):
        return mp.Topology()
    
    def test_topology_init(self):
            
        topology = mp.Topology()

    def test_add_atoms(self, topo):

        topo.add_atoms(3)  # add three vertices
        # topo.add_atom('O')  # add atom with name

        assert topo.n_atoms == 3

    def test_add_bonds(self, topo):

        topo.add_bonds([[0, 1], [0, 2]])  # add two bonds
        assert topo.n_bonds == 2

    def test_add_angles(self, topo):

        topo.add_angles([[1, 0, 2]])
        topo.simplify()
        assert topo.n_bonds == 2
        # assert topo.n_angles == 1

    def test_get_connect(self, topo):

        bond_idx = topo.get_bonds()
        npt.assert_equal(bond_idx, np.array([[0, 1], [0, 2]]))

        # angle_idx = topo.get_angles()
        # npt.assert_equal(angle_idx, np.array([[1, 0, 2]]))