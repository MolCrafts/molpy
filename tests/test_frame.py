# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from molpy.core.trajectory import Trajectory
import pytest
import numpy.testing as npt
import numpy as np

class TestFrame:

    @pytest.fixture(name='pdb')
    def read_pdb(self, test_data_path):

        path = test_data_path / 'pdb/hemo.pdb'
        traj = Trajectory.load(path)
        frame = traj.read()
        yield frame

    def test_pdb_frame(self, pdb):

        assert pdb.natoms == 522
        npt.assert_equal(pdb.box, np.zeros((3, 3)))
        assert pdb['positions'].shape == (522, 3)
        assert pdb.topology.nbonds == 482
        assert pdb.topology.nangles == 823
        assert pdb.topology.ndihedrals == 1126
        assert pdb.topology.nimpropers == 433

        assert pdb.nresidues == 25

    def test_get_residue(self, pdb):

        residue = pdb.get_residue('HEM')
        assert residue.natoms == 73
