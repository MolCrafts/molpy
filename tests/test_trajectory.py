# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

import pytest
from molpy_refactor.core.trajectory import Trajectory

class TestTrajectory:

    @pytest.fixture(name='pdb')
    def read_pdb(self, test_data_path):

        path = test_data_path / 'pdb/hemo.pdb'
        traj = Trajectory.load(path)
        print('read once')
        yield traj

    def test_load(self, pdb):

        assert pdb.nsteps == 1
