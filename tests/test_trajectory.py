# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

import pytest

from molpy_refactor.core.trajectory import Trajectory

class TestTrajectory:

    @pytest.fixture()
    def traj(self):

        data = '/home/roy/work/molpy-refactor/tests/tests-data/pdb/hemo.pdb'

        traj = Trajectory.load(data)

        yield traj


    def test_load(self, traj):

        assert traj.nsteps == 42
        assert traj.path

    def test_read(self, traj):

        frame = traj.read()

