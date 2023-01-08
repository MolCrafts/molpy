# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from molpy_refactor.core.trajectory import Trajectory

class TestTrajectory:

    def test_load(self):

        data = '/home/roy/work/molpy-refactor/tests/tests-data/lammps/polymer.lammpstrj'

        traj = Trajectory.load('/home/roy/work/molpy-refactor/tests/tests-data/lammps/polymer.lammpstrj')

        assert traj.nsteps == 100
        assert traj.path == data

