# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from molpy_refactor.core.trajectory import Trajectory

class TestFrame:

    def test_load_from_trajectory(self, ):

        data = '/home/roy/work/molpy-refactor/tests/tests-data/lammps/polymer.lammpstrj'

        traj = Trajectory.load(data)

        frame = traj.read()
        frame.natoms == 1000
        frame['positions'].shape == (1000, 3)