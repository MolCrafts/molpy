import pytest
import molpy as mp
import numpy.testing as npt
import numpy as np

class TestFrame:
    
    def test_molid(self, test_data_path):

        frame = mp.io.load_frame(test_data_path / 'lammps-data/molid.lmp', format='LAMMPS Data')
        assert frame.natoms == 12
        npt.assert_equal(frame.atoms['_molid'], np.array([0,0,0,1,1,1,2,2,2,3,3,3]))

class TestTrajectory:
    
    def test_lammps_polymer(self, test_data_path):

        traj = mp.io.load_trajectory(test_data_path / 'lammps/polymer.lammpstrj')
        assert traj.nsteps == 42
        frame = next(iter(traj))
        assert frame.natoms == 1714