import pytest
import molpy as mp
import numpy.testing as npt
import numpy as np

class TestFrame:
    
    def test_molid(self, test_data_path):

        frame = mp.io.load_frame(test_data_path / 'data/lammps-data/molid.lmp', format='LAMMPS Data')
        assert frame.n_atoms == 12
        npt.assert_equal(frame.atoms['molid'], np.array([0,0,0,1,1,1,2,2,2,3,3,3]))

class TestTrajectory:
    
    def test_lammps_polymer(self, test_data_path):

        traj = mp.io.load_traj(test_data_path / 'data/lammps/polymer.lammpstrj')
        assert traj.n_frames == 42
        frame = next(iter(traj))
        assert frame.n_atoms == 1714