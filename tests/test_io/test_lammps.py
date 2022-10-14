# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-06-12
# version: 0.0.1

from molpy.io import Readers
import numpy as np
import numpy.testing as npt

class TestLammps:

    def test_read_data(self):

        reader = Readers['DataReaders']['lammps']
        with reader('tests/test_io/data/lammps.data') as f:
            frame = f.get_data()

        # test info
        assert frame.n_atoms == 7
        box = frame.box
        box_matrix = box.to_matrix()
        assert box_matrix[0, 0] == box_matrix[1, 1] == box_matrix[2, 2] == 50.0
        # test atom section
        atom_list = frame.atoms
        assert atom_list[0]['id'] == 1
        assert atom_list[0]['molid'] == 1
        assert atom_list[0]['x'] == 24.88
        # test bond section
        assert frame.n_bonds == 6

        # test angle section
        
        # test dihedral section

    def test_read_dump(self):

        reader = Readers['TrajReaders']['lammps']
        
        # read traj not a context manager since the filehandle need to keep a long time
        f = reader('tests/test_io/data/lammps.dump')
        sframe = f.get_one_frame(1)

        # test info
        assert sframe.n_atoms == 7
        assert sframe.box.Lx == sframe.box.Ly == 50.0
        assert sframe.timestep == 100

        # test atoms
        npt.assert_equal(sframe['id'], np.array([1, 4, 2, 3, 5, 6, 7]))
