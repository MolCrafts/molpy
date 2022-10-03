# # author: Roy Kid
# # contact: lijichen365@126.com
# # date: 2022-06-12
# # version: 0.0.1

# from molpy.io import Readers
# import numpy as np
# import numpy.testing as npt

# class TestLammps:

#     def test_read_data(self):

#         reader = Readers['DataReaders']['lammps']('tests/test_io/data/lammps.data')
#         data = reader.get_data()
#         assert data
#         assert data['box']['xlo'] == 0
#         assert data['box']['yhi'] == 50
#         atoms = reader.get_atoms()

#         # test atom section
#         npt.assert_equal(data['Atoms']['type'], np.array([1,1,3,3,4,4,4]))
#         assert atoms.n_atoms == 7
#         npt.assert_equal(atoms['id'], np.arange(7)+1)

#         # test bond section
#         npt.assert_equal(data['Bonds']['type'], np.array([1,1,1,1,1,1]))
#         atoms.n_bonds == 6

#         # test angle section
        
#         # test dihedral section

#     def test_read_dump(self):

#         reader = Readers['TrajReaders']['lammps']('tests/test_io/data/lammps.dump')
#         assert reader.n_frames == 3
#         data = reader.get_frame(0)
#         assert data
#         assert data['box']['xlo'] == 0
#         assert data['box']['yhi'] == 50
#         npt.assert_equal(data['Atoms']['type'], np.array([1,1,3,3,4,4,4]))

#         atoms = reader.get_atoms()