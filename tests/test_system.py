# # author: Roy Kid
# # contact: lijichen365@126.com
# # date: 2022-06-12
# # version: 0.0.1

# import pytest
# from molpy.system import System
# import numpy as np
# import numpy.testing as npt

# class TestSystem:
    
#     def test_load_once(self):
        
#         system = System('test_lammps_io')
#         system.load('tests/test_io/data/lammps.data')  # load lammps data
        
#         assert system.atoms.n_atoms == 7
#         npt.assert_equal(system.atoms['type'], np.array([1,1,3,3,4,4,4]))
#         # npt.assert_equal(system.atoms['mass'], np.array([1.0]*system.atoms.n_atoms))
#         npt.assert_equal(system.atoms['q'], np.array([-11.8]*system.atoms.n_atoms))

#         system.load_traj('tests/test_io/data/lammps.dump', 'lammps')

#         system.select_frame(0)
#         npt.assert_equal(system.atoms['type'], np.array([1,1,3,3,4,4,4]))

#         system.select_frame(1)
#         npt.assert_equal(system.atoms['type'], np.array([1,3,1,3,4,4,4]))
#         npt.assert_equal(system.atoms['q'], np.array([11.8]*7))

#         ids = iter([0,1,2])
#         types = iter(np.array([[1,1,3,3,4,4,4], [1,3,1,3,4,4,4], [4,1,4,1,3,3,4]]))
#         for i in system.sample(0, -1, 1):
#             j = next(ids)
#             type = next(types)
#             assert i == j
#             npt.assert_equal(system.atoms['type'], type)