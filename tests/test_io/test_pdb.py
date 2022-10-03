# # author: Roy Kid
# # contact: lijichen365@126.com
# # date: 2022-06-12
# # version: 0.0.1

# from molpy.io import Readers
# import numpy as np
# import numpy.testing as npt
# from molpy.io.pdb import PdbStructure, PDBFile
# import pytest

# class TestPDBReader:

#     @pytest.fixture(name='pdbStructure')
#     def test_pdb_structure(self):

#         file = 'tests/test_io/data/lig.pdb'
#         extraParticleIdentifier='EP'
#         if isinstance(file, PdbStructure):
#             pdb = file
#         else:
#             inputfile = file
#             own_handle = False
#             if isinstance(file, str):
#                 inputfile = open(file)
#                 own_handle = True
#             pdb = PdbStructure(inputfile, load_all_models=True, extraParticleIdentifier=extraParticleIdentifier)
#             if own_handle:
#                 inputfile.close()
#         yield pdb

#     def test_pdb_reader(self, pdbStructure):

#         pdb = pdbStructure
#         PDBFile(pdb)
