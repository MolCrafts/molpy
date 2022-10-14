# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-16
# version: 0.0.2

from molpy.core.forcefield import ForceField
import molpy as mp
import pytest
import numpy as np

class TestForceField:

    @pytest.fixture(scope='class', name='ff')
    def init_forcefield(self):

        ff = ForceField()

        yield ff

    # def test_load_xml(self):

    #     ff = ForceField()
    #     ff.load_xml('tests/test_io/data/forcefield.xml')

    def test_def_atom(self, ff:ForceField):

        ff.def_atom('c1', 'C', mass=12.0107, charge=0.0)
        ff.def_atom('h1', 'H', mass=1.0079, charge=0.0)

        atom = ff.get_atom('c1')
        assert atom.name == 'c1'
        assert atom.mass == 12.0107
        assert atom.charge == 0.0

        atoms = ff.get_atom_by_class('H')
        assert len(atoms) == 1
        assert atoms[0].name == 'h1'

    def test_def_bond(self, ff:ForceField):
        
        pass

    def test_def_pair(self, ff:ForceField):

        at1 = ff.get_atom('c1')
        at2 = ff.get_atom('h1')

        ff.def_pair(ff.PairStyle.lj_cut, at1, at1, epsilon=2, sigma=0.1)
        ff.def_pair(ff.PairStyle.lj_cut, at2, at2, epsilon=2, sigma=0.1)

        # nblist -> pairs -> atomtypes -> nonbondParams
        # pairs = np.array([[0, 1], [1, 2]])
        # atomtype_id = atomTypes[pairs]
        # atomTypes = ff.get_atom(atomtype_id)
        # nonbondParams = ff.get_pair(atomtype_id)

        Params = ff.get_pair(at1, at1)
        Params = ff.get_pairs([[at1, at1]])

# class TestForceParamMatching:

#     @pytest.fixture(scope='class', name='dframe')
#     def init_dynamic(self, ):

#         dframe = mp.DynamicFrame()

#         # create a dframe with 10 atoms and 9 bonds
#         dframe.add_atoms(xyz=np.random.rand(10, 3), type=['c1']*10, mass=np.ones(10))
#         dframe.add_bonds(bonds=np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]))

#         yield dframe

#     @pytest.fixture(scope='class', name='sframe')
#     def init_static(self, dframe:mp.DynamicFrame):

#         sframe = dframe.to_static()
#         # create a dframe with 10 atoms and 9 bonds

#         yield sframe


#     def test_sframe(self, sframe):

#         assert sframe.n_atoms == 10
#         assert sframe.n_bonds == 9