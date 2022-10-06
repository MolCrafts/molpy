# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-16
# version: 0.0.2

from molpy.core.forcefield import ForceField
import pytest

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

    
