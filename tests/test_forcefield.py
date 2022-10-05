# # author: Roy Kid
# # contact: lijichen365@126.com
# # date: 2022-07-18
# # version: 0.0.1

# import pytest
# from molpy.atoms import Residue
# from molpy.core.forcefield import ForceField, Node

# class TestForcefield:

#     def test_load_xml(self):
        
#         ff = ForceField()
#         ff.load_xml('tests/test_io/data/forcefield.xml')
        
#     @pytest.fixture(scope='class', name='ff')
#     def test_def_force(self):

#         ff = ForceField()

#         ff.def_atom('c1', 'C', mass=12.0107, charge=0.0)
#         ff.def_atom('h1', 'H', mass=1.0079, charge=0.0)

#         ff.def_force('HarmonicBondForce', 'c1', 'c2', length=1.0)
#         ff.def_force('HarmonicBondForce', 'C', 'H', length=1.0)

#         ff.def_force('HarmonicAngleForce', 'c1', 'c2', 'c3', angle=120.0)
        
#         ff.def_force('HarmonicDihedralForce', 'c1', 'c2', 'c3', 'c4', angle=120.0)

#         yield ff

#     def test_get_force(self, ff):

#         atom = ff.get_atom('c1')
#         assert atom.name == 'c1'
#         assert atom.mass == 12.0107
#         assert atom.charge == 0.0

#         bond = ff.get_bond('HarmonicBondForce', 'c1', 'c2')
#         assert bond.type1 == 'c1'
#         assert bond.type2 == 'c2'
#         assert bond.length == 1.0

#         angle = ff.get_angle('HarmonicAngleForce', 'c1', 'c2', 'c3')
#         assert angle.type1 == 'c1'
#         assert angle.type2 == 'c2'
#         assert angle.type3 == 'c3'
#         assert angle.angle == 120.0

#         dihedral = ff.get_dihedral('HarmonicDihedralForce', 'c1', 'c2', 'c3', 'c4')
#         assert dihedral.type1 == 'c1'
#         assert dihedral.type2 == 'c2'
#         assert dihedral.type3 == 'c3'
#         assert dihedral.type4 == 'c4'
#         assert dihedral.angle == 120.0

#     def test_add_residue(self):

#         ff = ForceField()
#         ff.def_atom('h1', 'H', mass=1.0079, charge=0.0)
#         ff.def_atom('h2', 'H', mass=1.0079, charge=0.0)
#         ff.def_atom('o', 'O', mass=15.9994, charge=-0.834)

#         ff.def_force('HarmonicBondForce', 'h1', 'o', length=1.0)
#         ff.def_force('HarmonicBondForce', 'h2', 'o', length=1.0)

#         residue = Residue('h2o')
#         residue.add_atoms(type=[381, 381, 380])
#         residue.add_bonds([[0, 2], [1, 2]])

#         ff.def_residue(residue)

#         re:Node = ff.get_residue('h2o')
#         assert re.tag == 'h2o'
#         assert re.get_attribs('Atom', 'type') == [381, 381, 380]
#         assert re.get_attribs('Bond', 'from') == [0, 1]
#         assert re.get_attribs('Bond', 'to') == [2, 2]

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

    def test_def_nonbond(self, ff:ForceField):

        at1 = ff.get_atom('c1')
        at2 = ff.get_atom('h1')

        ff.def_nonbond(ff.Nonbond.lj126, at1, at1, epsilon=2, sigma=0.1)
        ff.def_nonbond(ff.Nonbond.lj126, at2, at2, epsilon=2, sigma=0.1)

        # nblist -> pairs -> atomtypes -> nonbondParams
        # pairs = np.array([[0, 1], [1, 2]])
        # atomtype_id = atomTypes[pairs]
        # atomTypes = ff.get_atom(atomtype_id)
        # nonbondParams = ff.get_nonbond(atomtype_id)

        Params = ff.get_nonbond(at1, at1)
        Params = ff.get_nonbonds([[at1, at1]])

    
