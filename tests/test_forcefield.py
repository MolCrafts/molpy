# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-18
# version: 0.0.1

import pytest
from molpy.forcefield import ForceField

class TestForcefield:

    def test_load_xml(self):
        
        ff = ForceField()
        ff.load_xml('tests/test_io/data/forcefield.xml')
        
    @pytest.fixture(scope='class', name='ff')
    def test_def_force(self):

        ff = ForceField()

        ff.def_atom('c1', 'C', mass=12.0107, charge=0.0)
        ff.def_atom('h1', 'H', mass=1.0079, charge=0.0)

        ff.def_force('HarmonicBondForce', 'c1', 'c2', length=1.0)
        ff.def_force('HarmonicBondForce', 'C', 'H', length=1.0)

        ff.def_force('HarmonicAngleForce', 'c1', 'c2', 'c3', angle=120.0)
        
        ff.def_force('HarmonicDihedralForce', 'c1', 'c2', 'c3', 'c4', angle=120.0)

        yield ff

    def test_get_force(self, ff):

        atom = ff.get_atom('c1')
        assert atom.name == 'c1'
        assert atom.mass == 12.0107
        assert atom.charge == 0.0

        bond = ff.get_bond('HarmonicBondForce', 'c1', 'c2')
        assert bond.type1 == 'c1'
        assert bond.type2 == 'c2'
        assert bond.length == 1.0

        angle = ff.get_angle('HarmonicAngleForce', 'c1', 'c2', 'c3')
        assert angle.type1 == 'c1'
        assert angle.type2 == 'c2'
        assert angle.type3 == 'c3'
        assert angle.angle == 120.0

        dihedral = ff.get_dihedral('HarmonicDihedralForce', 'c1', 'c2', 'c3', 'c4')
        assert dihedral.type1 == 'c1'
        assert dihedral.type2 == 'c2'
        assert dihedral.type3 == 'c3'
        assert dihedral.type4 == 'c4'
        assert dihedral.angle == 120.0
