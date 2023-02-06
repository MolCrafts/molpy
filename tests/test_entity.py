# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-04
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np
import numpy.testing as npt

class TestAtom:

    def test_init(self):

        atom = mp.Atom(name='O', xyz=[0, 0, 0])
        assert atom.name == 'O'
        assert atom.xyz == [0, 0, 0]


class TestBond:

    def test_init(self):

        bond = mp.Bond(0, 1)
        assert bond.itom == 0
        assert bond.jtom == 1

class TestResidue:

    def test_init(self):

        ch2 = mp.presets.molecules.CH2()
        assert ch2.name == 'CH2'
        assert ch2.natoms == 3
        assert ch2.nbonds == 2
        assert len(ch2.atoms) == 3
        assert len(ch2.bonds) == 2

        assert ch2.atoms[0]['name'] == 'C'
        assert ch2.atoms[1]['name'] == 'H1'
        assert ch2.atoms[2]['name'] == 'H2'

        assert ch2.bonds[0].itom['name'] == 'C'
        assert ch2.bonds[0].jtom['name'] == 'H1'
        assert ch2.bonds[1].itom['name'] == 'C'
        assert ch2.bonds[1].jtom['name'] == 'H2'
        
        npt.assert_equal(ch2.connect, np.array([[0, 1], [0, 2]]))

    def test_translate(self):
            
        ch2 = mp.presets.molecules.CH2()
        npt.assert_allclose(ch2.translate([1, 1, 1]).xyz, np.array([[1, 1, 1], [1, 2.089, 1], [2.026, 0.455, 1]]))


class TestMolecule:

    def test_tip3p(self):

        h2o = mp.presets.molecules.tip3p()
        assert h2o.natoms == 3
        assert h2o.atoms[0].name == 'O'

        bonds = h2o.topology.bonds['index']
        assert bonds[0][0] == 0