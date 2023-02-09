# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-18
# version: 0.0.1

import numpy as np
import molpy as mp
import numpy.testing as npt

class TestForcefield:

    def test_load_tip3p(self):

        ff = mp.preset.forcefields.tip3p()

        # test atomTypes
        O = ff.get_atomType('tip3p-O')
        H = ff.get_atomType('tip3p-H')
        assert O['class']['name'] == 'OW'
        assert H['class']['name'] == 'HW'
        assert O['element'] == 'O'
        assert H['element'] == 'H'

        # test bondTypes
        OH = ff.get_bondType(O, H)
        assert OH['length'] == '0.09572'
        assert OH['k'] == '462750.4'

        # test residue
        h2o = ff.get_residue('HOH')
        assert h2o.get_atomType('O')['type'] == 'tip3p-O'
        assert h2o.get_atomType('H1')['type'] == 'tip3p-H'
        assert h2o.get_atomType('H2')['type'] == 'tip3p-H'
        
    def test_match_atomType(self):

        ff = mp.preset.forcefields.tip3p()
        h2o = mp.preset.molecules.tip3p()

        # test match_atomType
        h2o_residue = ff.get_residue('HOH')
        literal_h2o_atom_name = ['O', 'H1', 'H2']
        atomTypeName = list(map(h2o_residue.get_atomType, literal_h2o_atom_name))
        assert atomTypeName[0]['type'] == 'tip3p-O'
        atomTypes = list(map(ff.get_atomType, map(lambda x: x['type'], atomTypeName)))
        assert atomTypes[0]['class']['name'] == 'OW'
        assert atomTypes[1]['class']['name'] == 'HW'

        # test match_bondType
        connect = h2o.connect
        bondTypes = list(map(lambda x: ff.get_bondType(atomTypes[x[0]], atomTypes[x[1]]), connect))

        assert bondTypes[0] == bondTypes[1]
        
    def test_render_residue(self):

        ff = mp.preset.forcefields.tip3p()
        h2o = mp.preset.molecules.tip3p().residues[0]

        ff.render_residue(h2o)
        assert h2o.atoms[0].mass == 15.99943
        assert h2o.atoms[1].mass == 1.007947
        assert h2o.atoms[2].mass == 1.007947
        
        assert h2o.bonds[0]['k'] == "462750.4"
        assert h2o.bonds[1]['k'] == "462750.4"
        assert h2o.bonds[0]['length'] == "0.09572"
        assert h2o.bonds[1]['length'] == "0.09572"

    def test_load_gaff(self):
        ff = mp.preset.forcefields.gaff()
