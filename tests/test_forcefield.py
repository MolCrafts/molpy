# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-18
# version: 0.0.1

import numpy as np
import molpy as mp
import numpy.testing as npt

class TestForcefield:

    def test_load_presets(self):

        ff = mp.presets.forcefields.tip3p()

        # test atomTypes
        O = ff.get_atomType('tip3p-O')
        H = ff.get_atomType('tip3p-H')
        assert O['class'] == 'OW'
        assert H['class'] == 'HW'
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

        ff = mp.presets.forcefields.tip3p()
        h2o = mp.presets.molecules.tip3p()

        # test match_atomType
        h2o_residue = ff.get_residue('HOH')
        literal_h2o_atom_name = h2o.atoms['name']
        atomTypeName = list(map(h2o_residue.get_atomType, literal_h2o_atom_name))
        assert atomTypeName[0]['type'] == 'tip3p-O'
        atomTypes = list(map(ff.get_atomType, map(lambda x: x['type'], atomTypeName)))
        assert atomTypes[0]['class'] == 'OW'
        assert atomTypes[1]['class'] == 'HW'

        # test match_bondType
        connect = h2o.topology.bonds['index']
        bondTypes = list(map(lambda x: ff.get_bondType(atomTypes[x[0]], atomTypes[x[1]]), connect))

        assert bondTypes[0] == bondTypes[1]
        
