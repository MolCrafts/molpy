# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-04
# version: 0.0.1

import molpy as mp

class TestMolecule:

    def test_tip3p(self):

        h2o = mp.presets.molecules.tip3p()
        assert h2o.natoms == 3
        assert h2o.atoms['name'][0] == 'O'

        bonds = h2o.topology.bonds['index']
        assert bonds[0][0] == 0