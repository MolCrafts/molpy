# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-26
# version: 0.0.1

import pytest

from molpy.io.smiles import read_smiles
from molpy.factory import full


class TestSmiles:
    @pytest.mark.parametrize('smiles, node_data, edge_data, explicit_h', (
    (
        'CCCC',
        [(0, {'element': 'C', 'charge': 0, 'hcount': 3, 'aromatic': False}),
         (1, {'element': 'C', 'charge': 0, 'hcount': 2, 'aromatic': False}),
         (2, {'element': 'C', 'charge': 0, 'hcount': 2, 'aromatic': False}),
         (3, {'element': 'C', 'charge': 0, 'hcount': 3, 'aromatic': False})],
        [(0, 1, {'order': 1}),
         (1, 2, {'order': 1}),
         (2, 3, {'order': 1})],
        False
    ), ))
    def test_read(self, smiles, node_data, edge_data, explicit_h):
        found = read_smiles(smiles, explicit_hydrogen=explicit_h)
        print(found.atoms)
        print(found.bonds)
        atomName = [n[0] for n in node_data]
        atomProperty = [n[1] for n in node_data]
        atomProperty_nested = {}
        for k in atomProperty[0]:
            atomProperty_nested[k] = []
        for d in atomProperty:
            for k, v in d.items():
                atomProperty_nested[k].append(v)
        expected = full("smiles", atomName, **atomProperty_nested)
        for bondidx in edge_data:
            expected.addBondByIndex(bondidx[0], bondidx[1], **bondidx[2])
        assert expected.nbonds == 3
