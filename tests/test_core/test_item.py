# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-06
# version: 0.0.1

import numpy as np
import molpy as mp

class TestItem:

    def test_bond(self):

        bond = mp.Bond(mp.Atom(idx=0), mp.Atom(idx=1), type=1)
        assert bond.atom1['idx'] == 0
        assert bond.atom2['idx'] == 1
        assert bond['type'] == 1
