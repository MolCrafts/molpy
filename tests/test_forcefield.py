# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-18
# version: 0.0.1

import numpy as np
import numpy.testing as npt

class TestForcefield:

    def test_bond_match(self):

        # mock a Frame
        frame = mp.Frame()
        frame.atoms['type'] = np.array([1, 2, 3, 4])
        
        connects = np.array([
            [0, 1],  # 1-2
            [0, 2],  # 1-3
            [0, 3],  # 1-4
            [1, 2],  # 2-3
            [1, 3],  # 2-4
            [2, 3]   # 3-4
        ])
        frame.add_connects(connects)

        ff = mp.Forcefield()
        at1 = ff.def_atom('1')
        at2 = ff.def_atom('2')
        at3 = ff.def_atom('3')
        at4 = ff.def_atom('4')
        bt1 = ff.def_bond(at1, at2)
        bt2 = ff.def_bond(at1, at3)
        bt3 = ff.def_bond(at1, at4)
        bt4 = ff.def_bond(at2, at3)
        bt5 = ff.def_bond(at2, at4)
        bt6 = ff.def_bond(at3, at4)

        atomtypes = frame.atoms['type'][connects]
        bondtypes = ff.match_bonds(atomtypes)
        npt.assert_equal(bondtypes, np.array([bt1, bt2, bt3, bt4, bt5, bt6]))
