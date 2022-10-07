# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-03
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np

class TestTopology:

    def test_add_bond(self):
        topo = mp.Topology()
        topo.add_bond(0, 1)
        assert topo.n_bonds == 1

        topo.add_bonds([[1, 2], [2, 3]])
        assert topo.n_bonds == 3

    def test_del_atom(self):

        # before:
        # AoS: [A, B, C, D, E, F]
        # idx: [0, 1, 2, 3, 4, 5]
        # node:[0, 1, 2, 3, 4, 5]

        # after:
        # AoS: [A, B, D, E, F]  # del C
        # idx: [0, 1, 2, 3, 4]
        # node:[0, 1, 3, 4, 5]  # del 2 

        topo = mp.Topology()
        topo.add_bonds([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], type=[1, 2, 3, 4, 5])
        topo.del_atom(2)  # frame passes inx 2 of del atom to topo
        assert topo.n_bonds == 4
        assert topo.node2idx(3) == 2
        assert topo.node2idx(5) == 4
        assert topo.idx2node(2) == 3
        assert topo.idx2node(4) == 5
