# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-03
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np

class TestTopology:

    @pytest.fixture(scope='function', name='ring')
    def create_ring(self):

        topo = mp.Topology()

        # bond list in Frame
        nbonds = 6
        atom_dict = {}
        bond_dict = {}
        for n in range(nbonds):
            atom = mp.Atom(id=n)
            atom_dict[id(atom)] = atom

        atom_list = list(atom_dict)
        for n in range(nbonds):
            bond = mp.Bond(atom_list[n-1], atom_list[n], id=n)
            bond_dict[id(bond)] = bond
            topo.add_bond(id(atom_list[n-1]), id(atom_list[n]), id(bond))

        yield topo, bond_dict, atom_list

    def test_atom_manage(self, ring):

        topo, bond_dict, atom_list = ring

        # -- Test add atoms --

        # -- Test del atoms --
        atom_id = topo.del_atom(id(atom_list[0]))
        assert topo.n_atoms == 5
        assert topo.n_bonds == 4

    def test_bond_manage(self, ring):

        topo, bond_dict, atom_list = ring

        # -- Test add bonds --
        assert topo.n_atoms == 6
        assert topo.n_bonds == 6
        assert bond_dict[topo.get_bond(id(atom_list[-1]), id(atom_list[0]))]['id'] == 0
        assert bond_dict[topo.get_bond(id(atom_list[0]), id(atom_list[1]))]['id'] == 1

        # -- Test del bonds --
        delete_bond_idx = topo.del_bond(id(atom_list[0]), id(atom_list[1]))
        bond_dict.pop(delete_bond_idx)
        assert topo.n_atoms == 6
        assert topo.n_bonds == 5
        assert bond_dict[topo.get_bond(id(atom_list[1]), id(atom_list[2]))]['id'] == 2


