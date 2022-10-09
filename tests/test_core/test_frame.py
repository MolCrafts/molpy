# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-05
# version: 0.0.1

import numpy as np
import pytest
import molpy as mp
import numpy.testing as npt

class TestFrame:

    @pytest.fixture(scope='class', name='data')
    def init_atom(self):

        xyz = np.array([[1, 0, 0], [6, 0, 0]])
        type = np.array([1, 2])

        yield {'xyz':xyz, 'type':type}

    def test_init_dynamic_frame(self, data):

        dframe = self.init_dynamic_frame(data)

        assert dframe.n_atoms == 2

    def init_dynamic_frame(self, data)->mp.DynamicFrame:

        dframe = mp.DynamicFrame()

        for i in zip(*data.values()):
            dframe.add_atom(**{k: v for k, v in zip(data.keys(), i)})

        return dframe

    def test_create_static_frame_from_dynamic_frame(self, data):

        dframe = self.init_dynamic_frame(data)


    def test_atom(self):

        dframe = mp.DynamicFrame.from_dict(
            {'id': np.arange(6),
            'type': np.arange(6),}
        )

        dframe.del_atom(2)
        assert dframe.n_atoms == 5

        dframe.del_atom(4)
        assert dframe.n_atoms == 4

    def test_bond(self):

        dframe = mp.DynamicFrame.from_dict(
            {'id': np.arange(6),
            'type': np.arange(6),}
        )

        dframe.add_bond(0, 1)
        assert dframe.n_atoms == 6
        assert dframe.n_bonds == 1
        assert dframe._topo.n_bonds == 1

        dframe.del_bond(0, 1)
        assert dframe.n_atoms == 6
        assert dframe.n_bonds == 0
        assert dframe._topo.n_bonds == 0

        dframe.add_bonds([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]], type=[1, 2, 3, 4, 5, 6])
        assert dframe.n_atoms == 6
        assert dframe.n_bonds == 6
        assert dframe._topo.n_bonds == 6

        dframe.del_atom(2)
        assert dframe.n_atoms == 5
        assert dframe.n_bonds == 4
        assert dframe._topo.n_bonds == 4
        # atom_list = [0, 1, 3, 4, 5]
        bond1 = dframe.get_bond(0, 1)
        assert bond1['type'] == 1
        with pytest.raises(KeyError):
            dframe.get_bond(1, 2)
            dframe.get_bond(2, 3)
        bond4 = dframe.get_bond(3, 4)
        assert bond4['type'] == 5
        bond6 = dframe.get_bond(4, 0)
        assert bond6['type'] == 6

