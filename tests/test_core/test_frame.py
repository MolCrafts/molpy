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
        sframe = dframe.atoms
        
        assert sframe.n_atoms == 2
        npt.assert_allclose(sframe['xyz'], data['xyz'])
        npt.assert_allclose(sframe['type'], data['type'])

    # def test_topo(self):

    #     idx = np.arange(6)
    #     dframe = self.init_dynamic_frame({'idx':idx})  # linear
    #     dframe.add_bond_by_index(0, 1)
    #     dframe.add_bond_by_index(1, 2)
    #     dframe.add_bond_by_index(2, 3)
    #     dframe.add_bond_by_index(3, 4)
    #     dframe.add_bond_by_index(4, 5)

    #     assert dframe.topo.n_bonds == 5
    #     assert dframe.topo.n_angles == 4
    #     assert dframe.topo.n_dihedrals == 3
