import pytest
import molpy as mp
import numpy as np
import numpy.testing as npt

Item = mp.core.frame.Item
ItemList = mp.core.frame.ItemList

class TestItem:

    def test_item_init(self):

        item = Item()
        
class TestItemList:

    @pytest.fixture(scope="class")
    def item_list(self):
        return ItemList([Item(name="O"), Item(name="H"), Item(name="H")])

    def test_get_item(self, item_list):

        assert item_list[0]['name'] == "O"

    def test_get_props(self, item_list):

        npt.assert_equal(item_list["name"], np.array(["O", "H", "H"]))


class TestFrame:

    @pytest.fixture(scope="class")
    def frame(self):
        return mp.Frame()

    def test_frame_init(self):

        frame = mp.Frame()
        
    def test_frame_props(self, frame):

        frame['int'] = 1
        assert frame['int'] == 1

        frame['double'] = 1.0
        assert frame['double'] == 1.0

        frame['str'] = "str"
        assert frame['str'] == "str"

        frame['bool'] = True
        assert frame['bool'] == True

        frame['ndarray'] = np.array([1, 2, 3])
        npt.assert_equal(frame['ndarray'], np.array([1, 2, 3]))

    def test_frame_atoms(self, frame):

        frame.add_atom(**{mp.Alias.name: "O", mp.Alias.R: [0, 0, 0]})
        frame.add_atom(name="H", R=[1, 0, 0])
        frame.add_atom(name="H", R=[0, 1, 0])

        assert frame.n_atoms == 3
        assert len(frame.atoms) == 3

        # get Atom and check properties
        assert frame.atoms[0][mp.Alias.name] == "O"

        # get atom properties
        npt.assert_equal(frame.atoms["name"], np.array(["O", "H", "H"]))
        assert frame.atoms["R"].shape == (3, 3)

    # def test_frame_bonds(self, frame):

    #     frame.add_bond(type=1)
    #     frame.add_bond(type=1)

    #     assert frame.n_bonds == 2
    #     assert len(frame.bonds) == 2
    #     assert frame.bonds[0]['bond_i'] == 0

    #     assert frame.bonds["bond_i"] == np.array([0, 0])

    # def test_link_topo(self, frame):

    #     frame.topology = mp.Topology()
    #     frame.topology.add_bonds([[0, 1], [0, 2]])

        # bond_idx = frame.topology.get_bonds()