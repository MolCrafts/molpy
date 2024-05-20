import pytest
import molpy as mp
import numpy as np
import numpy.testing as npt


# class TestItemList:

#     @pytest.fixture(scope="class", name="item_list")
#     def test_init(self):
#         return ItemList([Item(_name="O"), Item(_name="H"), Item(_name="H")])

#     def test_getitem(self, item_list):

#         assert item_list[0][mp.Alias.name] == "O"
#         npt.assert_equal(item_list[mp.Alias.name], np.array(["O", "H", "H"]))

#     def test_getattr(self, item_list):

#         npt.assert_equal(item_list.name, np.array(["O", "H", "H"]))


# class TestFrame:

#     @pytest.fixture(scope="class")
#     def frame(self):
#         return mp.Frame()

#     def test_frame_init(self):

#         frame = mp.Frame()

#     def test_frame_props(self, frame):

#         frame["int"] = 1
#         assert frame["int"] == 1

#         frame["double"] = 1.0
#         assert frame["double"] == 1.0

#         frame["str"] = "str"
#         assert frame["str"] == "str"

#         frame["bool"] = True
#         assert frame["bool"] == True

#         frame["ndarray"] = np.array([1, 2, 3])
#         npt.assert_equal(frame["ndarray"], np.array([1, 2, 3]))

#     def test_frame_atoms(self, frame):

#         frame.add_atom(**{mp.Alias.name: "O", mp.Alias.R: [0, 0, 0]})
#         frame.add_atom(name="H", R=[1, 0, 0])
#         frame.add_atom(name="H", R=[0, 1, 0])

#         assert frame.n_atoms == 3
#         assert len(frame.atoms) == 3

#         # get Atom and check properties
#         assert frame.atoms[0][mp.Alias.name] == "O"

#         # get atom properties
#         npt.assert_equal(frame.atoms["name"], np.array(["O", "H", "H"]))
#         assert frame.atoms["R"].shape == (3, 3)

# class TestStaticStruct:

#     def test_init(self):

#         struct = mp.StaticStruct()

#     def test_get_set_atoms(self):

#         struct = mp.StaticStruct()
#         struct.atoms.name = np.array(["O", "H", "H"])