import pytest
import numpy as np
from molpy.core.arraydict import ArrayDict
try:
    import h5py
except ImportError:
    h5py = None

class TestNumpy:
    """
    Test class for Numpy operations.
    """

    @pytest.fixture(scope="function", name="hete")
    def test_init_(self):

        atoms =  ArrayDict({
            "scalar": np.array([1, 2, 3]),
            "vectorial": np.random.rand(3, 3),
            "tensorial": np.random.rand(3, 3, 3),
        })
        return atoms
    
    @pytest.fixture(scope="function", name="homo")
    def test_init_homo(self):

        atoms = ArrayDict({
            "x": np.array([1, 2, 3]),
            "y": np.array([2, 3, 4]),
            "z": np.array([3, 4, 5])
        })
        return atoms
    
    def test_getitem(self, hete):
        """
        Test the __getitem__ method of NestDict.
        """
        assert np.array_equal(hete["scalar"], np.array([1, 2, 3]))
        assert set(hete[["scalar", "vectorial"]].keys()) == {"scalar", "vectorial"}
        assert hete[1] == {
            "scalar": [2],
            "vectorial": hete["vectorial"][1],
            "tensorial": hete["tensorial"][1],
        }
        assert hete[0:2] == {
            "scalar": np.array([1, 2]),
            "vectorial": hete["vectorial"][0:2],
            "tensorial": hete["tensorial"][0:2],
        }

    def test_setitem(self, hete):
        """
        Test the __setitem__ method of NestDict.
        """
        # Set a single item
        hete["scalar"] = np.array([4, 5, 6])
        assert np.array_equal(hete["scalar"], np.array([4, 5, 6]))

        # Set multiple items
        hete[["scalar", "vectorial"]] = ArrayDict({
            "scalar": np.array([4, 5, 6]),
            "vectorial": np.random.rand(3, 3),
            "tensorial": np.random.rand(3, 3, 3),
        })
        assert set(hete.keys()) == {"scalar", "vectorial", "tensorial"}
        assert np.array_equal(hete["scalar"], np.array([4, 5, 6]))

        # Set a slice
        hete[0:2] = ArrayDict({
            "scalar": np.array([7, 8]),
            "vectorial": np.random.rand(2, 3),
            "tensorial": np.random.rand(2, 3, 3),
        })
        assert np.array_equal(hete["scalar"][0:2], np.array([7, 8]))
        assert np.array_equal(hete["vectorial"][0:2], hete["vectorial"][0:2])

        with pytest.raises(ValueError):
            hete[0: 2] = ArrayDict({
                "scalar": np.array([4, 5, 6]),
                "vectorial": np.random.rand(3, 3),
                "tensorial": np.random.rand(3, 3, 3),
            })

        with pytest.raises(KeyError):
            hete[0: 2] = ArrayDict({
                "scalar": np.array([4, 5]),
                "tensorial": np.random.rand(3, 3),
            })
        

    def test_getitem_invalscalar(self, hete):
        """
        Test the __setitem__ method of NestDict with invalscalar input.
        """
        with pytest.raises(KeyError):
            hete["invalscalar_key"]

        with pytest.raises(KeyError):
            hete[["invalscalar_key"]]

    def test_iterrows(self, hete):
        """
        Test the iterrows method of NestDict.
        """
        # Iterate over rows
        for i, row in enumerate(hete.iterrows()):
            assert isinstance(row, ArrayDict)
            assert "scalar" in row
            assert row["vectorial"].shape == hete["vectorial"][i].shape

    def test_from_dicts(self):
        """
        Test the from_dicts method of ArrayDict.
        """
        dicts = [
            {"scalar": 1, "vectorial": np.random.rand(3)},
            {"scalar": 2, "vectorial": np.random.rand(3)},
            {"scalar": 3, "vectorial": np.random.rand(3)},
        ]
        hete = ArrayDict.from_dicts(dicts)
        assert len(hete) == 2
        assert set(hete.keys()) == {"scalar", "vectorial"}

    def test_to_hdf5(self, hete, tmp_path):
        """
        Test the to_hdf5 method of ArrayDict.
        """
        h5py_path = tmp_path / "test.h5"
        hete.to_hdf5(h5py_path)
        
        import h5py
        with h5py.File(h5py_path, "r") as f:
            assert "scalar" in f
            assert "vectorial" in f
            assert np.array_equal(f["scalar"][:], hete["scalar"])
            assert np.array_equal(f["vectorial"][:], hete["vectorial"])

    def test_to_hdf5_in_bytes(self, hete):
        """
        Test the to_hdf5_bytes method of ArrayDict.
        """
        h5py_bytes = hete.to_hdf5(path=None)
        
        import h5py
        with h5py.File(h5py_bytes, "r") as f:
            assert "scalar" in f
            assert "vectorial" in f
            assert np.array_equal(f["scalar"][:], hete["scalar"])
            assert np.array_equal(f["vectorial"][:], hete["vectorial"])

    def test_concat(self):

        """
        Test the concat method of ArrayDict.
        """
        hete1 = ArrayDict({
            "scalar": np.array([1, 2, 3]),
            "vectorial": np.random.rand(3, 3),
        })
        hete2 = ArrayDict({
            "scalar": np.array([4, 5, 6]),
            "vectorial": np.random.rand(3, 3),
        })
        hete_concat = ArrayDict.concat([hete1, hete2])
        
        assert np.array_equal(hete_concat["scalar"], np.array([1, 2, 3, 4, 5, 6]))
        assert hete_concat["vectorial"].shape == (6, 3)

    def test_get_multi_items(self, homo, hete):

        x = homo[["x", "y", "z"]].to_numpy()
        assert x.shape == (3, 3)

        y = hete[["scalar", "vectorial"]].to_numpy()
        assert y.shape == (3, 4)
        with pytest.raises(ValueError):
            hete[["scalar", "vectorial", "tensorial"]].to_numpy()
            