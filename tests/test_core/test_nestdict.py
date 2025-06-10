import pytest
import numpy as np
from nesteddict import NestDict
import io
try:
    import h5py
except ImportError:
    h5py = None

class TestNestDict:

    @pytest.fixture(scope="function", name="nd")
    def test_init(self):
        return NestDict(
            {
                "a1": 1,
                "a2": {"b1": 2, "b2": {"c1": 3, "c2": {"d1": 4, "d2": 5}}},
                3: "a",
            }
        )

    def test_init_failed(self):
        with pytest.raises(TypeError):
            NestDict(1)
            NestDict([1, 2, 3])

    def test_traverse_failed(self, nd):

        assert nd._traverse(3)
        with pytest.raises(KeyError):
            nd._traverse((1, 2, 3))

    def test_delitem(self, nd):

        del nd["a1"]
        assert "a1" not in nd

        del nd[["a2", "b2", "c2", "d2"]]
        assert "d2" not in nd[["a2", "b2", "c2"]]

        with pytest.raises(KeyError):
            del nd["a2", "b2"]  # only list but tuple

    def test_eq(self):
        assert NestDict({"a": 1}) == {"a": 1}
        assert NestDict({"a": {"b": 2}}) != {"a": {"b": 1}}

    def test_bool(self, nd):
        assert nd
        assert not NestDict()

    def test_getitem(self, nd):

        assert nd["a1"] == 1
        assert nd[["a2", "b1"]] == 2
        assert nd[["a2", "b2", "c1"]] == 3
        assert nd[["a2", "b2", "c2", "d1"]] == 4

    def test_setitem(self, nd):

        nd["a1"] = 10
        nd[["a2", "b1"]] = 20
        nd[["a2", "b2", "c1"]] = 30
        nd[["a2", "b2", "c2", "d1"]] = 40
        assert nd["a1"] == 10
        assert nd[["a2", "b1"]] == 20
        assert nd[["a2", "b2", "c1"]] == 30
        assert nd[["a2", "b2", "c2", "d1"]] == 40

        nd[["a3", "b1"]] = 50
        nd[["a3", "b2", "c1"]] = 60
        assert nd[["a3", "b1"]] == 50
        assert nd[["a3", "b2", "c1"]] == 60

        nd["tuple", "as", "key"] = 70
        assert nd[("tuple", "as", "key")] == 70

    def test_construct(self):

        nd = NestDict(
            {
                "a": {"b": 1},
            }
        )
        assert nd[["a", "b"]] == 1
        nd[["a", "c"]] = 2
        assert nd[["a", "c"]] == 2

    def test_str(self):
        nd = NestDict({"a": {"b": 1}})
        assert str(nd) == "<{'a': {'b': 1}}>"

    def test_repr(self):
        nd = NestDict({"a": {"b": 1}})
        assert repr(nd) == "<{'a': {'b': 1}}>"

    def test_copy(self, nd):

        nd_copy = nd.copy()
        assert nd == nd_copy

    def test_flatten(self, nd):

        fd = nd.flatten()
        assert ("a1",) in fd
        assert ("a2", "b1") in fd

    def test_get(self, nd):

        assert nd.get("a1") == 1
        assert nd.get("a2.b1") == 2
        assert nd.get("a2.b2.c1") == 3
        assert nd.get("a2.b2.c2.d1") == 4

        assert nd.get("a3.b1") is None
        assert nd.get("a3.b2.c1") is None

    def test_set(self, nd):

        nd.set("a1", 10)
        nd.set("a2.b1", 20)
        nd.set("a2.b2.c1", 30)
        nd.set("a2.b2.c2.d1", 40)
        assert nd["a1"] == 10
        assert nd[["a2", "b1"]] == 20
        assert nd[["a2", "b2", "c1"]] == 30
        assert nd[["a2", "b2", "c2", "d1"]] == 40

        nd.set("a3.b1", 50)
        nd.set("a3.b2.c1", 60)
        assert nd[["a3", "b1"]] == 50
        assert nd[["a3", "b2", "c1"]] == 60

    def test_clear(self, nd):

        nd.clear()
        assert len(nd) == 0

    def test_keys(self, nd):

        keys = nd.keys()
        assert "a1" in keys
        assert "a2" in keys

    def test_values(self, nd):

        values = nd.values()
        assert 1 in values

    def test_iter(self, nd):

        for key in nd:
            assert key in nd.keys()

    def test_update_dict(self, nd):

        nd.update({"a1": 10, "a2": {"b1": 20, "b2": {"c1": 30, "c2": {"d1": 40}}}})
        assert nd["a1"] == 10
        assert nd[["a2", "b1"]] == 20
        assert nd[["a2", "b2", "c1"]] == 30
        assert nd[["a2", "b2", "c2", "d1"]] == 40

    def test_update_nested(self, nd):

        new_nd = NestDict()
        new_nd.update(nd)

        assert new_nd["a1"] == 1
        assert new_nd[["a2", "b1"]] == 2
        assert new_nd[["a2", "b2", "c1"]] == 3
        assert new_nd[["a2", "b2", "c2", "d1"]] == 4

    def test_concat_lists(self):
        a = NestDict({'x': [1, 2]})
        b = NestDict({'x': [3, 4]})
        a.concat(b)
        assert a._data['x'] == [1, 2, 3, 4]

    def test_concat_numpy_arrays(self):
        a = NestDict({'x': np.array([1, 2])})
        b = NestDict({'x': np.array([3, 4])})
        a.concat(b)
        np.testing.assert_array_equal(a._data['x'], np.array([1, 2, 3, 4]))

    def test_concat_nested_dict(self):
        a = NestDict({'x': {'a': 1}})
        b = NestDict({'x': {'b': 2}})
        a.concat(b)
        assert a._data['x'] == {'a': 1, 'b': 2}

    def test_concat_nested_nestdict(self):
        a = NestDict({'x': NestDict({'a': [1]})})
        b = NestDict({'x': NestDict({'a': [2, 3]})})
        a.concat(b)
        assert a._data['x']._data['a'] == [1, 2, 3]

    def test_missing_key_raises(self):
        a = NestDict({'x': [1]})
        b = NestDict({'y': [2]})
        with pytest.raises(KeyError):
            a.concat(b)

    def test_incompatible_type_raises(self):
        a = NestDict({'x': [1]})
        b = NestDict({'x': {'a': [1]}})
        a.concat(b, {dict: lambda x, y: x + y['a']})
        assert a['x'] == [1, 1]

    def test_multiple_concat(self):
        a = NestDict({'x': [1]})
        b = NestDict({'x': [2, 3]})
        c = NestDict({'x': [4, 5]})
        a.concat([b, c])
        assert a._data['x'] == [1, 2, 3, 4, 5]

    def test_nested(self):

        a = NestDict({'x': {'y': 1}})
        
        assert isinstance(a['x'], dict)
        assert a[['x', 'y']] == 1
        with pytest.raises(KeyError):
            # not allowed to use tuple
            # since tuple usually used as key
            assert a['x', 'y'] == 1 

    def test_to_h5py(self, nd):

        import h5py

        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as f:
            nd.to_hdf5(f)

        with h5py.File(buffer, "r") as f:
            assert f["a1"][()] == 1
            assert f["a2/b1"][()]== 2
            assert f["a2/b2/c1"][()] == 3
            assert f["a2/b2/c2/d1"][()] == 4