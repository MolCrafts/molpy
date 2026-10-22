import json
import subprocess
import sys
from collections.abc import ItemsView, KeysView, Mapping, MutableMapping, ValuesView

import pickle

import molrs
import numpy as np
import pytest
from molrs.core import Block, Frame, MetaValue


class _SubFrame(molrs.core.Frame):
    """Module level, so pickle can find it."""


class TestFrameConstruction:
    def test_empty(self):
        f = Frame()
        assert len(f) == 0
        assert f.keys() == []
        assert f.box is None

    def test_setitem_populates_blocks_and_meta(self):
        f = Frame()
        atoms = Block()
        atoms.insert("symbol", ["C", "H"])
        atoms.insert("x", np.array([0.0, 1.0], dtype=np.float64))
        f["atoms"] = atoms
        f.meta = {"source": MetaValue("string", "pytest")}

        assert sorted(f.keys()) == ["atoms"]
        assert f["atoms"].n_rows == 2
        assert list(f["atoms"]["symbol"]) == ["C", "H"]
        np.testing.assert_allclose(f["atoms"]["x"], [0.0, 1.0])
        assert f.meta["source"] == "pytest"

    def test_repr_empty(self):
        r = repr(Frame())
        assert "Frame" in r
        assert "no" in r  # box=no


class TestOneFrame:
    """There is one ``Frame``: the PyO3 class, constructed and read natively."""

    def test_molrs_frame_is_the_native_class(self):
        assert molrs.core.Frame is molrs._native.Frame

    def test_a_frame_can_be_subclassed(self):
        """Core data classes are extensible; a subclass is still a Frame."""

        class Sub(molrs.core.Frame):
            pass

        assert isinstance(Sub(), molrs.core.Frame)

    def test_a_frame_subclass_pickles_as_itself(self):
        sub = _SubFrame()
        sub["atoms"] = Block({"x": np.array([0.5])})
        sub.tag = "kept"
        back = pickle.loads(pickle.dumps(sub))
        assert type(back) is _SubFrame
        assert back.tag == "kept"
        assert back["atoms"]["x"].tolist() == [0.5]

    def test_readers_return_the_one_class(self, tmp_path):
        path = tmp_path / "one.xyz"
        path.write_text("1\n\nH 0.0 0.0 0.0\n", encoding="utf-8")
        assert type(molrs.io.read_xyz(path)) is Frame

    def test_a_graph_serialises_to_the_one_class(self):
        mol = molrs.core.Atomistic()
        mol.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0)
        frame = mol.to_frame()
        assert type(frame) is Frame
        assert type(frame["atoms"]) is Block

    def test_ctor_takes_blocks_meta_and_box(self):
        f = Frame(
            {"atoms": {"x": [1.0, 2.0]}, "bonds": Block({"atomi": [0], "atomj": [1]})},
            meta={"title": MetaValue("string", "t")},
            box=molrs.core.Box(molrs.core.Cell.lengths([5.0, 5.0, 5.0])),
        )
        assert sorted(f.keys()) == ["atoms", "bonds"]
        np.testing.assert_array_equal(f["atoms"]["x"], [1.0, 2.0])
        assert f["bonds"].dtype("atomi") == "uint"
        assert f.meta["title"] == "t"
        assert f.box.volume() == pytest.approx(125.0)

    def test_a_frame_is_not_a_blocks_argument(self):
        # Copying is `frame.copy()`; there is no wrap-an-existing-frame form.
        with pytest.raises(TypeError, match="copy"):
            Frame(Frame())

    def test_setitem_accepts_a_mapping(self):
        f = Frame()
        f["atoms"] = {"x": [1.0, 2.0], "id": np.array([1, 2], dtype=np.int64)}
        assert f["atoms"].n_rows == 2
        assert f["atoms"].dtype("id") == "uint"

    def test_setitem_refuses_anything_else(self):
        with pytest.raises(TypeError):
            Frame()["atoms"] = 3

    def test_a_block_key_tuple_is_not_column_access(self):
        f = Frame({"atoms": {"x": [1.0]}})
        with pytest.raises(TypeError):
            f["atoms", "x"]


class TestFrameBlockHandle:
    """``frame["name"]`` is a handle on the stored block, not an empty stand-in.

    Every native Block member answers on the stored data (inventory bug 1).
    """

    @staticmethod
    def _grid_frame() -> Frame:
        grid = Block({"rho": np.array([0.0, 1.0, 2.0, 3.0])})
        grid.set_shape([2, 2])
        return Frame({"grid": grid, "atoms": {"x": [1.0, 2.0, 3.0]}})

    def test_structural_shape_reads_the_stored_block(self):
        assert self._grid_frame()["grid"].structural_shape == [2, 2]

    def test_shape_reads_the_stored_block(self):
        assert self._grid_frame()["atoms"].shape == [3]

    def test_a_row_selection_reads_the_stored_block(self):
        rows = self._grid_frame()["atoms"][np.array([2, 0])]
        np.testing.assert_array_equal(rows["x"], [3.0, 1.0])

    def test_write_csv_block_writes_the_stored_rows(self):
        sink = bytearray()
        molrs.io.write_csv_block(sink, self._grid_frame()["atoms"])
        text = sink.decode()
        lines = text.splitlines()
        assert lines[0] == "x"
        assert [float(v) for v in lines[1:]] == [1.0, 2.0, 3.0]

    def test_resize_writes_through_to_the_frame(self):
        f = self._grid_frame()
        f["atoms"].resize(5)
        assert f["atoms"].n_rows == 5
        np.testing.assert_array_equal(f["atoms"]["x"], [1.0, 2.0, 3.0, 0.0, 0.0])

    def test_a_column_write_through_the_handle_lands_in_the_frame(self):
        f = Frame({"atoms": {"x": [1.0, 2.0]}})
        f["atoms"]["y"] = np.array([3.0, 4.0])
        np.testing.assert_array_equal(f["atoms"]["y"], [3.0, 4.0])


class TestFrameCopy:
    def test_copy_does_not_share_buffers(self):
        f = Frame({"atoms": {"x": np.array([1.0, 2.0])}})
        f.box = molrs.core.Box(molrs.core.Cell.lengths([5.0, 5.0, 5.0]))
        g = f.copy()
        assert not np.shares_memory(f["atoms"]["x"], g["atoms"]["x"])
        g["atoms"]["x"][0] = 9.0
        np.testing.assert_array_equal(f["atoms"]["x"], [1.0, 2.0])
        assert g.box is not None

    def test_a_subset_does_not_share_buffers_with_its_source(self):
        f = Frame(
            {"atoms": {"x": np.array([1.0, 2.0])}, "extra": {"v": np.array([5.0])}}
        )
        sub = f.subset([0])
        sub["extra"]["v"][0] = 7.0
        np.testing.assert_array_equal(f["extra"]["v"], [5.0])


class TestFramePickle:
    def test_pickle_keeps_blocks_typed_meta_box_and_masks(self):
        import pickle

        f = Frame({"atoms": {"x": [1.0, 2.0]}})
        f["atoms"].insert("tag", np.array([4, 0], dtype=np.int32), [True, False])
        f.meta["temperature"] = MetaValue("f64", 300.0)
        f.box = molrs.core.Box(molrs.core.Cell.lengths([2.0, 2.0, 2.0]))
        restored = pickle.loads(pickle.dumps(f))
        assert type(restored) is Frame
        np.testing.assert_array_equal(restored["atoms"]["x"], [1.0, 2.0])
        np.testing.assert_array_equal(restored["atoms"].validity("tag"), [True, False])
        assert restored.meta.dtype("temperature") == "f64"
        assert restored.box.volume() == pytest.approx(8.0)


class TestFrameBlockAccess:
    def test_setitem_getitem(self):
        f = Frame()
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        f["atoms"] = b
        assert "atoms" in f
        assert len(f) == 1

        atoms = f["atoms"]
        assert atoms.n_rows == 2

    def test_getitem_returns_live_block_handle(self):
        f = Frame()
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        f["atoms"] = b

        atoms = f["atoms"]
        atoms.insert("y", np.array([3.0, 4.0], dtype=np.float64))

        np.testing.assert_allclose(f["atoms"]["y"], [3.0, 4.0])

    @pytest.mark.parametrize(
        "touch_meta",
        [
            lambda meta: meta.__setitem__("step", 1),
            lambda meta: meta.update({"step": 2}),
            lambda meta: meta.__delitem__("seed"),
        ],
        ids=["set", "update", "delete"],
    )
    def test_a_meta_write_keeps_block_handles_valid(self, touch_meta):
        f = Frame()
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        f["atoms"] = b
        f.meta["seed"] = 7

        atoms = f["atoms"]
        touch_meta(f.meta)

        np.testing.assert_allclose(atoms["x"], [1.0, 2.0])

    def test_getitem_missing_raises_key_error(self):
        f = Frame()
        with pytest.raises(KeyError):
            _ = f["missing"]

    def test_delitem(self):
        f = Frame()
        f["atoms"] = Block()
        del f["atoms"]
        assert "atoms" not in f

    def test_delitem_missing_raises_key_error(self):
        f = Frame()
        with pytest.raises(KeyError):
            del f["missing"]

    def test_contains(self):
        f = Frame()
        f["atoms"] = Block()
        assert "atoms" in f
        assert "bonds" not in f

    def test_keys(self):
        f = Frame()
        f["atoms"] = Block()
        f["bonds"] = Block()
        assert sorted(f.keys()) == ["atoms", "bonds"]

    def test_overwrite_block(self):
        f = Frame()
        b1 = Block()
        b1.insert("x", np.array([1.0], dtype=np.float64))
        f["atoms"] = b1
        assert f["atoms"].n_rows == 1

        b2 = Block()
        b2.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        f["atoms"] = b2
        assert f["atoms"].n_rows == 2


class TestFrameBox:
    def test_default_none(self):
        assert Frame().box is None

    def test_set_box(self):
        f = Frame()
        box_ = molrs.core.Box(molrs.core.Cell.lengths([10.0, 10.0, 10.0]))
        f.box = box_
        assert f.box is not None
        assert pytest.approx(f.box.volume(), abs=1) == 1000.0

    def test_clear_box(self):
        f = Frame()
        f.box = molrs.core.Box(molrs.core.Cell.lengths([10.0, 10.0, 10.0]))
        f.box = None
        assert f.box is None

    def test_repr_with_box(self):
        f = Frame()
        f.box = molrs.core.Box(molrs.core.Cell.lengths([10.0, 10.0, 10.0]))
        assert "yes" in repr(f)


class TestFrameMeta:
    def test_exact_dtype_roundtrip(self):
        f = Frame()
        f.meta = {
            "tag": MetaValue("i64", 9_007_199_254_740_993),
            "temperature": MetaValue("f64", 300.0),
            "stress": MetaValue("f64x6", [1, 2, 3, 4, 5, 6]),
        }
        assert f.meta["tag"] == 9_007_199_254_740_993
        assert f.meta["temperature"] == pytest.approx(300.0)
        assert f.meta["stress"] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        assert isinstance(f.meta["stress"], tuple)

    def test_json_document_values_are_accepted(self):
        f = Frame()
        f.meta = {"label": "string-only", "nested": {"tool": "molrec", "run": 3}}
        assert f.meta["label"] == "string-only"
        assert f.meta["nested"] == {"tool": "molrec", "run": 3}
        assert f.meta.dtype("nested") == "json"

    def test_set_and_get(self):
        f = Frame()
        f.meta = {"title": "test", "source": "pytest"}
        meta = f.meta
        assert meta["title"] == "test"
        assert meta["source"] == "pytest"
        assert meta == {"title": "test", "source": "pytest"}

    def test_empty_meta(self):
        f = Frame()
        assert len(f.meta) == 0
        assert dict(f.meta) == {}

    def test_overwrite_meta(self):
        f = Frame()
        f.meta = {"a": 1}
        f.meta = {"b": 2}
        assert "b" in f.meta
        assert "a" not in f.meta

    def test_update_and_get(self):
        f = Frame()
        f.meta.update({"a": 1})
        f.meta.update(b="x")
        assert f.meta.get("a") == 1
        assert f.meta.get("missing") is None
        assert f.meta.get("missing", 9) == 9

    def test_unsupported_value_is_rejected(self):
        f = Frame()
        with pytest.raises(TypeError, match="not JSON-serializable"):
            f.meta["bad"] = object()

    def test_a_self_referencing_list_is_rejected_not_a_crash(self):
        # Run in a child interpreter: the defect is a stack overflow that kills
        # the process, so asserting in-process would take pytest down with it.
        script = (
            "import molrs\n"
            "from molrs._native import Frame\n"
            "a = []\n"
            "a.append(a)\n"
            "try:\n"
            "    Frame().meta['x'] = a\n"
            "except ValueError:\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit('a self-referencing list was accepted')\n"
        )
        child = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            encoding="utf-8",
        )
        assert child.returncode == 0, (child.returncode, child.stderr[-2000:])

    def test_none_is_a_json_null_not_a_rejection(self):
        # `meta` is a JSON document, and JSON has null. Rejecting a bare None
        # while accepting {"a": None} would be an arbitrary split.
        f = Frame()
        f.meta["absent"] = None
        assert f.meta["absent"] is None
        assert f.meta.dtype("absent") == "json"

    def test_write_through(self):
        f = Frame()
        f.meta["title"] = "water"
        assert f.meta["title"] == "water"
        f.meta["title"] = "ice"
        assert f.meta["title"] == "ice"
        del f.meta["title"]
        assert "title" not in f.meta

    def test_narrow_float_dtypes_are_refused(self):
        # One float: `f64`. A narrow tag is refused, not promoted.
        with pytest.raises(TypeError, match="unknown metadata dtype"):
            MetaValue("f32", 300.0)

    def test_a_plain_write_takes_the_values_own_dtype(self):
        f = Frame()
        f.meta["temperature"] = MetaValue("f64", 300.0)
        assert f.meta.dtype("temperature") == "f64"
        f.meta["temperature"] = 310.0
        assert f.meta.dtype("temperature") == "f64"
        assert f.meta["temperature"] == pytest.approx(310.0)

    def test_reassigning_a_read_value_is_an_identity(self):
        f = Frame()
        f.meta = {
            "tag": MetaValue("i64", 9_007_199_254_740_993),
            "stress": MetaValue("f64x6", [1, 2, 3, 4, 5, 6]),
        }
        before = {k: f.meta.dtype(k) for k in f.meta}
        for key in list(f.meta):
            f.meta[key] = f.meta[key]
        assert {k: f.meta.dtype(k) for k in f.meta} == before
        assert f.meta["tag"] == 9_007_199_254_740_993
        # A plain write re-infers the slot's dtype from the value in hand:
        # the read value is a Python float, so the slot is f64.
        f.meta["temperature"] = MetaValue("f64", 300.0)
        f.meta["temperature"] = f.meta["temperature"]
        assert f.meta.dtype("temperature") == "f64"

    def test_a_plain_write_replaces_the_slots_dtype(self):
        f = Frame()
        f.meta["count"] = MetaValue("i64", 3)
        f.meta["count"] = 1.5
        assert f.meta["count"] == 1.5
        assert f.meta.dtype("count") == "f64"

    def test_enumeration_follows_insertion_order(self):
        # Not alphabetical: a reintroduced sort must fail this case.
        f = Frame()
        f.meta["zeta"] = 1
        f.meta["alpha"] = 2
        f.meta["mu"] = 3
        order = ["zeta", "alpha", "mu"]
        assert list(f.meta) == order
        assert list(f.meta.keys()) == order
        assert list(f.meta.items()) == [("zeta", 1), ("alpha", 2), ("mu", 3)]
        assert list(f.meta.values()) == [1, 2, 3]
        assert list(dict(f.meta)) == order
        assert repr(f.meta) == "{'zeta': 1, 'alpha': 2, 'mu': 3}"
        assert f.meta.popitem() == ("mu", 3)

    def test_mapping_protocol(self):
        f = Frame()
        f.meta = {"zeta": 1, "alpha": "two"}
        assert isinstance(f.meta, MutableMapping)
        assert list(f.meta) == ["zeta", "alpha"]
        assert dict(f.meta) == {"zeta": 1, "alpha": "two"}
        assert f.meta == {"zeta": 1, "alpha": "two"}
        assert f.meta.pop("zeta") == 1
        f.meta.setdefault("c", 3)
        assert f.meta["c"] == 3
        f.meta |= {"d": 4}
        assert f.meta["d"] == 4
        f.meta.clear()
        assert len(f.meta) == 0

    def test_copying_a_frame_keeps_exact_dtypes(self):
        # dict(meta) drops the tags, so the copy path must not go through it.
        f = Frame()
        f.meta = {"temperature": MetaValue("f64", 300.0)}
        copied = f.copy()
        assert copied.meta.dtype("temperature") == "f64"
        assert copied.meta.copy() == {"temperature": pytest.approx(300.0)}
        assert set(copied.meta.typed()) == {"temperature"}
        assert copied.meta.typed()["temperature"].dtype == "f64"

    def test_nested_document_is_a_snapshot(self):
        f = Frame()
        f.meta["run"] = {"step": 1}
        with pytest.raises(TypeError):
            f.meta["run"]["step"] = 2
        assert f.meta["run"] == {"step": 1}
        document = f.meta["run"].copy()
        document["step"] = 2
        f.meta["run"] = document
        assert f.meta["run"] == {"step": 2}
        assert f.meta.dtype("run") == "json"

    def _insertion_order_meta(self):
        # t, a, stress: alphabetical order is the failure mode.
        f = Frame()
        f.meta["t"] = MetaValue("f64", 300.0)
        f.meta["a"] = 1
        f.meta["stress"] = MetaValue("f64x6", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        return f.meta

    def test_mapping_views_are_collections_abc_views(self):
        m = self._insertion_order_meta()
        keys, values, items = m.keys(), m.values(), m.items()
        assert isinstance(keys, KeysView)
        assert isinstance(values, ValuesView)
        assert isinstance(items, ItemsView)
        assert (
            type(keys).__module__ == "collections.abc"
            and type(keys).__name__ == "KeysView"
        )
        assert (
            type(values).__module__ == "collections.abc"
            and type(values).__name__ == "ValuesView"
        )
        assert (
            type(items).__module__ == "collections.abc"
            and type(items).__name__ == "ItemsView"
        )

    def test_views_follow_insertion_order(self):
        m = self._insertion_order_meta()
        assert list(m.keys()) == ["t", "a", "stress"]
        assert list(m.values()) == [
            pytest.approx(300.0),
            1,
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        ]

    def test_views_stay_live_across_writes(self):
        m = self._insertion_order_meta()
        keys, values, items = m.keys(), m.values(), m.items()
        m["z"] = "late"
        assert "z" in keys
        assert len(keys) == len(values) == len(items) == 4

    def test_view_set_algebra_and_membership(self):
        m = self._insertion_order_meta()
        assert m.keys() & {"t"} == {"t"}
        assert m.keys() | {"z"} == {"t", "a", "stress", "z"}
        assert ("a", 1) in m.items()
        assert ("a", 2) not in m.items()
        assert 300.0 in m.values()

    def test_popitem_is_last_inserted_after_an_earlier_overwrite(self):
        m = self._insertion_order_meta()
        assert m.popitem() == ("stress", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

        m = self._insertion_order_meta()
        m["t"] = 9
        assert m.popitem() == ("stress", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

        m = self._insertion_order_meta()
        for _ in range(3):
            m.popitem()
        assert len(m) == 0
        with pytest.raises(KeyError):
            m.popitem()

    def test_empty_views_are_false_and_track_clear(self):
        m = Frame().meta
        assert len(m.keys()) == len(m.values()) == len(m.items()) == 0
        assert bool(m.keys()) is False
        m["t"] = 1
        keys = m.keys()
        m.clear()
        assert len(keys) == 0

    def test_iteration_does_not_recurse_into_keys(self):
        m = self._insertion_order_meta()
        assert list(m) == ["t", "a", "stress"]
        assert list(m.keys()) == ["t", "a", "stress"]
        assert list(dict(m)) == ["t", "a", "stress"]
        assert [key for key, _value in m.items()] == ["t", "a", "stress"]

    def test_a_non_str_lookup_is_absent_and_a_write_is_type_error(self):
        m = self._insertion_order_meta()
        assert (1 in m) is False
        assert m.get(1) is None
        assert m.get(1, "d") == "d"
        assert m.pop(1, "d") == "d"
        assert 1 not in m
        assert (1, 2) not in m.items()
        assert m.get([]) is None
        with pytest.raises(KeyError) as missing:
            m[1]
        assert missing.value.args == (1,)
        with pytest.raises(KeyError) as deleted:
            del m[1]
        assert deleted.value.args == (1,)
        with pytest.raises(TypeError):
            m[1] = 2
        with pytest.raises(TypeError):
            m.setdefault(1, 2)

    def test_update_self_reinfers_float_without_panic(self):
        def fresh():
            f = Frame()
            f.meta["t"] = MetaValue("f64", 300.0)
            f.meta["a"] = 1
            return f.meta

        m = fresh()
        m.update(m)
        assert list(m) == ["t", "a"]
        assert m["t"] == pytest.approx(300.0)
        assert m["a"] == 1
        assert m.dtype("a") == "i64"
        assert m.dtype("t") == "f64"

        m = fresh()
        m |= m
        assert list(m) == ["t", "a"]
        assert m["t"] == pytest.approx(300.0)
        assert m["a"] == 1
        assert m.dtype("a") == "i64"
        assert m.dtype("t") == "f64"

    def test_deleting_an_unvisited_key_during_values_raises_key_error(self):
        # The key list is materialized, so the miss is a KeyError from the
        # not-yet-visited lookup, not dict's RuntimeError for a size change.
        m = self._insertion_order_meta()
        with pytest.raises(KeyError):
            first = True
            for _value in m.values():
                if first:
                    del m["stress"]
                    first = False

    def test_frame_meta_is_unhashable(self):
        with pytest.raises(TypeError):
            hash(Frame().meta)
        left = Frame()
        left.meta["t"] = 1
        with pytest.raises(TypeError):
            hash(left.meta)
        right = Frame()
        right.meta["t"] = 1
        assert left.meta == right.meta

    # Every fixed-length vector dtype. Read back, these are tuples at every door.
    _VECTORS = (
        ("bool3", (True, False, True)),
        ("i32x3", (1, -2, 3)),
        ("i64x3", (1, -2, 3)),
        ("u32x3", (1, 2, 3)),
        ("u64x3", (1, 2, 3)),
        ("f64x3", (1.0, 2.0, 3.0)),
        ("f64x6", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
        ("f64x9", tuple(float(i) for i in range(1, 10))),
    )

    def _vector_meta(self, dtype, payload):
        frame = Frame()
        frame.meta[dtype] = MetaValue(dtype, list(payload))
        return frame.meta

    def test_every_vector_dtype_is_a_tuple_at_every_door(self):
        for dtype, payload in self._VECTORS:
            meta = self._vector_meta(dtype, payload)
            got = meta[dtype]
            assert isinstance(got, tuple) and got == payload
            assert isinstance(meta.get(dtype), tuple) and meta.get(dtype) == payload
            assert (
                isinstance(meta.setdefault(dtype), tuple)
                and meta.setdefault(dtype) == payload
            )
            assert meta.dtype(dtype) == dtype
            assert isinstance(next(iter(meta.values())), tuple)
            assert isinstance(next(iter(meta.items()))[1], tuple)
            copied = meta.copy()
            assert isinstance(copied[dtype], tuple) and copied[dtype] == payload

            meta = self._vector_meta(dtype, payload)
            popped = meta.pop(dtype)
            assert isinstance(popped, tuple) and popped == payload

            meta = self._vector_meta(dtype, payload)
            key, value = meta.popitem()
            assert key == dtype
            assert isinstance(value, tuple) and value == payload

        # A dict-literal comparison uses the frozen value, so a list does not match.
        stress = self._vector_meta("f64x6", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        assert stress == {"f64x6": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)}
        assert stress != {"f64x6": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}

    def test_reassigning_a_read_value_keeps_scalar_vector_and_json(self):
        frame = Frame()
        frame.meta["title"] = "water"
        frame.meta["stress"] = MetaValue("f64x6", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        frame.meta["run"] = {"step": 1, "tags": [1, 2]}
        frame.meta["again"] = frame.meta["stress"]
        assert frame.meta.dtype("again") == "f64x6"
        before = {key: frame.meta.dtype(key) for key in ("title", "stress", "run")}
        for key in ("title", "stress", "run"):
            frame.meta[key] = frame.meta[key]
        assert {key: frame.meta.dtype(key) for key in before} == before
        assert frame.meta["title"] == "water"
        assert frame.meta["stress"] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        assert isinstance(frame.meta["stress"], tuple)
        assert frame.meta["run"] == {"step": 1, "tags": (1, 2)}
        assert isinstance(frame.meta["run"]["tags"], tuple)


class TestMetaDocument:
    def test_equals_a_dict_and_is_not_one(self):
        frame = Frame()
        frame.meta["run"] = {
            "tool": "molrec",
            "run": 3,
            "tags": [1, 2],
            "inner": {"step": 1},
        }
        doc = frame.meta["run"]
        expected = {"tool": "molrec", "run": 3, "tags": (1, 2), "inner": {"step": 1}}
        # Equality and the runtime type stay in one function: `==` is true
        # while `isinstance(..., dict)` is false, and neither may drift.
        assert doc == expected
        assert isinstance(doc, dict) is False
        assert isinstance(doc, molrs.core.MetaDocument)
        assert isinstance(doc, Mapping)
        assert (doc != expected) is False
        assert doc != {"tool": "molrec", "run": 3, "tags": [1, 2], "inner": {"step": 1}}
        assert doc["tags"] == (1, 2)
        assert doc["tags"] != [1, 2]
        assert isinstance(doc["tags"], tuple)
        assert isinstance(doc["inner"], molrs.core.MetaDocument)
        assert doc["inner"] == {"step": 1}
        other = Frame()
        other.meta["run"] = {
            "tool": "molrec",
            "run": 3,
            "tags": [1, 2],
            "inner": {"step": 1},
        }
        assert doc == other.meta["run"]
        assert (doc != other.meta["run"]) is False

    def test_mapping_surface(self):
        frame = Frame()
        frame.meta["run"] = {"step": 1, "tool": "molrec"}
        doc = frame.meta["run"]
        assert len(doc) == 2
        assert sorted(doc) == ["step", "tool"]
        assert sorted(doc.keys()) == ["step", "tool"]
        assert dict(doc.items()) == {"step": 1, "tool": "molrec"}
        assert set(doc.values()) == {1, "molrec"}
        assert "step" in doc
        assert "missing" not in doc
        assert (1 in doc) is False
        assert doc.get("tool") == "molrec"
        assert doc.get("missing") is None
        assert doc.get("missing", 9) == 9
        assert doc.get(1, "d") == "d"
        with pytest.raises(KeyError):
            doc["missing"]
        assert isinstance(doc.keys(), KeysView)
        assert isinstance(doc.values(), ValuesView)
        assert isinstance(doc.items(), ItemsView)
        assert doc.keys() & {"step"} == {"step"}
        assert ("step", 1) in doc.items()
        one = Frame()
        one.meta["run"] = {"step": 1}
        assert repr(one.meta["run"]) == "MetaDocument({'step': 1})"
        empty = Frame()
        empty.meta["run"] = {}
        assert empty.meta["run"] == {}
        assert len(empty.meta["run"]) == 0
        assert repr(empty.meta["run"]) == "MetaDocument({})"

    def test_hash_and_inplace_write_raise(self):
        frame = Frame()
        frame.meta["run"] = {"step": 1, "inner": {"n": 2}}
        doc = frame.meta["run"]
        with pytest.raises(TypeError):
            hash(doc)
        with pytest.raises(TypeError):
            hash(frame.meta)
        with pytest.raises(TypeError):
            doc["step"] = 3
        with pytest.raises(TypeError):
            frame.meta["run"]["step"] = 3
        with pytest.raises(TypeError):
            frame.meta["run"]["inner"]["n"] = 4
        assert frame.meta["run"] == {"step": 1, "inner": {"n": 2}}

    def test_copy_is_a_deep_plain_dict_json_accepts(self):
        frame = Frame()
        frame.meta["run"] = {"step": 1, "tags": [1, 2], "rows": [{"a": 1}, {"a": None}]}
        doc = frame.meta["run"]
        with pytest.raises(TypeError):
            json.dumps(doc)
        plain = doc.copy()
        assert isinstance(plain, dict)
        assert isinstance(plain["tags"], list)
        assert isinstance(plain["rows"], list)
        assert isinstance(plain["rows"][0], dict)
        assert not isinstance(plain["rows"][0], molrs.core.MetaDocument)
        assert json.loads(json.dumps(plain)) == plain
        plain["step"] = 9
        plain["rows"][0]["a"] = 5
        assert frame.meta["run"] == {
            "step": 1,
            "tags": (1, 2),
            "rows": ({"a": 1}, {"a": None}),
        }
        assert isinstance(frame.meta["run"]["rows"], tuple)
        assert isinstance(frame.meta["run"]["rows"][0], molrs.core.MetaDocument)

    def test_json_array_written_back_stays_an_array(self):
        frame = Frame()
        frame.meta["tags"] = [1, 2]
        assert frame.meta.dtype("tags") == "json"
        assert isinstance(frame.meta["tags"], tuple)
        assert frame.meta["tags"] == (1, 2)
        frame.meta["tags"] = frame.meta["tags"]
        assert frame.meta.dtype("tags") == "json"
        assert frame.meta["tags"] == (1, 2)
        frame.meta["wrapped"] = {"tags": (1, 2)}
        assert frame.meta.dtype("wrapped") == "json"
        assert frame.meta["wrapped"]["tags"] == (1, 2)
        assert isinstance(frame.meta["wrapped"]["tags"], tuple)


class TestFrameValidation:
    def test_validate_empty(self):
        Frame().validate()

    def test_validate_consistent(self):
        f = Frame()
        b = Block()
        b.insert("x", np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b.insert("y", np.array([0.0, 1.0, 2.0], dtype=np.float64))
        f["atoms"] = b
        f.validate()


class TestFrameSubset:
    """Seam of ``molrs.core.Frame.subset``.

    Row gathering and relation renumbering are proven by
    ``crates/molrs/src/core/frame.rs``; these check the return type, the row
    normaliser (bool mask, int rows, negative wrap) and the error mapping.
    """

    @staticmethod
    def _chain() -> molrs.core.Frame:
        # 4 atoms at x = 0..3 (Å) in two molecules, bonded (0,1), (1,2), (2,3).
        return molrs.core.Frame(
            {
                "atoms": {
                    "x": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
                    "mol_id": np.array([1, 1, 2, 2]),
                },
                "bonds": {
                    "atomi": np.array([0, 1, 2]),
                    "atomj": np.array([1, 2, 3]),
                },
            }
        )

    def test_subset_returns_a_frame_with_the_selected_rows(self):
        sub = self._chain().subset([2, 3])

        assert type(sub) is molrs.core.Frame
        np.testing.assert_array_equal(sub["atoms"]["x"], [2.0, 3.0])
        np.testing.assert_array_equal(sub["atoms"]["mol_id"], [2, 2])
        assert sub["bonds"].n_rows == 1
        np.testing.assert_array_equal(sub["bonds"]["atomi"], [0])
        np.testing.assert_array_equal(sub["bonds"]["atomj"], [1])

    def test_a_bool_mask_and_int_rows_select_the_same_rows(self):
        frame = self._chain()

        by_mask = frame.subset(np.array([True, False, True, False]))
        by_rows = frame.subset([0, 2])

        np.testing.assert_array_equal(by_mask["atoms"]["x"], by_rows["atoms"]["x"])
        np.testing.assert_array_equal(by_mask["atoms"]["x"], [0.0, 2.0])
        assert by_mask["bonds"].n_rows == by_rows["bonds"].n_rows == 0

    def test_a_negative_row_wraps(self):
        sub = self._chain().subset([-1])

        np.testing.assert_array_equal(sub["atoms"]["x"], [3.0])

    def test_subset_by_a_mol_id_comparison_selects_one_molecule(self):
        frame = self._chain()

        one = frame.subset(frame["atoms"]["mol_id"] == 1)

        np.testing.assert_array_equal(one["atoms"]["x"], [0.0, 1.0])
        assert one["bonds"].n_rows == 1

    def test_a_row_past_the_end_is_a_value_error(self):
        with pytest.raises(ValueError):
            self._chain().subset([4])

    def test_a_mask_of_the_wrong_length_is_an_index_error(self):
        with pytest.raises(IndexError):
            self._chain().subset(np.array([True, False]))

    def test_a_missing_block_is_a_key_error(self):
        with pytest.raises(KeyError):
            self._chain().subset([0], block="missing")


class TestFrameConcat:
    @staticmethod
    def _chain(n: int, charge: bool = False) -> molrs.core.Frame:
        frame = molrs.core.Frame()
        atoms = molrs.core.Block()
        atoms.insert("x", np.arange(n, dtype=np.float64))
        if charge:
            atoms.insert("charge", np.zeros(n))
        bonds = molrs.core.Block()
        bonds.insert("atomi", np.arange(n - 1, dtype=np.uint64))
        bonds.insert("atomj", np.arange(1, n, dtype=np.uint64))
        frame["atoms"] = atoms
        frame["bonds"] = bonds
        return frame

    def test_endpoints_are_offset_past_earlier_parts(self):
        joined = molrs.core.Frame.concat([self._chain(2), self._chain(3)])
        assert joined["atoms"].n_rows == 5
        assert list(joined["bonds"]["atomi"]) == [0, 2, 3]
        assert list(joined["bonds"]["atomj"]) == [1, 3, 4]

    def test_concat_of_copies_equals_replicate(self):
        chain = self._chain(3)
        joined = molrs.core.Frame.concat([chain, chain])
        tiled = chain.replicate(2)
        assert list(joined["bonds"]["atomj"]) == list(tiled["bonds"]["atomj"])

    def test_a_column_one_part_lacks_is_null_there(self):
        joined = molrs.core.Frame.concat([self._chain(2, charge=True), self._chain(2)])
        assert list(joined["atoms"].validity("charge")) == [True, True, False, False]

    def test_no_parts_give_an_empty_frame(self):
        assert len(molrs.core.Frame.concat([])) == 0


class TestTopology:
    """``molrs.core.Topology``: the index-only bond graph of a frame."""

    @staticmethod
    def _chain_frame() -> Frame:
        # 0-1-2-3 chain plus an isolated atom 4.
        return Frame(
            {
                "atoms": {"x": [0.0, 1.0, 2.0, 3.0, 9.0]},
                "bonds": {"atomi": [0, 1, 2], "atomj": [1, 2, 3]},
            }
        )

    def test_from_frame_reads_atoms_bonds_and_derived_terms(self):
        topo = molrs.core.Topology.from_frame(self._chain_frame())
        assert (topo.n_atoms, topo.n_bonds) == (5, 3)
        assert (topo.n_angles, topo.n_dihedrals) == (2, 1)
        assert topo.n_components == 2
        np.testing.assert_array_equal(topo.connected_components(), [0, 0, 0, 0, 1])
        np.testing.assert_array_equal(topo.bonds(), [[0, 1], [1, 2], [2, 3]])
        np.testing.assert_array_equal(topo.angles(), [[0, 1, 2], [1, 2, 3]])
        np.testing.assert_array_equal(topo.dihedrals(), [[0, 1, 2, 3]])
        np.testing.assert_array_equal(topo.distances(0), [0, 1, 2, 3, -1])
        assert topo.neighbors(1) == [0, 2]
        assert topo.degree(1) == 2
        assert topo.are_bonded(2, 3) and not topo.are_bonded(0, 3)

    def test_from_frame_refuses_a_frame_without_atoms(self):
        with pytest.raises(ValueError, match="atoms"):
            molrs.core.Topology.from_frame(Frame())

    def test_build_with_add_atoms_and_add_bonds(self):
        topo = molrs.core.Topology(3)
        topo.add_atoms(1)
        topo.add_bonds([(0, 1), (1, 0), (2, 3)])
        assert (topo.n_atoms, topo.n_bonds) == (4, 2)
        np.testing.assert_array_equal(topo.bonds(), [[0, 1], [2, 3]])
        assert len(topo.impropers()) == 0
        with pytest.raises(IndexError):
            topo.add_bonds([(0, 9)])
        assert topo.n_bonds == 2

    def test_del_atom_swap_removes_and_drops_its_bonds(self):
        topo = molrs.core.Topology.from_frame(self._chain_frame())
        topo.del_atom(1)
        # Atom 4 (isolated) moved into slot 1; bonds 0-1 and 1-2 are gone.
        assert (topo.n_atoms, topo.n_bonds) == (4, 1)
        np.testing.assert_array_equal(topo.bonds(), [[2, 3]])
        assert topo.degree(1) == 0

    def test_del_atoms_takes_indices_from_before_the_call(self):
        topo = molrs.core.Topology.from_frame(self._chain_frame())
        topo.del_atoms([0, 3, 3])
        assert (topo.n_atoms, topo.n_bonds) == (3, 1)
        np.testing.assert_array_equal(topo.bonds(), [[1, 2]])

    def test_del_bond_and_del_bonds(self):
        topo = molrs.core.Topology.from_frame(self._chain_frame())
        topo.del_bond(0)
        assert topo.n_bonds == 2
        assert not topo.are_bonded(0, 1)
        topo.del_bonds([0, 1])
        assert topo.n_bonds == 0
        assert topo.n_components == 5

    def test_out_of_range_deletes_raise_and_change_nothing(self):
        topo = molrs.core.Topology.from_frame(self._chain_frame())
        with pytest.raises(IndexError):
            topo.del_atom(5)
        with pytest.raises(IndexError):
            topo.del_atoms([0, 5])
        with pytest.raises(IndexError):
            topo.del_bond(3)
        with pytest.raises(IndexError):
            topo.del_bonds([0, 3])
        assert (topo.n_atoms, topo.n_bonds) == (5, 3)
