"""FFI smoke tests for the ``*.mrec`` doors (``mp.io.write_mrec``,
``mp.io.mrec.MrecFile`` / ``MrecReader`` …).

Frame and Trajectory are the in-memory objects; the writer is flat on
``mp.io``, the per-section reads and the store machinery in
``mp.io.mrec``.
"""

from __future__ import annotations

from pathlib import Path

import molpy as mp
import numpy as np
import pytest

# Å; dyadic so a bit-exact f64 round-trip is the golden, not a tolerance.
_N_ATOMS = 3
_ATOM_X = (0.0, 1.0, 0.5)
_ATOM_Y = (0.25, 0.0, 2.0)
_ATOM_Z = (0.0, 4.0, 0.125)


def _coords_frame() -> mp.core.Frame:
    atoms = mp.core.Block()
    atoms["x"] = np.array(_ATOM_X, dtype=np.float64)
    atoms["y"] = np.array(_ATOM_Y, dtype=np.float64)
    atoms["z"] = np.array(_ATOM_Z, dtype=np.float64)
    frame = mp.core.Frame()
    frame["atoms"] = atoms
    return frame


def _assert_coords(frame: mp.core.Frame) -> None:
    atoms = frame["atoms"]
    assert atoms.n_rows == _N_ATOMS
    np.testing.assert_array_equal(
        np.asarray(atoms["x"]), np.array(_ATOM_X, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(atoms["y"]), np.array(_ATOM_Y, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(atoms["z"]), np.array(_ATOM_Z, dtype=np.float64)
    )


def _record(path: Path | bytes) -> mp.io.mrec.MrecFile:
    return mp.io.mrec.MrecFile(path)


def _frame_of(path: Path | bytes) -> mp.core.Frame:
    frame = _record(path).frame()
    assert frame is not None
    return frame


def _system_of(path: Path | bytes) -> mp.core.Frame:
    system = _record(path).system()
    assert system is not None
    return system


def _forcefield_of(path: Path | bytes) -> mp.io.mrec.ForceFieldSection:
    section = _record(path).forcefield()
    assert section is not None
    return section


def _trajectory_of(path: Path) -> mp.core.Trajectory:
    return mp.io.mrec.MrecReader(path).to_trajectory()


class TestMrecReader:
    def test_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "traj.mrec"
        trajectory = mp.core.Trajectory([_coords_frame()])
        mp.io.write_mrec(path, trajectory=trajectory)

        reader = mp.io.mrec.MrecReader(path)
        assert len(reader) == 1
        _assert_coords(reader[0])
        _assert_coords(reader[-1])
        with pytest.raises(IndexError):
            reader[1]
        frames = list(reader)
        assert len(frames) == 1
        _assert_coords(frames[0])

    def test_frame_columns_decodes_only_the_named_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "traj.mrec"
        mp.io.write_mrec(path, trajectory=mp.core.Trajectory([_coords_frame()]))
        frame = mp.io.mrec.MrecReader(path).frame_columns(0, [("atoms", "x")])
        assert "x" in frame["atoms"] and "y" not in frame["atoms"]
        np.testing.assert_array_equal(
            np.asarray(frame["atoms"]["x"]), np.array(_ATOM_X, dtype=np.float64)
        )

    def test_a_record_without_a_trajectory_is_refused_by_name(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "snapshot.mrec"
        mp.io.write_mrec(path, frame=_coords_frame())
        with pytest.raises(ValueError, match="trajectory"):
            mp.io.mrec.MrecReader(path)


class TestFrameDoors:
    def test_write_and_read_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "snapshot.mrec"
        mp.io.write_mrec(path, frame=_coords_frame())
        _assert_coords(_frame_of(path))
        assert _record(path).sections() == ["frame", "meta"]
        # A producer that handed in nothing gets an empty identity document.
        assert _record(path).meta() == {}
        # An absent section reads as None.
        assert _record(path).system() is None
        assert _record(path).forcefield() is None

    def test_a_packed_record_in_memory_reads_back(self) -> None:
        buf = bytearray(b"stale")
        mp.io.write_mrec(buf, frame=_coords_frame(), meta={"producer": "test"})
        _assert_coords(_frame_of(bytes(buf)))
        assert _record(bytes(buf)).meta() == {"producer": "test"}

    def test_a_str_path_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="pathlib.Path"):
            mp.io.write_mrec(str(tmp_path / "x.mrec"), frame=_coords_frame())
        with pytest.raises(TypeError, match="Text"):
            mp.io.mrec.MrecFile(str(tmp_path / "x.mrec"))

    def test_write_frame_with_system(self, tmp_path: Path) -> None:
        path = tmp_path / "both.mrec"
        mp.io.write_mrec(path, frame=_coords_frame(), system=_coords_frame())
        _assert_coords(_frame_of(path))
        _assert_coords(_system_of(path))
        assert _record(path).sections() == ["frame", "meta", "system"]


class TestSystemDoors:
    def test_write_and_read_system(self, tmp_path: Path) -> None:
        path = tmp_path / "system.mrec"
        mp.io.write_mrec(path, system=_coords_frame())
        _assert_coords(_system_of(path))
        assert "frame" not in _record(path).sections()
        assert _record(path).frame() is None


class TestTrajectoryDoors:
    def test_write_and_read_trajectory(self, tmp_path: Path) -> None:
        path = tmp_path / "traj.mrec"
        mp.io.write_mrec(path, trajectory=mp.core.Trajectory([_coords_frame()]))
        loaded = _trajectory_of(path)
        assert len(loaded) == 1
        _assert_coords(loaded[0])


class TestSchema:
    def test_a_molrec_version_key_is_an_ordinary_meta_key(self, tmp_path: Path) -> None:
        # Written as given, read back unchanged, checked by no door.
        for value in (99, "1", None):
            path = tmp_path / f"record-{value}.mrec"
            meta = {"producer": "other", "molrec_version": value}
            mp.io.write_mrec(path, frame=_coords_frame(), meta=meta)
            assert _record(path).meta() == meta
            _assert_coords(_frame_of(path))


class TestMrecSurface:
    def test_the_lazy_cursor_is_named_as_in_rust(self) -> None:
        from molpy.io.mrec import MrecReader, MrecWriter

        assert MrecReader is mp.io.mrec.MrecReader
        assert MrecWriter is mp.io.mrec.MrecWriter

    def test_a_formats_reader_is_its_class(self, water_dcd: Path) -> None:
        reader = mp.io.dcd.DcdReader(water_dcd)
        assert isinstance(reader, mp.io.dcd.DcdReader)
        assert len(reader) == 2
        assert reader[0]["atoms"].n_rows == 3

    def test_the_record_writer_is_flat_on_io(self) -> None:
        doors = sorted(name for name in dir(mp.io) if "mrec" in name and name != "mrec")
        assert doors == ["write_mrec"]
        assert not [n for n in dir(mp.io.mrec) if n.startswith(("read", "write"))]

    def test_no_make_helpers(self) -> None:
        makers = [name for name in dir(mp.io.mrec) if name.startswith("make_")]
        assert makers == []


class TestCanonicalWidths:
    def test_a_narrow_canonical_identifier_cannot_be_declared(self) -> None:
        schema = mp.io.mrec.SequenceSchema.from_frames([])
        with pytest.raises(ValueError, match="atomi"):
            schema.declare_column("bonds", "atomi", "u32")
        schema.declare_column("bonds", "atomi", "u64")
        schema.declare_column("bonds", "label", "u32")

    def test_a_narrow_insert_is_widened_and_round_trips_as_u64(
        self, tmp_path: Path
    ) -> None:
        bonds = mp.core.Block()
        bonds["atomi"] = np.array([0, 1], dtype=np.uint32)
        bonds["atomj"] = np.array([1, 2], dtype=np.uint32)
        frame = _coords_frame()
        frame["bonds"] = bonds
        path = tmp_path / "bonds.mrec"
        mp.io.write_mrec(path, frame=frame)
        back = _frame_of(path)
        assert np.asarray(back["bonds"]["atomi"]).dtype == np.uint64


class TestUnknownSections:
    def test_an_unknown_root_section_is_ignored(self, tmp_path: Path) -> None:
        import json
        import shutil

        path = tmp_path / "foreign.mrec"
        mp.io.write_mrec(path, frame=_coords_frame())
        # A section this build does not know, which would not even decode as
        # a frame group: its block claims more rows than its columns hold.
        shutil.copytree(path / "frame", path / "future")
        block_json = path / "future" / "atoms" / "zarr.json"
        doc = json.loads(block_json.read_text(encoding="utf-8"))
        doc.setdefault("attributes", {})["count"] = 99
        block_json.write_text(json.dumps(doc), encoding="utf-8")

        assert "future" in _record(path).sections()
        _assert_coords(_frame_of(path))
        # The record still holds no trajectory: that is named, not misread.
        assert "trajectory" not in _record(path).sections()
        with pytest.raises(ValueError, match="trajectory"):
            mp.io.mrec.MrecReader(path)


class TestMetaArgument:
    """``meta=`` takes back every form ``frame.meta`` hands out."""

    @staticmethod
    def _frame_with_meta() -> mp.core.Frame:
        frame = _coords_frame()
        frame.meta["run"] = {"engine": "md", "seeds": [1, 2]}
        frame.meta["cell"] = [1.0, 2.0, 3.0]
        return frame

    def test_write_accepts_a_document_and_tuples(self, tmp_path: Path) -> None:
        frame = self._frame_with_meta()
        run = frame.meta["run"]
        assert isinstance(run, mp.core.MetaDocument)
        assert isinstance(frame.meta["cell"], tuple)

        path = tmp_path / "doc.mrec"
        mp.io.write_mrec(path, frame=frame, meta=run)
        meta = _record(path).meta()
        assert meta["engine"] == "md"
        assert meta["seeds"] == [1, 2]

        path = tmp_path / "nested.mrec"
        mp.io.write_mrec(
            path, frame=frame, meta={"run": run, "cell": frame.meta["cell"]}
        )
        meta = _record(path).meta()
        assert meta["run"] == {"engine": "md", "seeds": [1, 2]}
        assert meta["cell"] == [1.0, 2.0, 3.0]

    def test_write_accepts_frame_meta_itself(self, tmp_path: Path) -> None:
        frame = self._frame_with_meta()
        path = tmp_path / "frame_meta.mrec"
        mp.io.write_mrec(path, system=frame, meta=frame.meta)
        meta = _record(path).meta()
        assert meta["run"]["engine"] == "md"
        assert meta["cell"] == [1.0, 2.0, 3.0]

    def test_write_trajectory_takes_meta(self, tmp_path: Path) -> None:
        frame = self._frame_with_meta()
        path = tmp_path / "traj.mrec"
        mp.io.write_mrec(
            path, trajectory=mp.core.Trajectory([frame]), meta=frame.meta["run"]
        )
        meta = _record(path).meta()
        assert meta == {"engine": "md", "seeds": [1, 2]}
        assert len(_trajectory_of(path)) == 1

    def test_frame_sequence_writer_accepts_a_document(self, tmp_path: Path) -> None:
        frame = self._frame_with_meta()
        path = tmp_path / "stream.mrec"
        schema = mp.io.mrec.SequenceSchema.from_frames([frame])
        with mp.io.mrec.MrecWriter(
            path, schema, meta={"run": frame.meta["run"], "cell": frame.meta["cell"]}
        ) as writer:
            writer.append(frame)
        meta = _record(path).meta()
        assert meta["run"]["seeds"] == [1, 2]
        assert meta["cell"] == [1.0, 2.0, 3.0]

    def test_a_non_mapping_meta_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="mapping"):
            mp.io.write_mrec(tmp_path / "bad.mrec", frame=_coords_frame(), meta=[1, 2])


class TestDeclaredPrecision:
    """molrec F1: a precision rounds an f64 column onto a binary grid."""

    _VALUES = np.array([0.123456789, -12.3456, 39.99991, 1e-7])
    _Q = 2.0**-10  # quantum of p = 1e-3

    def _stored(self) -> np.ndarray:
        return np.round(self._VALUES / self._Q) * self._Q

    def _frame(self, precision: float | None = 1e-3) -> mp.core.Frame:
        atoms = mp.core.Block({"x": self._VALUES.copy()})
        atoms.set_precision("x", precision)
        frame = mp.core.Frame()
        frame["atoms"] = atoms
        return frame

    def test_block_declares_and_withdraws(self) -> None:
        block = mp.core.Block({"x": self._VALUES.copy(), "n": np.array([1, 2, 3, 4])})
        assert block.precision("x") is None
        block.set_precision("x", 1e-3)
        assert block.precision("x") == 1e-3
        block.set_precision("x", None)
        assert block.precision("x") is None
        with pytest.raises(ValueError):
            block.set_precision("n", 1e-3)
        with pytest.raises(ValueError):
            block.set_precision("x", 0.0)
        with pytest.raises(KeyError):
            block.set_precision("nope", 1e-3)
        with pytest.raises(KeyError):
            block.precision("nope")

    def test_frame_round_trip_rounds_and_keeps_the_declaration(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "p.mrec"
        frame = self._frame()
        mp.io.write_mrec(path, frame=frame)
        # Memory is untouched; the store holds the rounded values.
        np.testing.assert_array_equal(np.asarray(frame["atoms"]["x"]), self._VALUES)
        back = _frame_of(path)["atoms"]
        np.testing.assert_array_equal(np.asarray(back["x"]), self._stored())
        assert back.precision("x") == 1e-3

    def test_trajectory_pins_the_precision(self, tmp_path: Path) -> None:
        path = tmp_path / "t.mrec"
        mp.io.write_mrec(path, trajectory=mp.core.Trajectory([self._frame()]))
        reader = mp.io.mrec.MrecReader(path)
        atoms = reader[0]["atoms"]
        np.testing.assert_array_equal(np.asarray(atoms["x"]), self._stored())
        assert atoms.precision("x") == 1e-3

    def test_sequence_schema_declares_a_precision(self, tmp_path: Path) -> None:
        schema = mp.io.mrec.SequenceSchema.from_frames([self._frame(None)])
        assert schema.precision("atoms", "x") is None
        schema.declare_precision("atoms", "x", 1e-3)
        assert schema.precision("atoms", "x") == 1e-3
        with pytest.raises(ValueError):
            schema.declare_precision("atoms", "x", 1e-2)
        path = tmp_path / "w.mrec"
        with mp.io.mrec.MrecWriter(path, schema) as writer:
            writer.append(self._frame(None))
        frame = mp.io.mrec.MrecReader(path)[0]
        np.testing.assert_array_equal(np.asarray(frame["atoms"]["x"]), self._stored())

    def test_pickle_keeps_the_precision(self) -> None:
        import pickle

        block = pickle.loads(pickle.dumps(self._frame()["atoms"]))
        assert block.precision("x") == 1e-3


class TestTypedFrameMeta:
    """molrec F3: a frame's meta reads back at its tag, NaN included."""

    def test_every_value_keeps_its_tag(self, tmp_path: Path) -> None:
        path = tmp_path / "m.mrec"
        frame = _coords_frame()
        frame.meta["n32"] = mp.core.MetaValue("i32", 7)
        frame.meta["big"] = 2**64 - 1
        frame.meta["one"] = 1.0
        frame.meta["nan"] = float("nan")
        frame.meta["inf"] = float("-inf")
        frame.meta["vec"] = mp.core.MetaValue("f64x3", (1.0, float("inf"), 2.0))
        frame.meta["doc"] = {"a": [1, 2]}
        mp.io.write_mrec(path, frame=frame)
        meta = _frame_of(path).meta
        assert meta.dtype("n32") == "i32" and meta["n32"] == 7
        assert meta.dtype("big") == "u64" and meta["big"] == 2**64 - 1
        assert meta.dtype("one") == "f64" and meta["one"] == 1.0
        assert meta.dtype("nan") == "f64" and np.isnan(meta["nan"])
        assert meta["inf"] == float("-inf")
        assert meta.dtype("vec") == "f64x3" and meta["vec"][1] == float("inf")
        assert meta.dtype("doc") == "json"
        assert "_meta_types" not in meta

    def test_a_non_finite_number_inside_a_document_is_refused(self) -> None:
        frame = mp.core.Frame()
        with pytest.raises(ValueError):
            frame.meta["doc"] = {"t": float("nan")}

    def test_a_nan_fill_survives_the_pin(self, tmp_path: Path) -> None:
        schema = mp.io.mrec.SequenceSchema.from_frames([_coords_frame()])
        schema.declare_meta("temp", "f64", float("nan"))
        path = tmp_path / "f.mrec"
        with mp.io.mrec.MrecWriter(path, schema) as writer:
            writer.append(_coords_frame())
        frame = mp.io.mrec.MrecReader(path)[0]
        assert np.isnan(frame.meta["temp"])

    def test_a_declared_meta_key_without_a_fill_is_required(
        self, tmp_path: Path
    ) -> None:
        schema = mp.io.mrec.SequenceSchema.from_frames([_coords_frame()])
        schema.declare_meta("temp", "f64")
        assert ("temp", "f64") in schema.meta_keys()
        writer = mp.io.mrec.MrecWriter(tmp_path / "g.mrec", schema)
        with pytest.raises(ValueError, match="temp"):
            writer.append(_coords_frame())
        writer.close()

    def test_a_fill_is_read_at_the_declared_dtype(self, tmp_path: Path) -> None:
        schema = mp.io.mrec.SequenceSchema.from_frames([_coords_frame()])
        schema.declare_meta("origin", "f64x3", (0.0, 1.0, 2.0))
        path = tmp_path / "h.mrec"
        with mp.io.mrec.MrecWriter(path, schema) as writer:
            writer.append(_coords_frame())
        meta = mp.io.mrec.MrecReader(path)[0].meta
        assert meta.dtype("origin") == "f64x3"
        assert tuple(meta["origin"]) == (0.0, 1.0, 2.0)


class TestRowReferences:
    """molrec F4: declared targets persist and are held to their rows."""

    def _frame(self, ibead: list[int]) -> mp.core.Frame:
        frame = _coords_frame()
        members = mp.core.Block(
            {
                "ibead": np.array(ibead, dtype=np.uint64),
                "atom": np.array([1] * len(ibead), dtype=np.uint64),
            }
        )
        members.set_target("atom", "/frame/atoms")
        frame["members"] = members
        return frame

    def test_targets_round_trip_and_renumber(self, tmp_path: Path) -> None:
        frame = self._frame([0, 2])
        assert frame["members"].targets() == {"atom": "/frame/atoms"}
        path = tmp_path / "t.mrec"
        mp.io.write_mrec(path, frame=frame)
        back = _frame_of(path)["members"]
        assert back.target("atom") == "/frame/atoms"
        assert back.target("ibead") is None
        with pytest.raises(ValueError):
            back.set_target("atom", "/trajectory/atoms")

    def test_a_broken_reference_is_refused(self, tmp_path: Path) -> None:
        frame = self._frame([5])
        frame["members"].set_target("ibead", "atoms")
        with pytest.raises(ValueError, match="atoms"):
            mp.io.write_mrec(tmp_path / "bad.mrec", frame=frame)

    def test_sequence_schema_declares_a_target(self) -> None:
        schema = mp.io.mrec.SequenceSchema.from_frames([self._frame([0])])
        assert schema.target("members", "atom") == "/frame/atoms"
        with pytest.raises(ValueError):
            schema.declare_target("members", "atom", "atoms")


class TestAlignedBlocks:
    """molrec F5: an aligned block keeps its target's row count."""

    def _frame(self, n: int, types: list[str] | None) -> mp.core.Frame:
        frame = mp.core.Frame()
        frame["atoms"] = mp.core.Block({"x": np.arange(n, dtype=np.float64)})
        if types is not None:
            frame["atom_types"] = mp.core.Block({"type": np.array(types)})
        return frame

    def _schema(self) -> mp.io.mrec.SequenceSchema:
        schema = mp.io.mrec.SequenceSchema.from_frames([])
        schema.declare_column("atoms", "x", "f64")
        schema.declare_column("atom_types", "type", "string")
        return schema.declare_aligned("atom_types", "atoms")

    def test_carry_forward_and_restate_on_growth(self, tmp_path: Path) -> None:
        path = tmp_path / "a.mrec"
        with mp.io.mrec.MrecWriter(path, self._schema()) as writer:
            writer.append(self._frame(2, ["A", "B"]))
            writer.append(self._frame(2, None))
            writer.append(self._frame(3, ["A", "B", "C"]))
        reader = mp.io.mrec.MrecReader(path)
        assert list(reader[1]["atom_types"]["type"]) == ["A", "B"]
        assert reader[2]["atom_types"].n_rows == 3

    def test_a_frame_that_breaks_the_alignment_is_refused(self, tmp_path: Path) -> None:
        schema = self._schema()
        assert schema.aligned_with("atom_types") == "atoms"
        writer = mp.io.mrec.MrecWriter(tmp_path / "b.mrec", schema)
        writer.append(self._frame(2, ["A", "B"]))
        with pytest.raises(ValueError):
            writer.append(self._frame(3, None))
        writer.close()
        with pytest.raises(ValueError):
            mp.io.mrec.SequenceSchema.from_frames([]).declare_aligned("a", "b")


class TestForceFieldSection:
    """The ``forcefield`` root section (molrec ``forcefield.md``)."""

    @staticmethod
    def _forcefield() -> mp.ff.forcefield.ForceField:
        ff = mp.ff.forcefield.ForceField("water", units="real")
        atoms = ff.def_style("atom", "full")
        ow = atoms.def_type("OW", mass=15.999, charge=-0.834)
        hw = atoms.def_type("HW", mass=1.008, charge=0.417)
        ff.def_style("bond", "harmonic").def_type(
            "OW-HW", ow, hw, k=1059.162, r0=0.9572
        )
        ff.def_style(
            "pair", "lj/cut", {"cutoff": 10.0, "mixing": "geometric"}
        ).def_type("OW", ow, epsilon=0.1521, sigma=3.1507)
        ff.set_special_bonds([0.0, 0.0, 0.5], [0.0, 0.0, 0.8333])
        return ff

    def test_a_force_field_round_trips_through_its_section(
        self, tmp_path: Path
    ) -> None:
        ff = self._forcefield()
        section = mp.io.mrec.ForceFieldSection.from_forcefield(ff)
        assert isinstance(section, mp.io.mrec.ForceFieldSection)
        assert section.name == "water"
        assert section.document["units"]["preset"] == "real"
        assert section.document["styles"][2]["params"] == {
            "cutoff": 10.0,
            "mixing": "geometric",
        }
        assert sorted(section.tables) == ["atom.full", "bond.harmonic", "pair.lj%2Fcut"]

        path = tmp_path / "ff.mrec"
        mp.io.write_mrec(path, forcefield=ff, meta={"producer": "test"})
        assert "forcefield" in _record(path).sections()
        back = _forcefield_of(path)
        assert back.document == section.document
        bonds = back.table("bond", "harmonic")
        assert list(bonds["name"]) == ["OW-HW"]
        assert bonds["k"][0] == 1059.162
        again = back.to_forcefield()
        assert again.name == "water"
        assert again.get_style("pair", "lj/cut") is not None

    def test_a_section_is_kept_whole_units_unconverted(self, tmp_path: Path) -> None:
        bonds = mp.core.Block(
            {
                "name": np.array(["CT-HC"]),
                "itom": np.array(["CT"]),
                "jtom": np.array(["HC"]),
                "r0": np.array([0.1090]),
                "k": np.array([284512.0]),
            }
        )
        notes = mp.core.Block({"text": np.array(["kept"])})
        document = {
            "name": "nm",
            "units": {"length": "nm", "energy": "kJ/mol", "angle": "radian"},
            "styles": [{"category": "bond", "style": "harmonic"}],
            "aromaticity_model": "OEAroModel_MDL",
        }
        section = mp.io.mrec.ForceFieldSection(
            document, {"bond.harmonic": bonds, "notes.free%20text": notes}
        )
        section.validate()
        path = tmp_path / "nm.mrec"
        mp.io.write_mrec(path, system=_coords_frame(), forcefield=section)
        back = _forcefield_of(path)
        assert back.document == document
        assert sorted(back.tables) == ["bond.harmonic", "notes.free%20text"]
        assert back.table("bond", "harmonic")["k"][0] == 284512.0
        # nm / kJ/mol is no molrs preset: refused, not converted.
        with pytest.raises(ValueError, match="units"):
            back.to_forcefield()

    def test_no_forcefield_reads_none_and_a_bad_one_is_refused(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "frame.mrec"
        mp.io.write_mrec(path, frame=_coords_frame())
        assert _record(path).forcefield() is None
        section = mp.io.mrec.ForceFieldSection({"name": "x", "units": {}})
        with pytest.raises(ValueError, match="units"):
            section.validate()
        with pytest.raises(ValueError):
            mp.io.write_mrec(path, frame=_coords_frame(), forcefield=section)
        with pytest.raises(TypeError):
            mp.io.write_mrec(path, frame=_coords_frame(), forcefield={"name": "x"})

    def test_a_pair_table_prices_each_unordered_pair_once(self) -> None:
        """molrec forcefield linking rule 3: B-A restating A-B is one row when
        the parameters agree (name and annotations aside), a refusal when not."""

        def section(
            epsilon: list[float], desc: list[str]
        ) -> mp.io.mrec.ForceFieldSection:
            rows = mp.core.Block(
                {
                    "name": np.array(["A", "B", "A-B", "B-A"]),
                    "itom": np.array(["A", "B", "A", "B"]),
                    "jtom": np.array(["A", "B", "B", "A"]),
                    "epsilon": np.array(epsilon),
                    "sigma": np.array([3.0, 3.6, 2.0, 2.0]),
                    "desc": np.array(desc),
                }
            )
            document = {
                "name": "nbfix",
                "units": {"preset": "real"},
                "styles": [{"category": "pair", "style": "lj/cut"}],
            }
            return mp.io.mrec.ForceFieldSection(document, {"pair.lj%2Fcut": rows})

        section([0.1, 0.4, 0.9, 0.9], ["a", "b", "nbfix", "restated"]).validate()
        conflict = section([0.1, 0.4, 0.9, 0.8], ["a", "b", "c", "d"])
        with pytest.raises(ValueError, match=r'"A-B" and "B-A".*epsilon'):
            conflict.validate()
        with pytest.raises(ValueError, match="epsilon"):
            conflict.to_forcefield()

    def test_block_name_percent_encodes_the_style(self) -> None:
        name = mp.io.mrec.ForceFieldSection.block_name("pair", "lj/cut/coul/long")
        assert name == "pair.lj%2Fcut%2Fcoul%2Flong"


def test_packed_trajectory_keeps_portable_section_paths(tmp_path: Path) -> None:
    import zipfile

    path = tmp_path / "packed.mrec"
    mp.io.write_mrec(path, trajectory=mp.core.Trajectory([_coords_frame()]))
    packed = Path(mp.io.mrec.pack_mrec_zip(path))
    with zipfile.ZipFile(packed) as archive:
        names = archive.namelist()
        assert any(name.startswith("trajectory/") for name in names)
        assert all("\\" not in name for name in names)
    assert "trajectory" in _record(packed).sections()
    reader = mp.io.mrec.MrecReader(packed)
    assert len(reader) == 1
    _assert_coords(reader[0])
