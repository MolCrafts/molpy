"""Tests for the lazy multi-frame readers and the frame writers of
``molrs.io`` (``molrs.io.<fmt>.<Fmt>Reader`` / ``<Fmt>Writer``).

Self-contained fixtures written by molrs. No external corpus.
"""

from __future__ import annotations

import molrs
import numpy as np
import pytest
from molrs.core import Box, Cell


def _cube(length: float) -> Box:
    return Box(Cell.lengths([length, length, length]))


class TestTheFormatsReader:
    """``molrs.io.<fmt>.<Fmt>Reader(source)`` is that format's lazy reader."""

    def test_dcd_reader(self, water_dcd):
        reader = molrs.io.dcd.DcdReader(water_dcd)
        assert isinstance(reader, molrs.io.dcd.DcdReader)
        assert len(reader) > 0

    def test_lammps_reader(self, water_lammpstrj):
        reader = molrs.io.lammps.LammpsDumpReader(water_lammpstrj)
        assert isinstance(reader, molrs.io.lammps.LammpsDumpReader)
        assert len(reader) > 0

    def test_xyz_reader_reads_all_into_a_list(self, water_xyz):
        reader = molrs.io.xyz.XyzReader(water_xyz)
        eager = list(molrs.io.xyz.XyzReader(water_xyz))
        assert isinstance(eager, list)
        assert len(reader) == len(eager)

    def test_a_reader_reads_bytes_as_its_path(self, water_dcd):
        from_bytes = molrs.io.dcd.DcdReader(water_dcd.read_bytes())
        from_path = molrs.io.dcd.DcdReader(water_dcd)
        assert len(from_bytes) == len(from_path)
        np.testing.assert_allclose(
            from_bytes[-1]["atoms"]["x"], from_path[-1]["atoms"]["x"]
        )

    def test_a_reader_takes_one_source(self, water_dcd):
        # Several files are no longer concatenated: the core reader opens one.
        with pytest.raises(TypeError):
            molrs.io.dcd.DcdReader([water_dcd, water_dcd])
        with pytest.raises(TypeError, match=r"pathlib\.Path"):
            molrs.io.dcd.DcdReader(str(water_dcd))


class TestTrajectoryReaderSurface:
    def test_index_and_negative_index(self, water_dcd):
        reader = molrs.io.dcd.DcdReader(water_dcd)
        n = len(reader)
        assert reader[0] is not None
        assert reader[-1]["atoms"].n_rows == reader[n - 1]["atoms"].n_rows

    def test_out_of_range_raises(self, water_dcd):
        reader = molrs.io.dcd.DcdReader(water_dcd)
        with pytest.raises(IndexError):
            reader[10_000_000]

    def test_list_and_slices_read_every_frame(self, water_dcd):
        reader = molrs.io.dcd.DcdReader(water_dcd)
        n = len(reader)
        assert len(list(reader)) == n
        assert len(reader[:]) == n
        assert len(reader[0:n]) == n

    def test_slice_step_zero_raises(self, water_dcd):
        reader = molrs.io.dcd.DcdReader(water_dcd)
        with pytest.raises(ValueError):
            reader[0:1:0]

    def test_slicing(self, water_dcd):
        reader = molrs.io.dcd.DcdReader(water_dcd)
        n = len(reader)
        assert isinstance(reader[0:1], list)
        assert len(reader[:]) == n
        assert len(reader[::-1]) == n

    def test_iteration(self, water_dcd):
        reader = molrs.io.dcd.DcdReader(water_dcd)
        assert sum(1 for _ in reader) == len(reader)
        assert sum(1 for _ in reader) == len(reader)

    def test_context_manager_closes(self, water_dcd):
        with molrs.io.dcd.DcdReader(water_dcd) as reader:
            assert len(reader) > 0
        with pytest.raises(ValueError):
            reader[0]


class TestFrameWriters:
    def test_a_buffer_holds_the_file_after_each_write(self, tmp_path, water_xyz):
        frame = molrs.io.read_xyz(water_xyz)
        path = tmp_path / "two.dcd"
        buf = bytearray(b"stale")
        with molrs.io.dcd.DcdWriter(path) as on_disk, molrs.io.dcd.DcdWriter(buf) as in_memory:
            assert buf == b""
            for _ in range(2):
                on_disk.write(frame)
                in_memory.write(frame)
                assert len(molrs.io.dcd.DcdReader(bytes(buf))) == len(
                    molrs.io.dcd.DcdReader(path)
                )
        assert path.read_bytes() == bytes(buf)
        assert len(molrs.io.dcd.DcdReader(bytes(buf))) == 2

    def test_a_closed_writer_refuses_writes(self, tmp_path, water_xyz):
        frame = molrs.io.read_xyz(water_xyz)
        writer = molrs.io.xyz.XyzWriter(tmp_path / "a.xyz")
        writer.write(frame)
        writer.close()
        writer.close()
        with pytest.raises(ValueError, match="closed"):
            writer.write(frame)


class TestDumpLocalWrite:
    def test_write_bonds_roundtrip(self, tmp_path):
        atoms = molrs.core.Block()
        atoms["id"] = np.array([1, 2, 3], dtype=np.uint64)
        atoms["x"] = np.array([0.0, 1.0, 2.0])
        atoms["y"] = np.zeros(3)
        atoms["z"] = np.zeros(3)
        bonds = molrs.core.Block()
        bonds["atomi"] = np.array([0, 1], dtype=np.uint64)
        bonds["atomj"] = np.array([1, 2], dtype=np.uint64)
        frame = molrs.core.Frame()
        frame["atoms"] = atoms
        frame["bonds"] = bonds
        frame.box = _cube(10.0)
        path = tmp_path / "bonds.dump.local"
        molrs.io.write_lammps_dump_local(path, [frame])
        text = path.read_text(encoding="utf-8")
        assert "ITEM: NUMBER OF ENTRIES" in text
        assert "batom1 batom2" in text
        loaded = list(molrs.io.lammps.LammpsDumpReader(path))
        assert loaded[0]["entries"].n_rows == 2


class TestDumpColumnChoice:
    @staticmethod
    def _frame():
        atoms = molrs.core.Block()
        atoms["id"] = np.array([1, 2], dtype=np.uint64)
        atoms["mol_id"] = np.array([1, 1], dtype=np.uint64)
        atoms["mass"] = np.array([16.0, 1.008])
        atoms["element"] = ["O", "H"]
        atoms["x"] = np.array([0.0, 1.0])
        atoms["y"] = np.array([0.0, 2.0])
        atoms["z"] = np.array([0.0, 3.0])
        frame = molrs.core.Frame()
        frame["atoms"] = atoms
        frame.box = _cube(10.0)
        return frame

    def test_writes_only_the_named_columns_in_order(self, tmp_path):
        path = tmp_path / "chosen.lammpstrj"
        molrs.io.write_lammps_dump(
            path, self._frame(), columns=["id", "element", "mol", "x", "y", "z"]
        )
        text = path.read_text(encoding="utf-8")
        assert "ITEM: ATOMS id element mol x y z" in text
        assert "mass" not in text

    def test_the_writer_class_writes_the_named_columns(self, tmp_path):
        path = tmp_path / "chosen.lammpstrj"
        with molrs.io.lammps.LammpsDumpWriter(path) as writer:
            writer.set_columns(["id", "element", "mol", "x", "y", "z"])
            writer.write(self._frame())
            writer.write(self._frame())
        text = path.read_text(encoding="utf-8")
        assert text.count("ITEM: ATOMS id element mol x y z") == 2
        assert "mass" not in text

    def test_default_writes_every_column(self, tmp_path):
        path = tmp_path / "all.lammpstrj"
        molrs.io.write_lammps_dump(path, self._frame())
        assert "ITEM: ATOMS id element mass mol x y z" in path.read_text(encoding="utf-8")

    def test_rejects_a_column_the_frame_lacks(self, tmp_path):
        path = tmp_path / "missing.lammpstrj"
        with pytest.raises(OSError, match="'q'"):
            molrs.io.write_lammps_dump(path, self._frame(), columns=["id", "q"])


class TestDumpTypeField:
    @staticmethod
    def _atoms(**extra):
        return {
            "id": [1, 2, 3],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.5, 0.0],
            "z": [0.0, 0.0, 0.25],
            **extra,
        }

    def test_string_type_labels_round_trip(self, tmp_path):
        path = tmp_path / "labels.lammpstrj"
        frame = molrs.core.Frame(
            {"atoms": self._atoms(type=["OW", "HW", "HW"])}, box=_cube(10.0)
        )
        molrs.io.write_lammps_dump(path, frame)
        text = path.read_text(encoding="utf-8")
        assert "ITEM: ATOMS id type x y z\n1 OW " in text
        atoms = molrs.io.lammps.LammpsDumpReader(path)[0]["atoms"]
        assert list(atoms["type"]) == ["OW", "HW", "HW"]
        assert "type_id" not in atoms

    def test_type_id_wins_the_type_field(self, tmp_path):
        path = tmp_path / "both.lammpstrj"
        frame = molrs.core.Frame(
            {"atoms": self._atoms(type=["OW", "HW", "HW"], type_id=[1, 2, 2])},
            box=_cube(10.0),
        )
        molrs.io.write_lammps_dump(path, frame)
        text = path.read_text(encoding="utf-8")
        assert "ITEM: ATOMS id type x y z\n1 1 " in text
        atoms = molrs.io.lammps.LammpsDumpReader(path)[0]["atoms"]
        assert list(atoms["type_id"]) == [1, 2, 2]
        assert "type" not in atoms


class TestPdbAndGroReaders:
    """PDB models and GRO frames are read lazily too, one class per format."""

    @staticmethod
    def _frames():
        out = []
        for shift in (0.0, 1.0):
            frame = molrs.core.Frame(
                {
                    "atoms": {
                        "element": ["O", "H", "H"],
                        "name": ["OW", "HW1", "HW2"],
                        "x": [0.0 + shift, 0.96, -0.24],
                        "y": [0.0, 0.0, 0.93],
                        "z": [0.0, 0.0, 0.0],
                    }
                },
                box=_cube(10.0),
            )
            out.append(frame)
        return out

    def test_pdb_models_by_index(self, tmp_path):
        path = tmp_path / "two.pdb"
        with molrs.io.pdb.PdbWriter(path) as writer:
            for frame in self._frames():
                writer.write(frame)
        reader = molrs.io.pdb.PdbReader(path)
        assert isinstance(reader, molrs.io.pdb.PdbReader)
        assert len(reader) == 2
        np.testing.assert_allclose(reader[-1]["atoms"]["x"][0], 1.0, atol=1e-3)

    def test_gro_frames_by_index(self, tmp_path):
        path = tmp_path / "two.gro"
        with molrs.io.gro.GroWriter(path) as writer:
            for frame in self._frames():
                writer.write(frame)
        reader = molrs.io.gro.GroReader(path)
        assert isinstance(reader, molrs.io.gro.GroReader)
        assert [f["atoms"].n_rows for f in reader] == [3, 3]
        np.testing.assert_allclose(reader[1]["atoms"]["x"][0], 1.0, atol=1e-2)


class TestCanonicalFields:
    def test_lammps_canonical_columns(self, water_lammpstrj):
        reader = molrs.io.lammps.LammpsDumpReader(water_lammpstrj)
        atoms = reader[0]["atoms"]
        assert "q" not in atoms
