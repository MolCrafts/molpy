"""Tests for trajectory index data structures, serialization, and stale detection."""

import time

import numpy as np
import pytest

import molpy as mp
from molpy.io.trajectory.index import (
    FileFingerprint,
    FrameEntry,
    TrajectoryIndex,
    _INDEX_VERSION,
    build_index,
    create_fingerprint,
    deserialize_index,
    index_file_path,
    is_index_fresh,
    load_index,
    save_index,
    serialize_index,
)


class TestFrameEntry:
    def test_immutable(self):
        entry = FrameEntry(file_idx=0, byte_offset=100, byte_length=500, timestep=42)
        with pytest.raises(AttributeError):
            entry.byte_offset = 200

    def test_fields(self):
        entry = FrameEntry(
            file_idx=1, byte_offset=1024, byte_length=2048, timestep=None
        )
        assert entry.file_idx == 1
        assert entry.byte_offset == 1024
        assert entry.byte_length == 2048
        assert entry.timestep is None


class TestFileFingerprint:
    def test_immutable(self):
        fp = FileFingerprint(
            path_relative="traj.dump",
            path_absolute="/tmp/traj.dump",
            size_bytes=1000,
            mtime_ns=123456789,
            header_hash="abc123",
        )
        with pytest.raises(AttributeError):
            fp.size_bytes = 2000

    def test_create_fingerprint(self, tmp_path):
        # Write a test file
        traj = tmp_path / "test.dump"
        traj.write_text("ITEM: TIMESTEP\n0\n")

        fp = create_fingerprint(traj, tmp_path)
        assert fp.path_relative == "test.dump"
        assert fp.path_absolute == str(traj.resolve())
        assert fp.size_bytes == traj.stat().st_size
        assert fp.mtime_ns == traj.stat().st_mtime_ns
        assert len(fp.header_hash) == 64  # SHA-256 hex


class TestTrajectoryIndex:
    def _make_index(self, n_frames=3) -> TrajectoryIndex:
        files = (FileFingerprint("a.dump", "/tmp/a.dump", 1000, 111, "hash_a"),)
        frames = tuple(
            FrameEntry(
                file_idx=0, byte_offset=i * 100, byte_length=100, timestep=i * 10
            )
            for i in range(n_frames)
        )
        return TrajectoryIndex(
            version=_INDEX_VERSION,
            format_id="lammps-dump",
            files=files,
            frames=frames,
            created_utc="2026-01-01T00:00:00+00:00",
        )

    def test_n_frames(self):
        idx = self._make_index(5)
        assert idx.n_frames == 5

    def test_frame_for_timestep(self):
        idx = self._make_index(10)
        entry = idx.frame_for_timestep(30)
        assert entry is not None
        assert entry.timestep == 30
        assert entry.byte_offset == 300

    def test_frame_for_timestep_not_found(self):
        idx = self._make_index(3)
        assert idx.frame_for_timestep(999) is None

    def test_immutable(self):
        idx = self._make_index()
        with pytest.raises(AttributeError):
            idx.version = 2


class TestSerialization:
    def _make_index(self) -> TrajectoryIndex:
        files = (
            FileFingerprint("a.dump", "/tmp/a.dump", 5000, 999, "abcdef1234"),
            FileFingerprint("b.dump", "/tmp/b.dump", 3000, 888, "fedcba4321"),
        )
        frames = (
            FrameEntry(0, 0, 500, 0),
            FrameEntry(0, 500, 600, 100),
            FrameEntry(1, 0, 400, 200),
        )
        return TrajectoryIndex(
            version=_INDEX_VERSION,
            format_id="lammps-dump",
            files=files,
            frames=frames,
            created_utc="2026-03-18T10:00:00+00:00",
        )

    def test_roundtrip(self):
        original = self._make_index()
        data = serialize_index(original)
        restored = deserialize_index(data)

        assert restored.version == original.version
        assert restored.format_id == original.format_id
        assert restored.n_frames == original.n_frames
        assert restored.files == original.files
        assert restored.frames == original.frames

    def test_save_and_load(self, tmp_path):
        original = self._make_index()
        path = tmp_path / "test.tridx.json"
        save_index(original, path)
        assert path.exists()

        loaded = load_index(path)
        assert loaded is not None
        assert loaded.frames == original.frames
        assert loaded.files == original.files

    def test_load_corrupted(self, tmp_path):
        path = tmp_path / "bad.tridx.json"
        path.write_text("not valid json at all")
        result = load_index(path)
        assert result is None

    def test_load_nonexistent(self, tmp_path):
        result = load_index(tmp_path / "does_not_exist.tridx.json")
        assert result is None


class TestStalenessDetection:
    def _write_traj(self, path, content="ITEM: TIMESTEP\n0\n"):
        path.write_text(content)
        return path

    def test_fresh_index(self, tmp_path):
        traj = self._write_traj(tmp_path / "traj.dump")
        idx = build_index("test", [traj], [FrameEntry(0, 0, 100, 0)])
        assert is_index_fresh(idx, [traj])

    def test_stale_after_size_change(self, tmp_path):
        traj = self._write_traj(tmp_path / "traj.dump", "short")
        idx = build_index("test", [traj], [FrameEntry(0, 0, 5, 0)])

        # Modify file (changes size)
        traj.write_text("much longer content now")
        assert not is_index_fresh(idx, [traj])

    def test_stale_after_content_change(self, tmp_path):
        content_a = "A" * 100
        content_b = "B" * 200  # different length -> size_bytes mismatch
        traj = self._write_traj(tmp_path / "traj.dump", content_a)
        idx = build_index("test", [traj], [FrameEntry(0, 0, 100, 0)])

        # Write different-length content so size_bytes detects staleness
        traj.write_text(content_b)
        assert not is_index_fresh(idx, [traj])

    def test_fresh_after_touch(self, tmp_path):
        content = "X" * 200
        traj = self._write_traj(tmp_path / "traj.dump", content)
        idx = build_index("test", [traj], [FrameEntry(0, 0, 200, 0)])

        # Touch the file (changes mtime but not content)
        time.sleep(0.05)
        traj.touch()
        # Header hash should match -> still fresh
        assert is_index_fresh(idx, [traj])

    def test_stale_version_mismatch(self, tmp_path):
        traj = self._write_traj(tmp_path / "traj.dump")
        idx = build_index("test", [traj], [FrameEntry(0, 0, 100, 0)])
        # Forge a version mismatch
        old_idx = TrajectoryIndex(
            version=0,  # old version
            format_id=idx.format_id,
            files=idx.files,
            frames=idx.frames,
            created_utc=idx.created_utc,
        )
        assert not is_index_fresh(old_idx, [traj])

    def test_stale_file_count_mismatch(self, tmp_path):
        traj = self._write_traj(tmp_path / "traj.dump")
        idx = build_index("test", [traj], [FrameEntry(0, 0, 100, 0)])
        # Add an extra file
        traj2 = self._write_traj(tmp_path / "traj2.dump")
        assert not is_index_fresh(idx, [traj, traj2])

    def test_stale_file_deleted(self, tmp_path):
        traj = self._write_traj(tmp_path / "traj.dump")
        idx = build_index("test", [traj], [FrameEntry(0, 0, 100, 0)])
        traj.unlink()
        assert not is_index_fresh(idx, [traj])


class TestIndexPersistence:
    """Test that readers auto-persist and reload indexes."""

    def test_lammps_index_persisted(self, tmp_path):
        """Test that LAMMPS reader creates and reuses an index file."""
        from molpy.io.trajectory.lammps import (
            LammpsTrajectoryReader,
            LammpsTrajectoryWriter,
        )

        # Write a small trajectory
        traj_file = tmp_path / "test.dump"
        writer = LammpsTrajectoryWriter(str(traj_file))
        for i in range(3):
            frame = mp.Frame()
            frame["atoms"] = {
                "id": [0, 1],
                "type": [1, 2],
                "x": [0.0, 1.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
            }
            frame.metadata["timestep"] = i * 100
            frame.box = mp.Box(np.eye(3) * 10.0)
            writer.write_frame(frame)
        writer.close()

        # First read: builds index
        reader1 = LammpsTrajectoryReader(str(traj_file))
        assert reader1.n_frames == 3
        frame0 = reader1.read_frame(0)
        assert frame0.metadata["timestep"] == 0
        reader1.close()

        # Index file should exist (reader uses {stem}_index.json format)
        idx_path = traj_file.parent / "test_index.json"
        assert idx_path.exists()

        # Second read: loads from index
        reader2 = LammpsTrajectoryReader(str(traj_file))
        assert reader2.n_frames == 3
        frame2 = reader2.read_frame(2)
        assert frame2.metadata["timestep"] == 200
        reader2.close()

    def test_index_rebuilt_after_modification(self, tmp_path):
        """Test that modifying trajectory file triggers index rebuild."""
        from molpy.io.trajectory.lammps import (
            LammpsTrajectoryReader,
            LammpsTrajectoryWriter,
        )

        traj_file = tmp_path / "test.dump"

        # Write 2 frames
        writer = LammpsTrajectoryWriter(str(traj_file))
        for i in range(2):
            frame = mp.Frame()
            frame["atoms"] = {
                "id": [0],
                "type": [1],
                "x": [0.0],
                "y": [0.0],
                "z": [0.0],
            }
            frame.metadata["timestep"] = i * 100
            frame.box = mp.Box(np.eye(3) * 5.0)
            writer.write_frame(frame)
        writer.close()

        # First read
        reader1 = LammpsTrajectoryReader(str(traj_file))
        assert reader1.n_frames == 2
        reader1.close()

        # Append a third frame (modifies file)
        writer = LammpsTrajectoryWriter.__new__(LammpsTrajectoryWriter)
        writer.fpath = traj_file
        writer._fp = open(traj_file, "ab")
        frame = mp.Frame()
        frame["atoms"] = {"id": [0], "type": [1], "x": [0.0], "y": [0.0], "z": [0.0]}
        frame.metadata["timestep"] = 200
        frame.box = mp.Box(np.eye(3) * 5.0)
        writer.write_frame(frame)
        writer.close()

        # Second read: should detect stale index and rebuild
        reader2 = LammpsTrajectoryReader(str(traj_file))
        assert reader2.n_frames == 3
        reader2.close()
