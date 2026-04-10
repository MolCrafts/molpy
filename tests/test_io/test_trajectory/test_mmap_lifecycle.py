"""Tests for MappedFile lifecycle, madvise hints, and resource cleanup."""

import numpy as np
import pytest

import molpy as mp
from molpy.io.trajectory.base import MappedFile
from molpy.io.trajectory.lammps import LammpsTrajectoryReader, LammpsTrajectoryWriter


class TestMappedFile:
    def test_open_and_read(self, tmp_path):
        """Test that MappedFile can open and read a file."""
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        mf = MappedFile.open_readonly(f)
        assert mf.mm[:5] == b"hello"
        mf.close()

    def test_empty_file_raises(self, tmp_path):
        """Test that opening an empty file raises ValueError."""
        f = tmp_path / "empty.txt"
        f.write_text("")
        with pytest.raises(ValueError, match="empty"):
            MappedFile.open_readonly(f)

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MappedFile.open_readonly(tmp_path / "no_such_file.txt")

    def test_fd_stays_alive(self, tmp_path):
        """Test that fd is kept alive while mmap is in use (Bug 2 fix)."""
        f = tmp_path / "data.bin"
        f.write_bytes(b"A" * 4096)
        mf = MappedFile.open_readonly(f)
        # The fd should be accessible (not closed by GC)
        assert not mf._fd.closed
        # mmap should be readable
        assert mf.mm[0:1] == b"A"
        mf.close()
        assert mf._fd.closed

    def test_close_releases_resources(self, tmp_path):
        """Test that close() releases both mmap and fd."""
        f = tmp_path / "data.bin"
        f.write_bytes(b"X" * 1024)
        mf = MappedFile.open_readonly(f)
        mm = mf.mm
        fd = mf._fd
        mf.close()
        assert fd.closed
        # mmap should be closed too (accessing it after close raises)
        with pytest.raises(ValueError):
            _ = mm[0]


class TestReaderResourceLifecycle:
    def _write_traj(self, path, n_frames=2):
        writer = LammpsTrajectoryWriter(str(path))
        for i in range(n_frames):
            frame = mp.Frame()
            frame["atoms"] = {
                "id": [0, 1],
                "type": [1, 2],
                "x": [float(i), float(i) + 1],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
            }
            frame.metadata["timestep"] = i * 100
            frame.box = mp.Box(np.eye(3) * 10.0)
            writer.write_frame(frame)
        writer.close()

    def test_context_manager_cleanup(self, tmp_path):
        """Test that context manager properly closes all resources."""
        traj = tmp_path / "test.dump"
        self._write_traj(traj)

        with LammpsTrajectoryReader(str(traj)) as reader:
            mapped_files = reader._mapped_files.copy()
            _ = reader.read_frame(0)

        # After context exit, all fds should be closed
        for mf in mapped_files:
            assert mf._fd.closed

    def test_explicit_close(self, tmp_path):
        """Test explicit close() releases resources."""
        traj = tmp_path / "test.dump"
        self._write_traj(traj)

        reader = LammpsTrajectoryReader(str(traj))
        mapped_files = reader._mapped_files.copy()
        reader.close()

        for mf in mapped_files:
            assert mf._fd.closed
        assert len(reader._mapped_files) == 0

    def test_multi_file_resource_management(self, tmp_path):
        """Test resource management with multiple files."""
        files = []
        for i in range(3):
            f = tmp_path / f"traj{i}.dump"
            self._write_traj(f)
            files.append(str(f))

        with LammpsTrajectoryReader(files) as reader:
            assert len(reader._mapped_files) == 3
            all_mfs = reader._mapped_files.copy()
            _ = reader.read_frame(0)

        for mf in all_mfs:
            assert mf._fd.closed

    def test_madvise_called_during_read(self, tmp_path):
        """Test that reading a frame doesn't crash (madvise may be no-op)."""
        traj = tmp_path / "test.dump"
        self._write_traj(traj, n_frames=5)

        reader = LammpsTrajectoryReader(str(traj))
        # Sequential read should work with madvise hints
        for i in range(reader.n_frames):
            frame = reader.read_frame(i)
            assert frame.metadata["timestep"] == i * 100
        reader.close()
