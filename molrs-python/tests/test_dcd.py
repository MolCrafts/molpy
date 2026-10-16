"""FFI smoke tests for DCD read/write bindings.

Self-contained: fixtures are written with ``molrs.io.dcd.DcdWriter``. No
chemfiles corpus and no third-party trajectory software.
"""

from __future__ import annotations

from pathlib import Path

import molrs
import pytest


class TestReadDcd:
    def test_parses_written_trajectory(self, water_dcd):
        frames = list(molrs.io.dcd.DcdReader(water_dcd))
        assert len(frames) == 2
        for i, frame in enumerate(frames):
            assert "atoms" in frame, f"frame {i} missing atoms block"
            assert frame["atoms"].n_rows == 3

    def test_the_one_frame_door_reads_the_first_frame(self, water_dcd):
        first = molrs.io.read_dcd(water_dcd)
        assert first["atoms"].n_rows == molrs.io.dcd.DcdReader(water_dcd)[0]["atoms"].n_rows

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.dcd.DcdReader(Path("/nonexistent/path.dcd"))


class TestDcdReader:
    def test_random_access_matches_sequential(self, water_dcd):
        sequential = list(molrs.io.dcd.DcdReader(water_dcd))
        reader = molrs.io.dcd.DcdReader(water_dcd)
        assert len(reader) == len(sequential)
        for n in reversed(range(len(sequential))):
            frame = reader[n]
            assert frame["atoms"].n_rows == sequential[n]["atoms"].n_rows

    def test_iteration(self, water_dcd):
        reader = molrs.io.dcd.DcdReader(water_dcd)
        n = len(reader)
        assert sum(1 for _ in reader) == n
        assert sum(1 for _ in reader) == n

    def test_negative_index(self, water_dcd):
        reader = molrs.io.dcd.DcdReader(water_dcd)
        eager = list(molrs.io.dcd.DcdReader(water_dcd))
        assert reader[-1]["atoms"].n_rows == eager[-1]["atoms"].n_rows

    def test_index_error(self, water_dcd):
        reader = molrs.io.dcd.DcdReader(water_dcd)
        with pytest.raises(IndexError):
            _ = reader[10_000_000]


class TestDcdReaderIndexing:
    def test_index_matches_eager(self, water_dcd):
        eager = list(molrs.io.dcd.DcdReader(water_dcd))
        reader = molrs.io.dcd.DcdReader(water_dcd)
        assert reader[0]["atoms"].n_rows == eager[0]["atoms"].n_rows
        assert reader[-1]["atoms"].n_rows == eager[-1]["atoms"].n_rows

    def test_several_indices(self, water_dcd):
        reader = molrs.io.dcd.DcdReader(water_dcd)
        n = len(reader)
        frames = [reader[i] for i in [0, n - 1, -1]]
        assert len(frames) == 3

    def test_full_slice_matches_eager(self, water_dcd):
        eager = list(molrs.io.dcd.DcdReader(water_dcd))
        reader = molrs.io.dcd.DcdReader(water_dcd)
        assert len(reader[:]) == len(eager)


class TestWriteDcd:
    def test_round_trip_atom_count(self, water_frame, tmp_path):
        path = tmp_path / "out.dcd"
        with molrs.io.dcd.DcdWriter(path) as writer:
            writer.write(water_frame)
            writer.write(water_frame)
        frames = list(molrs.io.dcd.DcdReader(path))
        assert len(frames) == 2
        assert frames[0]["atoms"].n_rows == 3
