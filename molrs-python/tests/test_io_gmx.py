"""GROMACS TRR/XTC binding smoke tests.

Self-contained: trajectories are written with molrs, then read back.
No external corpus, no third-party MD packages.
"""

from __future__ import annotations

import molrs
import molrs.io as mio
import numpy as np
import pytest


class TestEagerReads:
    def test_read_trr_into_a_list_of_frames(self, water_trr):
        frames = list(molrs.io.trr.TrrReader(water_trr))
        assert isinstance(frames, list) and len(frames) >= 1
        atoms = frames[0]["atoms"]
        assert atoms.n_rows > 0
        for axis in ("x", "y", "z"):
            assert np.all(np.isfinite(atoms[axis]))

    def test_read_xtc_into_a_list_of_frames(self, water_xtc):
        frames = list(molrs.io.xtc.XtcReader(water_xtc))
        assert isinstance(frames, list) and len(frames) >= 1
        assert frames[0]["atoms"].n_rows > 0

    def test_one_frame_doors_read_the_first_frame(self, water_trr, water_xtc):
        np.testing.assert_allclose(
            mio.read_trr(water_trr)["atoms"]["x"], mio.trr.TrrReader(water_trr)[0]["atoms"]["x"]
        )
        np.testing.assert_allclose(
            mio.read_xtc(water_xtc)["atoms"]["x"], mio.xtc.XtcReader(water_xtc)[0]["atoms"]["x"]
        )


class TestLazyReaders:
    def test_trr_reader(self, water_trr):
        reader = mio.trr.TrrReader(water_trr)
        assert isinstance(reader, mio.trr.TrrReader)
        assert len(reader) > 0

    def test_xtc_reader(self, water_xtc):
        reader = mio.xtc.XtcReader(water_xtc)
        assert isinstance(reader, mio.xtc.XtcReader)
        assert len(reader) > 0

    def test_random_access_matches_sequential(self, water_trr):
        reader = mio.trr.TrrReader(water_trr)
        eager = list(molrs.io.trr.TrrReader(water_trr))
        assert len(reader) == len(eager)
        last = reader[-1]["atoms"]["x"]
        assert np.allclose(last, eager[-1]["atoms"]["x"])

    def test_out_of_range_raises(self, water_xtc):
        reader = mio.xtc.XtcReader(water_xtc)
        with pytest.raises(IndexError):
            reader[10_000_000]

    def test_bytes_read_as_the_path(self, water_trr):
        from_bytes = mio.trr.TrrReader(water_trr.read_bytes())
        assert len(from_bytes) == len(mio.trr.TrrReader(water_trr))


class TestWriteRoundTrip:
    def test_trr_roundtrip_exact(self, water_trr, tmp_path):
        frames = list(molrs.io.trr.TrrReader(water_trr))
        out = tmp_path / "out.trr"
        with mio.trr.TrrWriter(out) as writer:
            for frame in frames:
                writer.write(frame)
        back = list(molrs.io.trr.TrrReader(out))
        assert len(back) == len(frames)
        for a, b in zip(frames, back):
            assert np.allclose(a["atoms"]["x"], b["atoms"]["x"], atol=1e-5)

    def test_xtc_roundtrip_within_precision(self, water_xtc, tmp_path):
        frames = list(molrs.io.xtc.XtcReader(water_xtc))
        out = tmp_path / "out.xtc"
        with mio.xtc.XtcWriter(out) as writer:
            for frame in frames:
                writer.write(frame)
        back = list(molrs.io.xtc.XtcReader(out))
        assert len(back) == len(frames)
        for a, b in zip(frames, back):
            # XTC is lossy at 1/precision (default 1000 → 1e-3 nm).
            assert np.allclose(a["atoms"]["x"], b["atoms"]["x"], atol=2e-3)
