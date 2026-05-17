"""Tests for dielectric Compute classes in molpy.compute.dielectric."""

from unittest.mock import patch

import numpy as np
import pytest

from molpy.compute.result import ACFResult, DielectricSusceptibilityResult
from molpy.core.box import Box
from molpy.core.frame import Block, Frame


def make_test_trajectory(n_frames=10, n_atoms=5, with_charges=True, with_box=True):
    """Build a synthetic Trajectory for testing."""
    rng = np.random.default_rng(42)
    frames = []
    for i in range(n_frames):
        block = Block()
        block["x"] = rng.random(n_atoms) + i * 0.1
        block["y"] = rng.random(n_atoms)
        block["z"] = rng.random(n_atoms)
        if with_charges:
            block["charge"] = np.ones(n_atoms) * 0.5
        frame = Frame()
        frame["atoms"] = block
        if with_box:
            frame.box = Box.cubic(10.0)
        frames.append(frame)

    class ListTrajectory:
        def __init__(self, frames):
            self._frames = frames

        def __getitem__(self, idx):
            return self._frames[idx]

        def __len__(self):
            return len(self._frames)

        def __iter__(self):
            return iter(self._frames)

    return ListTrajectory(frames)


class TestACFAnalyzer:
    def test_import(self):
        from molpy.compute.dielectric import ACFAnalyzer
        from molpy.compute.base import Compute

        assert issubclass(ACFAnalyzer, Compute)

    def test_delegates_to_molrs_acf_fft(self):
        from molpy.compute.dielectric import ACFAnalyzer

        traj = make_test_trajectory(n_frames=10)

        with patch("molpy.compute.dielectric.acf_fft") as mock_acf:
            mock_acf.return_value = np.ones(3)
            analyzer = ACFAnalyzer(columns=["x", "y", "z"], max_lag=2, unwrap=False)
            result = analyzer(traj)
            assert mock_acf.call_count >= 1
            assert isinstance(result, ACFResult)

    def test_raises_missing_column(self):
        from molpy.compute.dielectric import ACFAnalyzer

        traj = make_test_trajectory(n_frames=10)
        analyzer = ACFAnalyzer(columns=["x", "vx"], max_lag=2, unwrap=False)
        with pytest.raises(ValueError, match="Missing column"):
            analyzer(traj)

    def test_raises_insufficient_frames(self):
        from molpy.compute.dielectric import ACFAnalyzer

        traj = make_test_trajectory(n_frames=1)
        analyzer = ACFAnalyzer(columns=["x", "y", "z"], max_lag=2, unwrap=False)
        with pytest.raises(ValueError, match="at least 2"):
            analyzer(traj)

    def test_raises_missing_box_when_unwrap(self):
        from molpy.compute.dielectric import ACFAnalyzer

        traj = make_test_trajectory(n_frames=10, with_box=False)
        analyzer = ACFAnalyzer(columns=["x", "y", "z"], max_lag=2, unwrap=True)
        with pytest.raises(ValueError, match="Box"):
            analyzer(traj)

    def test_returns_normalized_acf(self):
        from molpy.compute.dielectric import ACFAnalyzer

        traj = make_test_trajectory(n_frames=50, n_atoms=10)
        analyzer = ACFAnalyzer(columns=["x", "y", "z"], max_lag=5, unwrap=False)
        result = analyzer(traj)
        assert result.acf.shape == (6,)
        assert np.allclose(result.acf[0], 1.0, atol=1e-10)

    def test_immutability(self):
        from molpy.compute.dielectric import ACFAnalyzer

        traj = make_test_trajectory(n_frames=10)
        frame0_x = traj[0]["atoms"]["x"].copy()
        analyzer = ACFAnalyzer(columns=["x", "y", "z"], max_lag=2, unwrap=False)
        analyzer(traj)
        assert np.array_equal(traj[0]["atoms"]["x"], frame0_x)


class TestSpectralAnalyzer:
    def test_import(self):
        from molpy.compute.dielectric import SpectralAnalyzer
        from molpy.compute.base import Compute

        assert issubclass(SpectralAnalyzer, Compute)

    def test_shape(self):
        from molpy.compute.dielectric import SpectralAnalyzer

        acf_result = ACFResult(
            acf=np.ones(10), n_lags=10, time=np.arange(10, dtype=np.float64)
        )
        analyzer = SpectralAnalyzer(dt=0.001, window_type="hann")
        result = analyzer(acf_result)
        assert len(result.frequency) == len(result.spectrum)


class TestDielectricSusceptibility:
    def test_import(self):
        from molpy.compute.dielectric import DielectricSusceptibility
        from molpy.compute.base import Compute

        assert issubclass(DielectricSusceptibility, Compute)

    def test_raises_missing_charge(self):
        from molpy.compute.dielectric import DielectricSusceptibility

        traj = make_test_trajectory(n_frames=10, with_charges=False)
        comp = DielectricSusceptibility(
            dt=0.001, temperature=300.0, max_correlation_time=5
        )
        with pytest.raises(ValueError, match="column"):
            comp(traj)

    def test_raises_insufficient_frames(self):
        from molpy.compute.dielectric import DielectricSusceptibility

        traj = make_test_trajectory(n_frames=1)
        comp = DielectricSusceptibility(
            dt=0.001, temperature=300.0, max_correlation_time=5
        )
        with pytest.raises(ValueError, match="at least 2"):
            comp(traj)

    def test_raises_missing_box(self):
        from molpy.compute.dielectric import DielectricSusceptibility

        traj = make_test_trajectory(n_frames=10, with_box=False)
        comp = DielectricSusceptibility(
            dt=0.001, temperature=300.0, max_correlation_time=5
        )
        with pytest.raises(ValueError, match="Box"):
            comp(traj)

    def test_returns_result_with_keys(self):
        from molpy.compute.dielectric import DielectricSusceptibility

        traj = make_test_trajectory(n_frames=20, n_atoms=5)
        comp = DielectricSusceptibility(
            dt=0.001,
            temperature=300.0,
            max_correlation_time=5,
            routes=["einstein-helfand", "green-kubo"],
        )
        result = comp(traj)
        assert isinstance(result, DielectricSusceptibilityResult)
        assert "EH-full" in result.results
        assert "GK-full" in result.results

    def test_immutability(self):
        from molpy.compute.dielectric import DielectricSusceptibility

        traj = make_test_trajectory(n_frames=20, n_atoms=5)
        frame0_x = traj[0]["atoms"]["x"].copy()
        comp = DielectricSusceptibility(
            dt=0.001,
            temperature=300.0,
            max_correlation_time=5,
        )
        comp(traj)
        assert np.array_equal(traj[0]["atoms"]["x"], frame0_x)

    def test_zero_python_physics(self):
        """Verify no np.fft calls in dielectric.py."""
        import inspect
        from molpy.compute import dielectric as dm

        source = inspect.getsource(dm)
        assert "np.fft" not in source
        assert "np.sum(" not in source or "# np.sum used for indexing only" in source
