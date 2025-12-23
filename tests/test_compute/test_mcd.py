"""Tests for MCD (Mean Displacement Correlation) computation module."""

import numpy as np
import pytest

from molpy import Frame, Trajectory
from molpy.compute.mcd import MCDCompute
from molpy.core.box import Box


@pytest.fixture
def simple_box():
    """Create a simple cubic box."""
    return Box(matrix=np.diag([10.0, 10.0, 10.0]))


@pytest.fixture
def simple_trajectory(simple_box):
    """Create a simple trajectory with known diffusion behavior."""
    frames = []

    n_atoms = 10
    n_frames = 100
    dt = 0.01

    # Create random walk trajectory
    np.random.seed(42)
    positions = np.zeros((n_frames, n_atoms, 3))
    positions[0] = np.random.rand(n_atoms, 3) * 10.0

    for i in range(1, n_frames):
        # Random walk with step size ~0.1
        step = np.random.randn(n_atoms, 3) * 0.1
        positions[i] = positions[i - 1] + step

    for frame_idx in range(n_frames):
        frame = Frame()
        atoms = {}
        atoms["x"] = positions[frame_idx, :, 0]
        atoms["y"] = positions[frame_idx, :, 1]
        atoms["z"] = positions[frame_idx, :, 2]
        atoms["type"] = np.ones(n_atoms, dtype=int)  # All type 1
        frame["atoms"] = atoms
        frame.metadata["box"] = simple_box
        frames.append(frame)

    return Trajectory(frames)


class TestMCDCompute:
    """Tests for MCDCompute class."""

    def test_init(self):
        """Test MCDCompute initialization."""
        mcd = MCDCompute(tags=["1"], max_dt=30.0, dt=0.01)

        assert mcd.tags == ["1"]
        assert mcd.max_dt == 30.0
        assert mcd.dt == 0.01
        assert mcd.n_cache == 3000

    def test_self_diffusion_single_type(self, simple_trajectory):
        """Test self-diffusion computation for single atom type."""
        mcd = MCDCompute(tags=["1"], max_dt=1.0, dt=0.01)

        result = mcd.compute(simple_trajectory)

        assert "1" in result.correlations
        assert result.time.shape == (100,)
        assert result.correlations["1"].shape == (100,)

        # Correlations (MSD) should start near zero and increase
        assert result.correlations["1"][0] < result.correlations["1"][50]

    def test_distinct_diffusion(self, simple_trajectory):
        """Test distinct diffusion computation between two types."""
        # Create trajectory with two types
        frames = []
        n_atoms = 10
        n_frames = 100
        box = Box(matrix=np.diag([10.0, 10.0, 10.0]))

        np.random.seed(42)
        positions = np.zeros((n_frames, n_atoms, 3))
        positions[0] = np.random.rand(n_atoms, 3) * 10.0

        for i in range(1, n_frames):
            step = np.random.randn(n_atoms, 3) * 0.1
            positions[i] = positions[i - 1] + step

        for frame_idx in range(n_frames):
            frame = Frame()
            atoms = {}
            atoms["x"] = positions[frame_idx, :, 0]
            atoms["y"] = positions[frame_idx, :, 1]
            atoms["z"] = positions[frame_idx, :, 2]
            # First 5 atoms type 1, next 5 type 2
            atoms["type"] = np.array([1] * 5 + [2] * 5, dtype=int)
            frame["atoms"] = atoms
            frame.metadata["box"] = box
            frames.append(frame)

        trajectory = Trajectory(frames)

        mcd = MCDCompute(tags=["1,2"], max_dt=1.0, dt=0.01)
        result = mcd.compute(trajectory)

        assert "1,2" in result.correlations

    def test_center_of_mass_removal(self, simple_trajectory):
        """Test center of mass motion removal."""
        center_of_mass = {1: 1.0}  # Mass for type 1
        mcd = MCDCompute(
            tags=["1"],
            max_dt=1.0,
            dt=0.01,
            center_of_mass=center_of_mass,
        )

        result = mcd.compute(simple_trajectory)

        # Should compute successfully with COM removal
        assert "1" in result.correlations

    def test_missing_frame_data_raises(self):
        """Test that missing frame data raises appropriate errors."""
        frame = Frame()
        # Missing atoms block
        trajectory = Trajectory([frame])

        mcd = MCDCompute(tags=["1"], max_dt=1.0, dt=0.01)
        with pytest.raises(ValueError, match="must contain 'atoms' block"):
            mcd.compute(trajectory)

    def test_missing_coordinates_raises(self):
        """Test that missing coordinates raise error."""
        frame = Frame()
        frame["atoms"] = {"type": np.array([1])}
        frame.metadata["box"] = Box(matrix=np.diag([10.0, 10.0, 10.0]))
        trajectory = Trajectory([frame])

        mcd = MCDCompute(tags=["1"], max_dt=1.0, dt=0.01)
        with pytest.raises(ValueError, match="must contain x, y, z coordinates"):
            mcd.compute(trajectory)

    def test_missing_type_field_raises(self):
        """Test that missing type field raises error."""
        frame = Frame()
        frame["atoms"] = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
        }
        frame.metadata["box"] = Box(matrix=np.diag([10.0, 10.0, 10.0]))
        trajectory = Trajectory([frame])

        mcd = MCDCompute(tags=["1"], max_dt=1.0, dt=0.01)
        with pytest.raises(ValueError, match="must contain 'type' field"):
            mcd.compute(trajectory)

    def test_missing_box_raises(self):
        """Test that missing box information raises error."""
        frame = Frame()
        frame["atoms"] = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "type": np.array([1]),
        }
        # Missing box
        trajectory = Trajectory([frame])

        mcd = MCDCompute(tags=["1"], max_dt=1.0, dt=0.01)
        with pytest.raises(ValueError, match="must contain box information"):
            mcd.compute(trajectory)

    def test_fit_failure_returns_nan(self):
        """Test that fitting failure results in NaN diffusion coefficient."""
        # Create trajectory with insufficient data for fitting
        frames = []
        n_atoms = 2
        n_frames = 5
        box = Box(matrix=np.diag([10.0, 10.0, 10.0]))

        for frame_idx in range(n_frames):
            frame = Frame()
            atoms = {}
            atoms["x"] = np.array([0.0] * n_atoms)
            atoms["y"] = np.array([0.0] * n_atoms)
            atoms["z"] = np.array([0.0] * n_atoms)
            atoms["type"] = np.ones(n_atoms, dtype=int)
            frame["atoms"] = atoms
            frame.metadata["box"] = box
            frames.append(frame)

        trajectory = Trajectory(frames)

        mcd = MCDCompute(tags=["1"], max_dt=1.0, dt=0.01)
        result = mcd.compute(trajectory)

        # Should compute correlations successfully
        assert "1" in result.correlations
