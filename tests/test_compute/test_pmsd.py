"""Tests for PMSD (Polarization Mean Square Displacement) computation module."""

import numpy as np
import pytest

from molpy import Frame, Trajectory
from molpy.compute.pmsd import PMSDCompute
from molpy.core.box import Box


@pytest.fixture
def simple_box():
    """Create a simple cubic box."""
    return Box(matrix=np.diag([10.0, 10.0, 10.0]))


@pytest.fixture
def ionic_trajectory(simple_box):
    """Create a trajectory with cations and anions."""
    frames = []

    n_cations = 5
    n_anions = 5
    n_atoms = n_cations + n_anions
    n_frames = 100
    dt = 0.01

    # Create random walk for ions
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
        # First n_cations are type 1 (cations), rest are type 2 (anions)
        atoms["type"] = np.array([1] * n_cations + [2] * n_anions, dtype=int)
        frame["atoms"] = atoms
        frame.metadata["box"] = simple_box
        frames.append(frame)

    return Trajectory(frames)


class TestPMSDCompute:
    """Tests for PMSDCompute class."""

    def test_init(self):
        """Test PMSDCompute initialization."""
        pmsd = PMSDCompute(cation_type=1, anion_type=2, max_dt=30.0, dt=0.01)

        assert pmsd.cation_type == 1
        assert pmsd.anion_type == 2
        assert pmsd.max_dt == 30.0
        assert pmsd.dt == 0.01
        assert pmsd.n_cache == 3000

    def test_compute_pmsd(self, ionic_trajectory):
        """Test PMSD computation."""
        pmsd = PMSDCompute(cation_type=1, anion_type=2, max_dt=1.0, dt=0.01)

        result = pmsd.compute(ionic_trajectory)

        assert result.time.shape == (100,)
        assert result.pmsd.shape == (100,)

        # PMSD should start near zero and generally increase
        assert (
            result.pmsd[0] < result.pmsd[50]
            or abs(result.pmsd[0] - result.pmsd[50]) < 1e-6
        )

    def test_missing_frame_data_raises(self):
        """Test that missing frame data raises appropriate errors."""
        frame = Frame()
        # Missing atoms block
        trajectory = Trajectory([frame])

        pmsd = PMSDCompute(cation_type=1, anion_type=2, max_dt=1.0, dt=0.01)
        with pytest.raises(ValueError, match="must contain 'atoms' block"):
            pmsd.compute(trajectory)

    def test_missing_coordinates_raises(self):
        """Test that missing coordinates raise error."""
        frame = Frame()
        frame["atoms"] = {"type": np.array([1, 2])}
        frame.metadata["box"] = Box(matrix=np.diag([10.0, 10.0, 10.0]))
        trajectory = Trajectory([frame])

        pmsd = PMSDCompute(cation_type=1, anion_type=2, max_dt=1.0, dt=0.01)
        with pytest.raises(ValueError, match="must contain x, y, z coordinates"):
            pmsd.compute(trajectory)

    def test_missing_type_field_raises(self):
        """Test that missing type field raises error."""
        frame = Frame()
        frame["atoms"] = {
            "x": np.array([0.0, 0.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
        }
        frame.metadata["box"] = Box(matrix=np.diag([10.0, 10.0, 10.0]))
        trajectory = Trajectory([frame])

        pmsd = PMSDCompute(cation_type=1, anion_type=2, max_dt=1.0, dt=0.01)
        with pytest.raises(ValueError, match="must contain 'type' field"):
            pmsd.compute(trajectory)

    def test_missing_box_raises(self):
        """Test that missing box information raises error."""
        frame = Frame()
        frame["atoms"] = {
            "x": np.array([0.0, 0.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "type": np.array([1, 2]),
        }
        # Missing box
        trajectory = Trajectory([frame])

        pmsd = PMSDCompute(cation_type=1, anion_type=2, max_dt=1.0, dt=0.01)
        with pytest.raises(ValueError, match="must contain box information"):
            pmsd.compute(trajectory)
