#!/usr/bin/env python3
"""Unit tests for the refactored Trajectory and TrajectoryReader classes.

This module contains comprehensive tests for:
- Trajectory class functionality (caching, indexing, metadata)
- TrajectoryReader base class functionality
- LammpsTrajectoryReader implementation
- Integration between Trajectory and TrajectoryReader
- Edge cases and error handling

Uses pytest framework with modern Python 3.10+ type hints and Google-style docstrings.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from molpy.core.box import Box
from molpy.core.frame import Block, Frame
from molpy.core.trajectory import Trajectory
from molpy.io.trajectory.lammps import LammpsTrajectoryReader, LammpsTrajectoryWriter


class TestTrajectory:
    """Test suite for the refactored Trajectory class."""

    @pytest.fixture
    def trajectory(self) -> Trajectory:
        """Create an empty trajectory instance.
        
        Returns:
            Empty Trajectory instance for testing.
        """
        return Trajectory()

    @pytest.fixture
    def test_frames(self) -> list[Frame]:
        """Create test frame instances.
        
        Returns:
            List of 5 test Frame instances with random atom data.
        """
        frames = []
        for i in range(5):
            atoms_data = {
                'id': np.arange(1, 11),  # 10 atoms
                'type': np.ones(10, dtype=int),
                'x': np.random.random(10),
                'y': np.random.random(10),
                'z': np.random.random(10)
            }
            
            box = Box(
                matrix=np.diag([10.0, 10.0, 10.0]),
                origin=np.zeros(3)
            )
            
            frame = Frame(box=box, timestep=i)
            frame["atoms"] = Block(atoms_data)
            frames.append(frame)
        return frames
    
    def test_empty_trajectory_initialization(self, trajectory: Trajectory) -> None:
        """Test creating an empty trajectory.
        
        Args:
            trajectory: Empty trajectory fixture.
        """
        assert len(trajectory) == 0
        assert len(trajectory.get_loaded_indices()) == 0
        assert trajectory._total_frames is None

    def test_trajectory_with_frames_initialization(self, test_frames: list[Frame]) -> None:
        """Test creating trajectory with initial frames.
        
        Args:
            test_frames: List of test frames fixture.
        """
        traj = Trajectory(test_frames[:3])
        assert len(traj.get_loaded_indices()) == 3
        assert traj.get_loaded_indices() == [0, 1, 2]

    def test_append_frame(self, trajectory: Trajectory, test_frames: list[Frame]) -> None:
        """Test appending frames to trajectory.
        
        Args:
            trajectory: Empty trajectory fixture.
            test_frames: List of test frames fixture.
        """
        trajectory.append(test_frames[0])
        assert len(trajectory.get_loaded_indices()) == 1
        assert 0 in trajectory.frames

    def test_frame_access_by_index(self, test_frames: list[Frame]) -> None:
        """Test accessing frames by index.
        
        Args:
            test_frames: List of test frames fixture.
        """
        traj = Trajectory(test_frames[:3])
        
        # Test positive indexing
        frame = traj[1]
        assert isinstance(frame, Frame)
        
        # Test negative indexing (requires total frames to be set)
        traj.set_total_frames(3)
        frame = traj[-1]
        assert isinstance(frame, Frame)

    def test_frame_access_unloaded_raises_keyerror(self) -> None:
        """Test that accessing unloaded frames raises KeyError."""
        traj = Trajectory()
        traj.set_total_frames(5)
        
        with pytest.raises(KeyError):
            _ = traj[2]

    def test_slice_access(self, test_frames: list[Frame]) -> None:
        """Test slice access returns new trajectory.
        
        Args:
            test_frames: List of test frames fixture.
        """
        traj = Trajectory(test_frames[:4])
        
        sliced = traj[1:3]
        assert isinstance(sliced, Trajectory)
        assert len(sliced.get_loaded_indices()) == 2

    def test_is_loaded(self, trajectory: Trajectory, test_frames: list[Frame]) -> None:
        """Test checking if frame is loaded.
        
        Args:
            trajectory: Empty trajectory fixture.
            test_frames: List of test frames fixture.
        """
        trajectory.append(test_frames[0])
        
        assert trajectory.is_loaded(0) is True
        assert trajectory.is_loaded(1) is False

    def test_need_more(self, test_frames: list[Frame]) -> None:
        """Test need_more method.
        
        Args:
            test_frames: List of test frames fixture.
        """
        traj = Trajectory(test_frames[:2])
        
        # We have 2 frames loaded, need more if total > 2
        assert traj.need_more(5) is True  # This sets total frames to 5
        
        # Now we know there are 5 total frames and 2 loaded
        assert traj.need_more(5) is True
        
        # If total equals loaded, no need for more
        assert traj.need_more(2) is False

    def test_cache_size_limit(self, test_frames: list[Frame]) -> None:
        """Test LRU cache behavior with size limits.
        
        Args:
            test_frames: List of test frames fixture.
        """
        traj = Trajectory(max_cache_size=2)
        
        # Add 3 frames - should evict the first one
        for i in range(3):
            traj.append(test_frames[i])
        
        # Should only have 2 frames cached
        assert len(traj.frames) == 2
        assert traj.is_loaded(0) is False  # First frame should be evicted

    def test_lru_access_order(self, test_frames: list[Frame]) -> None:
        """Test that LRU eviction works correctly.
        
        Args:
            test_frames: List of test frames fixture.
        """
        # Create trajectory with cache size 3, add 3 frames
        traj = Trajectory(max_cache_size=3)
        
        # Add frames manually to have more control
        for i in range(3):
            traj._add_frame(i, test_frames[i])
        
        # Access frame 0 to make it recently used
        _ = traj[0]
        
        # Add another frame - should evict frame 1, not frame 0
        traj._add_frame(3, test_frames[3])
        
        assert traj.is_loaded(0) is True   # Recently accessed
        assert traj.is_loaded(1) is False  # Should be evicted (LRU)
        assert traj.is_loaded(2) is True   # Still there
        assert traj.is_loaded(3) is True   # Just added

    def test_clear_cache(self, test_frames: list[Frame]) -> None:
        """Test clearing the cache.
        
        Args:
            test_frames: List of test frames fixture.
        """
        traj = Trajectory(test_frames[:3])
        traj.clear_cache()
        
        assert len(traj.frames) == 0
        assert len(traj.get_loaded_indices()) == 0

    def test_copy(self, test_frames: list[Frame]) -> None:
        """Test copying trajectory.
        
        Args:
            test_frames: List of test frames fixture.
        """
        traj = Trajectory(test_frames[:2], test_meta="value")
        copy_traj = traj.copy()
        
        assert len(copy_traj.get_loaded_indices()) == 2
        assert copy_traj.meta.get("test_meta") == "value"
        assert traj is not copy_traj

    def test_iteration(self, test_frames: list[Frame]) -> None:
        """Test iterating over trajectory.
        
        Args:
            test_frames: List of test frames fixture.
        """
        traj = Trajectory(test_frames[:3])
        frames = list(traj)
        
        assert len(frames) == 3
        for frame in frames:
            assert isinstance(frame, Frame)


class TestLammpsTrajectoryReader:
    """Test suite for the LammpsTrajectoryReader class."""

    @pytest.fixture
    def temp_files(self) -> list[str]:
        """Track temporary files for cleanup.
        
        Returns:
            Empty list to track temporary files.
        """
        return []

    @pytest.fixture(autouse=True)
    def cleanup_temp_files(self, temp_files: list[str]):
        """Clean up temporary files after each test.
        
        Args:
            temp_files: List of temporary files to clean up.
        """
        yield
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)

    def create_test_file(self, temp_files: list[str], n_frames: int = 3, n_atoms: int = 5) -> str:
        """Create a test LAMMPS trajectory file.
        
        Args:
            temp_files: List to track temporary files for cleanup.
            n_frames: Number of frames to create.
            n_atoms: Number of atoms per frame.
            
        Returns:
            Path to the created test file.
        """
        with tempfile.NamedTemporaryFile(suffix='.dump', delete=False) as tmp:
            filename = tmp.name
        temp_files.append(filename)
        
        # Create test data
        trajectory = Trajectory()
        for i in range(n_frames):
            atoms_data = {
                'id': np.arange(1, n_atoms + 1),
                'type': np.ones(n_atoms, dtype=int),
                'x': np.random.random(n_atoms),
                'y': np.random.random(n_atoms),
                'z': np.random.random(n_atoms),
                'q': np.zeros(n_atoms)
            }
            
            box = Box(
                matrix=np.diag([10.0, 10.0, 10.0]),
                origin=np.zeros(3)
            )
            
            frame = Frame(box=box, timestep=i)
            frame["atoms"] = Block(atoms_data)
            trajectory.append(frame)
        
        # Write to file
        with LammpsTrajectoryWriter(filename) as writer:
            for frame in trajectory:
                writer.write_frame(frame)
        
        return filename

    def test_reader_initialization(self, temp_files: list[str]) -> None:
        """Test reader initialization.
        
        Args:
            temp_files: List to track temporary files for cleanup.
        """
        filename = self.create_test_file(temp_files)
        trajectory = Trajectory()
        
        with LammpsTrajectoryReader(trajectory, filename) as reader:
            assert len(reader) == 3
            assert reader.n_frames == 3
            assert len(trajectory) == 3  # Total frames should be set

    def test_load_frame(self, temp_files: list[str]) -> None:
        """Test loading individual frames.
        
        Args:
            temp_files: List to track temporary files for cleanup.
        """
        filename = self.create_test_file(temp_files, n_frames=5)
        trajectory = Trajectory()
        
        with LammpsTrajectoryReader(trajectory, filename) as reader:
            # Load frame 2
            frame = reader.load_frame(2)
            
            assert isinstance(frame, Frame)
            assert trajectory.is_loaded(2) is True
            assert "atoms" in frame._blocks  # Check if atoms block exists
            
            # Loading again should return cached frame
            frame2 = reader.load_frame(2)
            assert frame is frame2  # Same object

    def test_load_frames_multiple(self, temp_files: list[str]) -> None:
        """Test loading multiple frames.
        
        Args:
            temp_files: List to track temporary files for cleanup.
        """
        filename = self.create_test_file(temp_files, n_frames=5)
        trajectory = Trajectory()
        
        with LammpsTrajectoryReader(trajectory, filename) as reader:
            frames = reader.load_frames([0, 2, 4])
            
            assert len(frames) == 3
            assert all(trajectory.is_loaded(i) for i in [0, 2, 4])

    def test_load_range(self, temp_files: list[str]) -> None:
        """Test loading range of frames.
        
        Args:
            temp_files: List to track temporary files for cleanup.
        """
        filename = self.create_test_file(temp_files, n_frames=6)
        trajectory = Trajectory()
        
        with LammpsTrajectoryReader(trajectory, filename) as reader:
            frames = reader.load_range(1, 5, 2)  # frames 1, 3
            
            assert len(frames) == 2
            assert trajectory.is_loaded(1) is True
            assert trajectory.is_loaded(3) is True
            assert trajectory.is_loaded(2) is False

    def test_preload_all(self, temp_files: list[str]) -> None:
        """Test preloading all frames.
        
        Args:
            temp_files: List to track temporary files for cleanup.
        """
        filename = self.create_test_file(temp_files, n_frames=4)
        trajectory = Trajectory()
        
        with LammpsTrajectoryReader(trajectory, filename) as reader:
            reader.preload_all()
            
            loaded_indices = trajectory.get_loaded_indices()
            assert len(loaded_indices) == 4
            assert loaded_indices == [0, 1, 2, 3]

    def test_iteration(self, temp_files: list[str]) -> None:
        """Test iteration through reader.
        
        Args:
            temp_files: List to track temporary files for cleanup.
        """
        filename = self.create_test_file(temp_files, n_frames=4)
        trajectory = Trajectory()
        
        with LammpsTrajectoryReader(trajectory, filename) as reader:
            frames = list(reader)
            
            assert len(frames) == 4
            # All frames should now be loaded in trajectory
            assert len(trajectory.get_loaded_indices()) == 4

    def test_index_out_of_range(self, temp_files: list[str]) -> None:
        """Test handling of invalid frame indices.
        
        Args:
            temp_files: List to track temporary files for cleanup.
        """
        filename = self.create_test_file(temp_files, n_frames=3)
        trajectory = Trajectory()
        
        with LammpsTrajectoryReader(trajectory, filename) as reader:
            with pytest.raises(IndexError):
                reader.load_frame(5)
            
            with pytest.raises(IndexError):
                reader.load_frame(-5)

    def test_multiple_files(self, temp_files: list[str]) -> None:
        """Test reading from multiple files.
        
        Args:
            temp_files: List to track temporary files for cleanup.
        """
        file1 = self.create_test_file(temp_files, n_frames=2)
        file2 = self.create_test_file(temp_files, n_frames=3)
        trajectory = Trajectory()
        
        with LammpsTrajectoryReader(trajectory, [file1, file2]) as reader:
            assert len(reader) == 5  # 2 + 3 frames
            
            # Load frames from both files
            frame0 = reader.load_frame(0)  # From file1
            frame3 = reader.load_frame(3)  # From file2
            
            assert isinstance(frame0, Frame)
            assert isinstance(frame3, Frame)


class TestTrajectoryReaderIntegration:
    """Test suite for integration between Trajectory and TrajectoryReader."""

    @pytest.fixture
    def temp_files(self) -> list[str]:
        """Track temporary files for cleanup.
        
        Returns:
            Empty list to track temporary files.
        """
        return []

    @pytest.fixture(autouse=True)
    def cleanup_temp_files(self, temp_files: list[str]):
        """Clean up temporary files after each test.
        
        Args:
            temp_files: List of temporary files to clean up.
        """
        yield
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)

    def create_test_file(self, temp_files: list[str], n_frames: int = 5, n_atoms: int = 10) -> str:
        """Create a test LAMMPS trajectory file.
        
        Args:
            temp_files: List to track temporary files for cleanup.
            n_frames: Number of frames to create.
            n_atoms: Number of atoms per frame.
            
        Returns:
            Path to the created test file.
        """
        with tempfile.NamedTemporaryFile(suffix='.dump', delete=False) as tmp:
            filename = tmp.name
        temp_files.append(filename)
        
        # Create and write test data
        trajectory = Trajectory()
        for i in range(n_frames):
            atoms_data = {
                'id': np.arange(1, n_atoms + 1),
                'type': np.ones(n_atoms, dtype=int),
                'x': np.random.random(n_atoms) * 10,
                'y': np.random.random(n_atoms) * 10,
                'z': np.random.random(n_atoms) * 10,
                'q': np.random.random(n_atoms) - 0.5
            }
            
            frame = Frame(timestep=i)
            frame["atoms"] = Block(atoms_data)
            trajectory.append(frame)
        
        with LammpsTrajectoryWriter(filename) as writer:
            for frame in trajectory:
                writer.write_frame(frame)
        
        return filename

    def test_lazy_loading_workflow(self, temp_files: list[str]) -> None:
        """Test the complete lazy loading workflow.
        
        Args:
            temp_files: List to track temporary files for cleanup.
        """
        filename = self.create_test_file(temp_files, n_frames=10)
        
        # Create trajectory with cache limit
        trajectory = Trajectory(max_cache_size=3)
        
        with LammpsTrajectoryReader(trajectory, filename) as reader:
            # Initially no frames loaded
            assert len(trajectory.get_loaded_indices()) == 0
            
            # Load some frames
            reader.load_frame(0)
            reader.load_frame(5)
            reader.load_frame(9)
            
            # Should have 3 frames loaded
            assert len(trajectory.get_loaded_indices()) == 3
            
            # Load one more - should evict LRU
            reader.load_frame(2)
            assert len(trajectory.get_loaded_indices()) == 3
            
            # Can access loaded frames directly from trajectory
            frame_2 = trajectory[2]
            assert isinstance(frame_2, Frame)

    def test_random_access_pattern(self, temp_files: list[str]) -> None:
        """Test random access pattern with caching.
        
        Args:
            temp_files: List to track temporary files for cleanup.
        """
        filename = self.create_test_file(temp_files, n_frames=8)
        trajectory = Trajectory(max_cache_size=4)
        
        with LammpsTrajectoryReader(trajectory, filename) as reader:
            # Random access pattern
            access_pattern = [3, 1, 6, 2, 7, 0, 4, 5]
            
            for frame_idx in access_pattern:
                frame = reader.load_frame(frame_idx)
                assert isinstance(frame, Frame)
                
                # Verify frame contains expected data
                atoms = frame["atoms"]
                assert "id" in atoms
                assert "x" in atoms

    def test_sequential_then_random_access(self, temp_files: list[str]) -> None:
        """Test mixing sequential and random access.
        
        Args:
            temp_files: List to track temporary files for cleanup.
        """
        filename = self.create_test_file(temp_files, n_frames=6)
        trajectory = Trajectory()
        
        with LammpsTrajectoryReader(trajectory, filename) as reader:
            # Sequential access first
            for i in range(3):
                reader.load_frame(i)
            
            # Then random access
            reader.load_frame(5)
            reader.load_frame(4)
            
            # Check all expected frames are loaded
            loaded = trajectory.get_loaded_indices()
            expected = [0, 1, 2, 4, 5]
            assert sorted(loaded) == expected

    def test_iteration_integration(self, temp_files: list[str]) -> None:
        """Test iteration behavior fills trajectory.
        
        Args:
            temp_files: List to track temporary files for cleanup.
        """
        filename = self.create_test_file(temp_files, n_frames=5)
        trajectory = Trajectory()
        
        with LammpsTrajectoryReader(trajectory, filename) as reader:
            # Iterate through first 3 frames
            frames = []
            for i, frame in enumerate(reader):
                frames.append(frame)
                if i >= 2:
                    break
            
            # Should have loaded 3 frames
            assert len(trajectory.get_loaded_indices()) == 3
            
            # Continue iteration
            remaining_frames = list(reader)[3:]  # Get rest of frames
            
            # All frames should now be loaded
            assert len(trajectory.get_loaded_indices()) == 5
