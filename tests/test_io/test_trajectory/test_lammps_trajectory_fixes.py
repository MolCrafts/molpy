"""
Tests for LAMMPS trajectory writer functionality.
This focuses on basic trajectory writing and reading capabilities
with the fixes we've implemented.
"""

import pytest
import tempfile
import numpy as np
import molpy as mp
from pathlib import Path


class TestLammpsTrajectoryFixes:
    """Test LAMMPS trajectory functionality with fixes."""

    def test_basic_trajectory_write_read(self):
        """Test basic trajectory writing and reading."""
        # Create frames
        frames = []
        for timestep in [0, 100, 200]:
            frame = mp.Frame()
            
            atoms_data = {
                'id': [0, 1],
                'type': [1, 2],  # Use numeric types for trajectory
                'x': [0.0 + timestep*0.01, 1.0 + timestep*0.01],
                'y': [0.0, 0.0],
                'z': [0.0, 0.0],
                'q': [0.5, -0.5]
            }
            frame["atoms"] = atoms_data
            frame["timestep"] = timestep
            frame.box = mp.Box(np.eye(3) * 10.0)
            frames.append(frame)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as tmp:
            # Write trajectory
            writer = mp.io.trajectory.LammpsTrajectoryWriter(tmp.name)
            for frame in frames:
                timestep = frame.get("timestep", 0)
                writer.write_frame(frame, timestep=timestep)
            writer.close()
            
            # Verify file exists and has content
            assert Path(tmp.name).exists()
            assert Path(tmp.name).stat().st_size > 0
            
            # Read back and verify
            reader = mp.io.trajectory.LammpsTrajectoryReader(tmp.name)
            
            read_frames = []
            for frame in reader:
                read_frames.append(frame)
                if len(read_frames) >= 3:
                    break
            
            assert len(read_frames) == 3
            
            # Verify timesteps
            expected_timesteps = [0, 100, 200]
            for i, frame in enumerate(read_frames):
                assert frame["timestep"] == expected_timesteps[i]

    def test_trajectory_with_xyz_coordinates(self):
        """Test trajectory writing with xyz coordinate format."""
        frame = mp.Frame()
        
        atoms_data = {
            'id': [0, 1, 2],
            'type': [1, 1, 2],
            'xyz': [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0]]
        }
        frame["atoms"] = atoms_data
        frame["timestep"] = 0
        frame.box = mp.Box(np.eye(3) * 5.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as tmp:
            writer = mp.io.trajectory.LammpsTrajectoryWriter(tmp.name)
            writer.write_frame(frame, timestep=0)
            writer.close()
            
            # Should not crash and should create valid file
            assert Path(tmp.name).exists()
            assert Path(tmp.name).stat().st_size > 0

    def test_trajectory_context_manager(self):
        """Test trajectory writer as context manager."""
        frame = mp.Frame()
        
        atoms_data = {
            'id': [0, 1],
            'type': [1, 1],
            'x': [0.0, 1.0],
            'y': [0.0, 0.0], 
            'z': [0.0, 0.0]
        }
        frame["atoms"] = atoms_data
        frame["timestep"] = 42
        frame.box = mp.Box(np.eye(3) * 5.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as tmp:
            with mp.io.trajectory.LammpsTrajectoryWriter(tmp.name) as writer:
                writer.write_frame(frame, timestep=42)
            
            # File should exist and have content
            assert Path(tmp.name).exists()
            assert Path(tmp.name).stat().st_size > 0

    def test_multiple_frame_trajectory(self):
        """Test writing multiple frames to trajectory."""
        frames = []
        
        # Create several frames with moving atoms
        for i in range(5):
            frame = mp.Frame()
            
            atoms_data = {
                'id': [0, 1, 2],
                'type': [1, 1, 2],
                'x': [0.0 + i*0.1, 1.0 + i*0.1, 0.5 + i*0.1],
                'y': [0.0, 0.0, 1.0],
                'z': [0.0, 0.0, 0.0]
            }
            frame["atoms"] = atoms_data
            frame["timestep"] = i * 10
            frame.box = mp.Box(np.eye(3) * 10.0)
            frames.append(frame)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as tmp:
            writer = mp.io.trajectory.LammpsTrajectoryWriter(tmp.name)
            
            for frame in frames:
                timestep = frame.get("timestep", 0)
                writer.write_frame(frame, timestep=timestep)
            
            writer.close()
            
            # Verify file content
            with open(tmp.name, 'r') as f:
                content = f.read()
            
            # Should contain all timesteps
            assert "ITEM: TIMESTEP" in content
            assert "0\n" in content  # First timestep
            assert "40\n" in content  # Last timestep
            
            # Should contain atom data for all frames
            assert content.count("ITEM: ATOMS") == 5

    def test_trajectory_box_handling(self):
        """Test that box information is correctly written to trajectory."""
        frame = mp.Frame()
        
        atoms_data = {
            'id': [0],
            'type': [1],
            'x': [0.0],
            'y': [0.0],
            'z': [0.0]
        }
        frame["atoms"] = atoms_data
        frame["timestep"] = 0
        
        # Test with custom box dimensions
        frame.box = mp.Box(np.diag([5.0, 7.5, 10.0]))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', delete=False) as tmp:
            writer = mp.io.trajectory.LammpsTrajectoryWriter(tmp.name)
            writer.write_frame(frame, timestep=0)
            writer.close()
            
            with open(tmp.name, 'r') as f:
                content = f.read()
            
            # Should contain box bounds
            assert "ITEM: BOX BOUNDS" in content
            assert "5.000000" in content  # x dimension
            assert "7.500000" in content  # y dimension
            assert "10.000000" in content  # z dimension
