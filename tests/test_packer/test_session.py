"""
Unit tests for the packer.session module.
Tests Session class functionality.
"""

import pytest
import numpy as np
import molpy as mp
from molpy.pack.session import Session
from molpy.pack.target import Target
import molpy.pack.constraint as mpk_constraint


class MockPacker:
    """Mock packer for testing Session without external dependencies."""
    
    def __init__(self, *args, **kwargs):
        self.targets = []
        
    def add_target(self, target):
        self.targets.append(target)
        
    def def_target(self, frame, number, constraint, name=""):
        target = Target(frame, number, constraint, name=name)
        self.add_target(target)
        return target
        
    def pack(self, output_file=None, seed=None, **kwargs):
        # Mock implementation that returns a frame with repositioned atoms
        if not self.targets:
            return mp.Frame()
        
        # For testing, just return the first target's frame
        return self.targets[0].frame
        
    @property
    def n_points(self):
        return sum([t.n_points for t in self.targets])
from molpy.pack import constraint as mpk_constraint


class TestSession:
    """Test Session class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple frame with atoms
        self.frame = mp.Frame()
        atoms_data = {
            'id': [0, 1, 2],
            'name': ['C1', 'C2', 'C3'],
            'x': [0.0, 1.0, 2.0],
            'y': [0.0, 0.0, 0.0],
            'z': [0.0, 0.0, 0.0]
        }
        self.frame["atoms"] = atoms_data

        # Create a simple constraint
        self.constraint = mpk_constraint.InsideBoxConstraint([5.0, 5.0, 5.0], [0.0, 0.0, 0.0])

    def test_session_initialization_default_packer(self):
        """Test Session initialization with default packer."""
        session = Session()
        
        assert len(session.targets) == 0
        # Default should be "packmol" but we can't test actual packer without packmol installed
        assert session.packer is not None

    def test_session_initialization_custom_packer(self):
        """Test Session initialization with custom packer."""
        # Test with nlopt (even if not available, should try to create)
        try:
            session = Session(packer="nlopt")
            assert session.packer is not None
        except ImportError:
            # nlopt not available, which is fine for testing
            pass

    def test_session_initialization_invalid_packer(self):
        """Test Session initialization with invalid packer."""
        with pytest.raises(NotImplementedError, match="Optimizer invalid_packer not implemented"):
            Session(packer="invalid_packer")

    def test_session_add_target(self):
        """Test adding targets to session."""
        session = Session()
        
        # Add first target
        session.add_target(self.frame, 2, self.constraint)
        assert len(session.targets) == 1
        
        target = session.targets[0]
        assert isinstance(target, Target)
        assert target.frame == self.frame
        assert target.number == 2
        assert target.constraint == self.constraint

    def test_session_add_multiple_targets(self):
        """Test adding multiple targets to session."""
        session = Session()
        
        # Create different constraints
        box_constraint = mpk_constraint.InsideBoxConstraint([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])
        sphere_constraint = mpk_constraint.InsideSphereConstraint(1.5, [0.0, 0.0, 0.0])
        
        # Add multiple targets
        session.add_target(self.frame, 2, box_constraint)
        session.add_target(self.frame, 3, sphere_constraint)
        
        assert len(session.targets) == 2
        
        # Check first target
        target1 = session.targets[0]
        assert target1.number == 2
        assert target1.constraint == box_constraint
        
        # Check second target
        target2 = session.targets[1]
        assert target2.number == 3
        assert target2.constraint == sphere_constraint

    def test_session_add_target_with_different_frames(self):
        """Test adding targets with different molecular frames."""
        session = Session()
        
        # Create a different frame
        frame2 = mp.Frame()
        atoms_data2 = {
            'id': [0, 1],
            'name': ['O1', 'H1'],
            'x': [0.0, 1.0],
            'y': [0.0, 0.0],
            'z': [0.0, 0.0]
        }
        frame2["atoms"] = atoms_data2
        
        # Add targets with different frames
        session.add_target(self.frame, 2, self.constraint)
        session.add_target(frame2, 1, self.constraint)
        
        assert len(session.targets) == 2
        assert session.targets[0].frame == self.frame
        assert session.targets[1].frame == frame2

    def test_session_optimize_default_parameters(self):
        """Test optimize method with default parameters."""
        session = Session()
        session.add_target(self.frame, 1, self.constraint)
        
        # Mock the packer to avoid actually running packmol
        class MockPacker:
            def pack(self, targets, max_steps, seed):
                # Return a simple frame
                result_frame = mp.Frame()
                result_frame["atoms"] = targets[0].frame["atoms"]
                return result_frame
        
        session.packer = MockPacker()
        
        result = session.optimize()
        assert isinstance(result, mp.Frame)

    def test_session_optimize_custom_parameters(self):
        """Test optimize method with custom parameters."""
        session = Session()
        session.add_target(self.frame, 1, self.constraint)
        
        # Mock the packer to capture parameters
        class MockPacker:
            def __init__(self):
                self.last_max_steps = None
                self.last_seed = None
                
            def pack(self, targets, max_steps, seed):
                self.last_max_steps = max_steps
                self.last_seed = seed
                result_frame = mp.Frame()
                result_frame["atoms"] = targets[0].frame["atoms"]
                return result_frame
        
        mock_packer = MockPacker()
        session.packer = mock_packer
        
        # Test with custom parameters
        result = session.optimize(max_steps=500, seed=12345)
        
        assert isinstance(result, mp.Frame)
        assert mock_packer.last_max_steps == 500
        assert mock_packer.last_seed == 12345

    def test_session_optimize_random_seed(self):
        """Test optimize method with random seed generation."""
        session = Session()
        session.add_target(self.frame, 1, self.constraint)
        
        # Mock the packer to capture seed
        class MockPacker:
            def __init__(self):
                self.captured_seeds = []
                
            def pack(self, targets, max_steps, seed):
                self.captured_seeds.append(seed)
                result_frame = mp.Frame()
                result_frame["atoms"] = targets[0].frame["atoms"]
                return result_frame
        
        mock_packer = MockPacker()
        session.packer = mock_packer
        
        # Run optimize twice with None seed (should generate random seeds)
        session.optimize(seed=None)
        session.optimize(seed=None)
        
        # Should have captured two different seeds
        assert len(mock_packer.captured_seeds) == 2
        assert mock_packer.captured_seeds[0] != mock_packer.captured_seeds[1]
        # Seeds should be in valid range
        for seed in mock_packer.captured_seeds:
            assert 1 <= seed <= 10000

    def test_session_optimize_no_targets(self):
        """Test optimize method when no targets are added."""
        session = Session()
        
        # Mock the packer
        class MockPacker:
            def pack(self, targets, max_steps, seed):
                assert len(targets) == 0
                return mp.Frame()
        
        session.packer = MockPacker()
        
        # Should handle empty targets list
        result = session.optimize()
        assert isinstance(result, mp.Frame)

    def test_session_workflow_integration(self):
        """Test complete workflow integration."""
        session = Session()
        
        # Create multiple molecular species
        water_frame = mp.Frame()
        water_atoms = {
            'id': [0, 1, 2],
            'name': ['O', 'H1', 'H2'],
            'x': [0.0, 0.7, -0.7],
            'y': [0.0, 0.5, 0.5],
            'z': [0.0, 0.0, 0.0]
        }
        water_frame["atoms"] = water_atoms
        
        co2_frame = mp.Frame()
        co2_atoms = {
            'id': [0, 1, 2],
            'name': ['C', 'O1', 'O2'],
            'x': [0.0, 1.2, -1.2],
            'y': [0.0, 0.0, 0.0],
            'z': [0.0, 0.0, 0.0]
        }
        co2_frame["atoms"] = co2_atoms
        
        # Create constraints
        box_constraint = mpk_constraint.InsideBoxConstraint([10.0, 10.0, 10.0], [0.0, 0.0, 0.0])
        sphere_exclusion = mpk_constraint.OutsideSphereConstraint(2.0, [5.0, 5.0, 5.0])
        
        # Add targets
        session.add_target(water_frame, 50, box_constraint & sphere_exclusion)
        session.add_target(co2_frame, 20, box_constraint)
        
        assert len(session.targets) == 2
        assert session.targets[0].number == 50
        assert session.targets[1].number == 20
        
        # Mock optimization
        class MockPacker:
            def pack(self, targets, max_steps, seed):
                # Verify targets are passed correctly
                assert len(targets) == 2
                assert targets[0].number == 50
                assert targets[1].number == 20
                
                # Create mock result
                result_frame = mp.Frame()
                result_atoms = {
                    'id': list(range(210)),  # 50*3 + 20*3 = 210 atoms
                    'x': [i * 0.1 for i in range(210)],
                    'y': [0.0] * 210,
                    'z': [0.0] * 210
                }
                result_frame["atoms"] = result_atoms
                return result_frame
        
        session.packer = MockPacker()
        
        result = session.optimize(max_steps=2000, seed=54321)
        assert isinstance(result, mp.Frame)
        assert len(result["atoms"]["id"]) == 210

    def test_session_target_access(self):
        """Test accessing targets from session."""
        session = Session()
        
        # Add some targets
        constraint1 = mpk_constraint.InsideBoxConstraint([3.0, 3.0, 3.0], [0.0, 0.0, 0.0])
        constraint2 = mpk_constraint.InsideSphereConstraint(2.0, [1.0, 1.0, 1.0])
        
        session.add_target(self.frame, 5, constraint1)
        session.add_target(self.frame, 3, constraint2)
        
        # Access targets directly
        targets = session.targets
        assert len(targets) == 2
        assert targets[0].number == 5
        assert targets[1].number == 3
        
        # Targets should be Target instances
        for target in targets:
            assert isinstance(target, Target)

    def test_session_edge_cases(self):
        """Test edge cases for Session."""
        session = Session()
        
        # Add target with zero molecules
        session.add_target(self.frame, 0, self.constraint)
        assert len(session.targets) == 1
        assert session.targets[0].number == 0
        
        # Mock packer to handle zero molecules
        class MockPacker:
            def pack(self, targets, max_steps, seed):
                assert targets[0].number == 0
                return mp.Frame()
        
        session.packer = MockPacker()
        result = session.optimize()
        assert isinstance(result, mp.Frame)
