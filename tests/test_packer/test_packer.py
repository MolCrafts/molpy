"""
Unit tests for the packer.packer module.
Tests base packer functionality and implementations.
"""

import pytest
import numpy as np
import tempfile
import molpy as mp
from pathlib import Path
from molpy.pack.packer.base import Packer
from molpy.pack.packer import get_packer
from molpy.pack.target import Target
from molpy.pack import constraint as mpk_constraint


class TestGetPacker:
    """Test the get_packer factory function."""

    def test_get_packer_packmol(self):
        """Test getting packmol packer."""
        try:
            packer = get_packer("packmol")
            from molpy.pack.packer.packmol import Packmol
            assert isinstance(packer, Packmol)
        except FileNotFoundError:
            # Packmol not installed, which is fine for testing
            pytest.skip("Packmol not available")

    def test_get_packer_nlopt(self):
        """Test getting nlopt packer."""
        try:
            packer = get_packer("nlopt")
            from molpy.pack.packer.nlopt import NloptPacker
            assert isinstance(packer, NloptPacker)
        except ImportError:
            # nlopt not installed, which is fine for testing
            pytest.skip("nlopt not available")

    def test_get_packer_invalid(self):
        """Test getting invalid packer."""
        with pytest.raises(NotImplementedError, match="Optimizer invalid not implemented"):
            get_packer("invalid")

    def test_get_packer_with_args(self):
        """Test getting packer with additional arguments."""
        try:
            # Try with custom workdir for packmol
            with tempfile.TemporaryDirectory() as tmpdir:
                packer = get_packer("packmol", workdir=Path(tmpdir))
                assert packer.workdir == Path(tmpdir)
        except FileNotFoundError:
            pytest.skip("Packmol not available")


class MockPacker(Packer):
    """Mock packer implementation for testing base functionality."""
    
    def __init__(self):
        super().__init__()
        self.pack_called = False
        self.pack_args = None
        
    def pack(self, targets=None, max_steps: int = 1000, seed: int | None = None):
        self.pack_called = True
        self.pack_args = (targets, max_steps, seed)
        
        # Return a simple frame
        result_frame = mp.Frame()
        if targets and len(targets) > 0:
            # Combine all atoms from targets
            all_atoms = []
            for target in targets:
                for i in range(target.number):
                    atoms = target.frame["atoms"].copy()
                    all_atoms.append(atoms)
            
            if all_atoms:
                import xarray as xr
                result_frame["atoms"] = xr.concat(all_atoms, dim="index")
        
        return result_frame


class TestBasePacker:
    """Test the base Packer class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.packer = MockPacker()
        
        # Create test frame
        self.frame = mp.Frame()
        atoms_data = {
            'id': [0, 1, 2],
            'name': ['C1', 'C2', 'C3'],
            'x': [0.0, 1.0, 2.0],
            'y': [0.0, 0.0, 0.0],
            'z': [0.0, 0.0, 0.0]
        }
        self.frame["atoms"] = atoms_data
        
        # Create test constraint
        self.constraint = mpk_constraint.InsideBoxConstraint([5.0, 5.0, 5.0], [0.0, 0.0, 0.0])

    def test_packer_initialization(self):
        """Test packer initialization."""
        assert len(self.packer.targets) == 0
        assert not self.packer.pack_called

    def test_packer_add_target(self):
        """Test adding targets to packer."""
        target = Target(self.frame, 2, self.constraint, name="test")
        
        self.packer.add_target(target)
        assert len(self.packer.targets) == 1
        assert self.packer.targets[0] == target

    def test_packer_add_multiple_targets(self):
        """Test adding multiple targets to packer."""
        target1 = Target(self.frame, 2, self.constraint, name="test1")
        target2 = Target(self.frame, 3, self.constraint, name="test2")
        
        self.packer.add_target(target1)
        self.packer.add_target(target2)
        
        assert len(self.packer.targets) == 2
        assert self.packer.targets[0] == target1
        assert self.packer.targets[1] == target2

    def test_packer_def_target(self):
        """Test def_target method."""
        target = self.packer.def_target(
            frame=self.frame,
            number=4,
            constraint=self.constraint,
            is_fixed=True,
            name="defined_target"
        )
        
        assert isinstance(target, Target)
        assert target.frame == self.frame
        assert target.number == 4
        assert target.constraint == self.constraint
        assert target.is_fixed is True
        assert target.name == "defined_target"
        
        # Should also be added to targets list
        assert len(self.packer.targets) == 1
        assert self.packer.targets[0] == target

    def test_packer_def_target_defaults(self):
        """Test def_target method with default parameters."""
        target = self.packer.def_target(
            frame=self.frame,
            number=2,
            constraint=self.constraint
        )
        
        assert target.is_fixed is False
        assert target.optimizer is None
        assert target.name == ""

    def test_packer_n_points_property(self):
        """Test n_points property calculation."""
        # Add targets with different numbers
        target1 = Target(self.frame, 2, self.constraint)  # 3 atoms * 2 = 6 points
        target2 = Target(self.frame, 3, self.constraint)  # 3 atoms * 3 = 9 points
        
        self.packer.add_target(target1)
        self.packer.add_target(target2)
        
        assert self.packer.n_points == 15  # 6 + 9 = 15

    def test_packer_n_points_empty(self):
        """Test n_points property with no targets."""
        assert self.packer.n_points == 0

    def test_packer_points_property(self):
        """Test points property."""
        # Create targets with known coordinates
        frame1 = mp.Frame()
        frame1["atoms"] = {
            'id': [0, 1],
            'x': [0.0, 1.0],
            'y': [0.0, 0.0],
            'z': [0.0, 0.0]
        }
        
        frame2 = mp.Frame()
        frame2["atoms"] = {
            'id': [0],
            'x': [2.0],
            'y': [2.0],
            'z': [2.0]
        }
        
        target1 = Target(frame1, 1, self.constraint)
        target2 = Target(frame2, 1, self.constraint)
        
        self.packer.add_target(target1)
        self.packer.add_target(target2)
        
        points = self.packer.points
        assert isinstance(points, np.ndarray)
        assert points.shape == (3, 3)  # 3 total atoms, 3 coordinates each
        
        # Check concatenated points
        expected = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 2.0, 2.0]
        ])
        assert np.allclose(points, expected)

    def test_packer_points_empty(self):
        """Test points property with no targets."""
        points = self.packer.points
        assert isinstance(points, np.ndarray)
        assert points.shape == (0, 3)  # Empty array with correct shape

    def test_packer_pack_method(self):
        """Test pack method is called correctly."""
        target = Target(self.frame, 2, self.constraint)
        self.packer.add_target(target)
        
        result = self.packer.pack(max_steps=500, seed=12345)
        
        assert self.packer.pack_called
        targets, max_steps, seed = self.packer.pack_args
        assert targets is None  # Default behavior
        assert max_steps == 500
        assert seed == 12345
        assert isinstance(result, mp.Frame)

    def test_packer_pack_with_explicit_targets(self):
        """Test pack method with explicit targets."""
        target1 = Target(self.frame, 1, self.constraint)
        target2 = Target(self.frame, 2, self.constraint)
        
        # Add to packer but also pass explicitly
        self.packer.add_target(target1)
        custom_targets = [target2]
        
        result = self.packer.pack(targets=custom_targets, max_steps=100)
        
        assert self.packer.pack_called
        targets, max_steps, seed = self.packer.pack_args
        assert targets == custom_targets
        assert max_steps == 100

    def test_packer_workflow_integration(self):
        """Test complete packer workflow."""
        # Create multiple molecular species
        water_frame = mp.Frame()
        water_frame["atoms"] = {
            'id': [0, 1, 2],
            'name': ['O', 'H1', 'H2'],
            'x': [0.0, 0.7, -0.7],
            'y': [0.0, 0.5, 0.5],
            'z': [0.0, 0.0, 0.0]
        }
        
        methane_frame = mp.Frame()
        methane_frame["atoms"] = {
            'id': [0, 1, 2, 3, 4],
            'name': ['C', 'H1', 'H2', 'H3', 'H4'],
            'x': [0.0, 1.0, -1.0, 0.0, 0.0],
            'y': [0.0, 0.0, 0.0, 1.0, -1.0],
            'z': [0.0, 0.0, 0.0, 0.0, 0.0]
        }
        
        # Define targets
        box_constraint = mpk_constraint.InsideBoxConstraint([10.0, 10.0, 10.0], [0.0, 0.0, 0.0])
        
        water_target = self.packer.def_target(
            frame=water_frame,
            number=10,
            constraint=box_constraint,
            name="water"
        )
        
        methane_target = self.packer.def_target(
            frame=methane_frame,
            number=5,
            constraint=box_constraint,
            name="methane"
        )
        
        # Check packer state
        assert len(self.packer.targets) == 2
        assert self.packer.n_points == 55  # 10*3 + 5*5 = 55
        
        # Run packing
        result = self.packer.pack(max_steps=1000, seed=42)
        assert isinstance(result, mp.Frame)

    def test_packer_edge_cases(self):
        """Test edge cases for packer."""
        # Empty frame
        empty_frame = mp.Frame()
        empty_frame["atoms"] = {'id': [], 'x': [], 'y': [], 'z': []}
        
        target = Target(empty_frame, 5, self.constraint)
        self.packer.add_target(target)
        
        assert self.packer.n_points == 0  # 0 atoms * 5 = 0
        
        points = self.packer.points
        assert points.shape == (0, 3)
        
        # Should still be able to pack
        result = self.packer.pack()
        assert isinstance(result, mp.Frame)

    def test_packer_target_modification(self):
        """Test that modifying targets after adding doesn't affect packer."""
        target = Target(self.frame, 2, self.constraint, name="original")
        self.packer.add_target(target)
        
        original_name = self.packer.targets[0].name
        
        # Modify target name
        target.name = "modified"
        
        # Packer should reference the same object (modification visible)
        assert self.packer.targets[0].name == "modified"

    def test_packer_with_complex_constraints(self):
        """Test packer with complex constraint combinations."""
        # Create complex constraint
        box = mpk_constraint.InsideBoxConstraint([8.0, 8.0, 8.0], [0.0, 0.0, 0.0])
        sphere = mpk_constraint.OutsideSphereConstraint(2.0, [4.0, 4.0, 4.0])
        min_dist = mpk_constraint.MinDistanceConstraint(1.0)
        
        complex_constraint = box & sphere & min_dist
        
        target = Target(self.frame, 3, complex_constraint, name="complex")
        self.packer.add_target(target)
        
        # Should handle complex constraints
        assert len(self.packer.targets) == 1
        assert self.packer.targets[0].constraint == complex_constraint
        
        result = self.packer.pack()
        assert isinstance(result, mp.Frame)
