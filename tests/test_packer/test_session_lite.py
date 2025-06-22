"""
Lightweight tests for Session that work without external dependencies.
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


class TestSessionLite:
    """Lightweight Session tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.frame = mp.Frame()
        self.frame["atoms"] = {
            'id': [0, 1, 2],
            'x': [0.0, 1.0, 2.0],
            'y': [0.0, 0.0, 0.0],
            'z': [0.0, 0.0, 0.0]
        }
        
        self.constraint = mpk_constraint.InsideBoxConstraint([5.0, 5.0, 5.0], [0.0, 0.0, 0.0])

    def test_session_with_mock_packer(self):
        """Test Session functionality with mock packer."""
        # Create session with mock packer
        session = Session.__new__(Session)  # Create without calling __init__
        session.targets = []
        session.packer = MockPacker()
        
        # Add target
        target = session.add_target(
            frame=self.frame,
            number=2,
            constraint=self.constraint
        )
        
        assert isinstance(target, Target)
        assert target.number == 2
        assert len(session.targets) == 1
        assert len(session.packer.targets) == 1
        
    def test_session_target_properties(self):
        """Test target properties through session."""
        session = Session.__new__(Session)
        session.targets = []
        session.packer = MockPacker()
        
        target = session.add_target(
            frame=self.frame,
            number=3,
            constraint=self.constraint
        )
        
        # Check target properties
        assert target.n_points == 9  # 3 atoms * 3 number
        assert len(session.targets) == 1
        assert len(session.packer.targets) == 1
        assert session.packer.n_points == 9
