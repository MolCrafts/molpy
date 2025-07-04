"""Core XSF format tests - simplified version with only essential functionality."""

import tempfile
import numpy as np
from pathlib import Path
import pytest
import molpy as mp


class TestXSFCore:
    """Core XSF functionality tests."""

    def test_read_crystal_structure(self):
        """Test reading a crystal structure with periodic box."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xsf', delete=False) as tmp:
            tmp.write("CRYSTAL\n")
            tmp.write("PRIMVEC\n")
            tmp.write("3.0 0.0 0.0\n")
            tmp.write("0.0 3.0 0.0\n")
            tmp.write("0.0 0.0 3.0\n")
            tmp.write("PRIMCOORD\n")
            tmp.write("2 1\n")
            tmp.write("1  0.0  0.0  0.0\n")
            tmp.write("8  1.5  1.5  1.5\n")
            
        system = mp.io.read_xsf(Path(tmp.name))
        frame = system._wrapped
        
        # Check atoms
        assert len(frame["atoms"]["atomic_number"]) == 2
        assert frame["atoms"]["atomic_number"][0] == 1  # Hydrogen
        assert frame["atoms"]["atomic_number"][1] == 8  # Oxygen
        
        # Check box
        assert system.box.style == mp.Box.Style.ORTHOGONAL
        np.testing.assert_array_almost_equal(system.box.matrix, np.diag([3.0, 3.0, 3.0]))

    def test_read_molecule_structure(self):
        """Test reading a molecule structure (non-periodic)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xsf', delete=False) as tmp:
            tmp.write("MOLECULE\n")
            tmp.write("PRIMCOORD\n")
            tmp.write("2 1\n")
            tmp.write("1  0.0  0.0  0.0\n")
            tmp.write("1  1.0  0.0  0.0\n")
            
        system = mp.io.read_xsf(Path(tmp.name))
        frame = system._wrapped
        
        # Check atoms
        assert len(frame["atoms"]["atomic_number"]) == 2
        assert all(an == 1 for an in frame["atoms"]["atomic_number"])
        
        # Should have a free box for molecule (non-periodic)
        assert system.box.style == mp.Box.Style.FREE

    def test_write_crystal_structure(self):
        """Test writing a crystal structure."""
        # Create test system
        frame = mp.Frame()
        frame["atoms"] = mp.Block({
            'atomic_number': np.array([1, 8]),
            'xyz': np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]),
            'element': np.array(['H', 'O']),
            'x': np.array([0.0, 1.5]),
            'y': np.array([0.0, 1.5]),
            'z': np.array([0.0, 1.5])
        })
        
        box = mp.Box(matrix=np.diag([3.0, 3.0, 3.0]))
        system = mp.FrameSystem(frame=frame, box=box)
        
        # Write to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xsf', delete=False) as tmp:
            mp.io.write_xsf(tmp.name, system)
            
        # Read back and verify
        system2 = mp.io.read_xsf(Path(tmp.name))
        frame2 = system2._wrapped
        
        # Check atoms
        assert len(frame2["atoms"]["atomic_number"]) == 2
        np.testing.assert_array_equal(frame2["atoms"]["atomic_number"], [1, 8])
        
        # Check box
        assert system2.box.style == mp.Box.Style.ORTHOGONAL
        np.testing.assert_array_almost_equal(system2.box.matrix, np.diag([3.0, 3.0, 3.0]))

    def test_write_molecule_structure(self):
        """Test writing a molecule structure."""
        # Create test system
        frame = mp.Frame()
        frame["atoms"] = mp.Block({
            'atomic_number': np.array([1, 1]),
            'xyz': np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            'element': np.array(['H', 'H']),
            'x': np.array([0.0, 1.0]),
            'y': np.array([0.0, 0.0]),
            'z': np.array([0.0, 0.0])
        })
        
        # Create with free box
        box = mp.Box()  # Free box
        system = mp.FrameSystem(frame=frame, box=box)
        
        # Write to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xsf', delete=False) as tmp:
            mp.io.write_xsf(tmp.name, system)
            
        # Read back and verify
        system2 = mp.io.read_xsf(Path(tmp.name))
        frame2 = system2._wrapped
        
        # Check atoms
        assert len(frame2["atoms"]["atomic_number"]) == 2
        assert all(an == 1 for an in frame2["atoms"]["atomic_number"])
        
        # Should have free box
        assert system2.box.style == mp.Box.Style.FREE

    def test_roundtrip_consistency(self):
        """Test that write->read maintains data consistency."""
        # Original system
        frame = mp.Frame()
        frame["atoms"] = mp.Block({
            'atomic_number': np.array([6, 1, 1, 1, 1]),
            'xyz': np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0]
            ]),
            'element': np.array(['C', 'H', 'H', 'H', 'H']),
            'x': np.array([0.0, 1.0, -1.0, 0.0, 0.0]),
            'y': np.array([0.0, 0.0, 0.0, 1.0, -1.0]),
            'z': np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        })
        
        box = mp.Box(matrix=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
        system1 = mp.FrameSystem(frame=frame, box=box)
        
        # Write and read back
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xsf', delete=False) as tmp:
            mp.io.write_xsf(tmp.name, system1)
            system2 = mp.io.read_xsf(Path(tmp.name))
            
        frame1 = system1._wrapped
        frame2 = system2._wrapped
        
        # Check consistency
        np.testing.assert_array_equal(
            frame1["atoms"]["atomic_number"], 
            frame2["atoms"]["atomic_number"]
        )
        np.testing.assert_array_almost_equal(
            frame1["atoms"]["xyz"], 
            frame2["atoms"]["xyz"]
        )
        np.testing.assert_array_almost_equal(
            system1.box.matrix, 
            system2.box.matrix
        )

    def test_error_handling(self):
        """Test basic error handling."""
        # Test reading non-existent file
        with pytest.raises(FileNotFoundError):
            mp.io.read_xsf("nonexistent.xsf")
            
        # Test reading empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xsf', delete=False) as tmp:
            tmp.write("")  # Empty file
            
        with pytest.raises(ValueError, match="Empty XSF file"):
            mp.io.read_xsf(Path(tmp.name))
            
        # Test malformed PRIMCOORD section
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xsf', delete=False) as tmp:
            tmp.write("MOLECULE\n")
            tmp.write("PRIMCOORD\n")
            tmp.write("invalid_number 1\n")  # Invalid atom count
            
        with pytest.raises(ValueError):
            mp.io.read_xsf(Path(tmp.name))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
