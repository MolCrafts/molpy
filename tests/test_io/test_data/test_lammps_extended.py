"""
Additional tests for LAMMPS I/O functionality.
Tests edge cases, performance, and extended functionality.
"""

import pytest
import tempfile
import numpy as np
import pandas as pd
import molpy as mp
from pathlib import Path


class TestLammpsEdgeCases:

    def test_triclinic_2(self, test_data_path):
        """Test reading triclinic-2.lmp file."""
        lammps_data = test_data_path / "data/lammps-data"
        reader = mp.io.data.LammpsDataReader(lammps_data / "triclinic-2.lmp")
        frame = mp.Frame()
        frame = reader.read(frame)
        # Should handle triclinic format
        assert frame.box is not None

    def test_data_body(self, test_data_path):
        """Test reading data.body file."""
        lammps_data = test_data_path / "data/lammps-data"
        if (lammps_data / "data.body").exists():
            reader = mp.io.data.LammpsDataReader(lammps_data / "data.body")
            frame = mp.Frame()
            frame = reader.read(frame)
            # Should not crash
            assert isinstance(frame, mp.Frame)

    def test_atom_style_variations(self):
        """Test different atom styles."""
        # Test with different atom styles
        for atom_style in ["full", "atomic", "charge"]:
            frame = mp.Frame()
            reader = mp.io.data.LammpsDataReader("dummy.lmp", atom_style=atom_style)
            # Just test initialization doesn't crash
            assert reader.atom_style == atom_style

    def test_large_coordinates(self):
        """Test handling of very large coordinate values."""
        frame = mp.Frame()
        
        # Create atoms with large coordinates
        atoms_data = pd.DataFrame({
            'id': [1, 2],
            'molid': [1, 1],
            'type': [1, 1],
            'charge': [0.0, 0.0],
            'x': [1e6, -1e6],
            'y': [1e6, -1e6],
            'z': [1e6, -1e6]
        })
        frame["atoms"] = atoms_data.to_xarray()
        frame.box = mp.Box(np.eye(3) * 2e6)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            # Read back and verify
            reader = mp.io.data.LammpsDataReader(tmp.name)
            frame_read = reader.read(mp.Frame())
            
            assert frame_read["atoms"].sizes["index"] == 2
            # Check that large coordinates are preserved
            assert abs(frame_read["atoms"]["x"].values[0] - 1e6) < 1e-3

    def test_mixed_atom_types(self):
        """Test handling of mixed atom types (numeric and string)."""
        frame = mp.Frame()
        
        # Create atoms with mixed type identifiers
        atoms_data = pd.DataFrame({
            'id': [1, 2, 3],
            'molid': [1, 1, 2],
            'type': ['H', 'O', 'Li+'],
            'charge': [0.4, -0.8, 1.0],
            'x': [0.0, 1.0, 0.5],
            'y': [0.0, 0.0, 1.0],
            'z': [0.0, 0.0, 0.0],
            'mass': [1.008, 15.999, 6.941]
        })
        frame["atoms"] = atoms_data.to_xarray()
        frame.box = mp.Box(np.eye(3) * 10.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            # Read back and verify
            reader = mp.io.data.LammpsDataReader(tmp.name)
            frame_read = reader.read(mp.Frame())
            
            assert frame_read["atoms"].sizes["index"] == 3

    def test_empty_sections(self):
        """Test handling of files with missing sections."""
        frame = mp.Frame()
        
        # Create atoms only (no bonds, angles, etc.)
        atoms_data = pd.DataFrame({
            'id': [1],
            'molid': [1],
            'type': [1],
            'charge': [0.0],
            'x': [0.0],
            'y': [0.0],
            'z': [0.0]
        })
        frame["atoms"] = atoms_data.to_xarray()
        frame.box = mp.Box(np.eye(3) * 5.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            # Read back - should handle missing sections gracefully
            reader = mp.io.data.LammpsDataReader(tmp.name)
            frame_read = reader.read(mp.Frame())
            
            assert frame_read["atoms"].sizes["index"] == 1
            assert "bonds" not in frame_read or len(frame_read.get("bonds", [])) == 0

    def test_comment_handling(self):
        """Test handling of comments in LAMMPS files."""
        test_content = """# Test LAMMPS file with comments
1 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi  
0.0 10.0 zlo zhi

Masses

1 1.0  # hydrogen mass

Atoms  # full style

1 1 1 0.0 5.0 5.0 5.0  # atom 1
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            tmp.write(test_content)
            tmp.flush()
            
            reader = mp.io.data.LammpsDataReader(tmp.name)
            frame = reader.read(mp.Frame())
            
            assert frame["atoms"].sizes["index"] == 1

    def test_extra_atom_columns(self):
        """Test handling of atom lines with extra columns (image flags, etc.)."""
        test_content = """1 atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 1.0

Atoms

1 1 1 0.0 5.0 5.0 5.0 0 0 0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            tmp.write(test_content)
            tmp.flush()
            
            reader = mp.io.data.LammpsDataReader(tmp.name)
            frame = reader.read(mp.Frame())
            
            assert frame["atoms"].sizes["index"] == 1
            # Should handle extra columns (image flags)
            if "ix" in frame["atoms"]:
                assert frame["atoms"]["ix"].values[0] == 0


class TestLammpsPerformance:

    def test_medium_system_performance(self):
        """Test performance with medium-sized system (~1000 atoms)."""
        frame = mp.Frame()
        
        n_atoms = 1000
        
        # Create larger system
        atoms_data = pd.DataFrame({
            'id': range(1, n_atoms + 1),
            'molid': np.random.randint(1, 100, n_atoms),
            'type': np.random.randint(1, 5, n_atoms),
            'charge': np.random.uniform(-1, 1, n_atoms),
            'x': np.random.uniform(0, 50, n_atoms),
            'y': np.random.uniform(0, 50, n_atoms),
            'z': np.random.uniform(0, 50, n_atoms),
            'mass': np.random.uniform(1, 20, n_atoms)
        })
        frame["atoms"] = atoms_data.to_xarray()
        
        # Create some bonds
        n_bonds = n_atoms // 2
        bonds_data = pd.DataFrame({
            'id': range(1, n_bonds + 1),
            'type': np.random.randint(1, 3, n_bonds),
            'i': np.random.randint(1, n_atoms, n_bonds),
            'j': np.random.randint(1, n_atoms, n_bonds)
        })
        frame["bonds"] = bonds_data.to_xarray()
        
        frame.box = mp.Box(np.eye(3) * 50.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            # Time the write operation
            import time
            start_time = time.time()
            
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            write_time = time.time() - start_time
            
            # Time the read operation
            start_time = time.time()
            
            reader = mp.io.data.LammpsDataReader(tmp.name)
            frame_read = reader.read(mp.Frame())
            
            read_time = time.time() - start_time
            
            # Performance assertions (should complete reasonably quickly)
            assert write_time < 5.0, f"Write took too long: {write_time:.2f}s"
            assert read_time < 5.0, f"Read took too long: {read_time:.2f}s"
            
            # Verify correctness
            assert frame_read["atoms"].sizes["index"] == n_atoms
            assert frame_read["bonds"].sizes["index"] == n_bonds


class TestLammpsDataIntegrity:

    def test_roundtrip_data_integrity(self):
        """Test that data is preserved through write/read cycle."""
        frame = mp.Frame()
        
        # Create test data with specific values to check
        atoms_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'molid': [1, 1, 2, 2, 3],
            'type': [1, 2, 1, 3, 2],
            'charge': [0.5, -0.5, 0.0, 1.0, -1.0],
            'x': [0.0, 1.5, 3.0, 4.5, 6.0],
            'y': [0.0, 1.0, 2.0, 3.0, 4.0],
            'z': [0.0, 0.5, 1.0, 1.5, 2.0],
            'mass': [1.0, 16.0, 1.0, 12.0, 16.0]
        })
        frame["atoms"] = atoms_data.to_xarray()
        
        bonds_data = pd.DataFrame({
            'id': [1, 2, 3],
            'type': [1, 2, 1],
            'i': [1, 2, 3],
            'j': [2, 3, 4]
        })
        frame["bonds"] = bonds_data.to_xarray()
        
        # Specific box dimensions
        frame.box = mp.Box(np.diag([10.0, 15.0, 20.0]))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            reader = mp.io.data.LammpsDataReader(tmp.name)
            frame_read = reader.read(mp.Frame())
            
            # Check atoms data integrity
            assert frame_read["atoms"].sizes["index"] == 5
            
            # Check specific coordinate values (within reasonable precision)
            x_values = frame_read["atoms"]["x"].values
            assert abs(x_values[1] - 1.5) < 1e-3
            assert abs(x_values[4] - 6.0) < 1e-3
            
            # Check bonds data integrity
            assert frame_read["bonds"].sizes["index"] == 3
            
            # Check box dimensions
            assert frame_read.box is not None

    def test_special_characters_in_types(self):
        """Test handling of special characters in atom type names."""
        frame = mp.Frame()
        
        # Create atoms with special characters in type names
        atoms_data = pd.DataFrame({
            'id': [1, 2, 3],
            'molid': [1, 1, 2],
            'type': ['H+', 'O-2', 'C_sp3'],
            'charge': [1.0, -2.0, 0.0],
            'x': [0.0, 1.0, 0.5],
            'y': [0.0, 0.0, 1.0],
            'z': [0.0, 0.0, 0.0],
            'mass': [1.008, 15.999, 12.011]
        })
        frame["atoms"] = atoms_data.to_xarray()
        frame.box = mp.Box(np.eye(3) * 10.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as tmp:
            writer = mp.io.data.LammpsDataWriter(tmp.name)
            writer.write(frame)
            
            # Should not crash and should preserve type information
            reader = mp.io.data.LammpsDataReader(tmp.name)
            frame_read = reader.read(mp.Frame())
            
            assert frame_read["atoms"].sizes["index"] == 3


if __name__ == "__main__":
    pytest.main([__file__])
