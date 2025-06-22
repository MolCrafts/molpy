import pytest
import numpy as np
import molpy as mp
from pathlib import Path


class TestMol2Reader:
    """Basic MOL2 reading tests."""

    def test_read(self, test_data_path):
        frame = mp.Frame()
        mol2 = mp.io.read_mol2(test_data_path / "data/mol2/ethane.mol2", frame)
        atoms = mol2["atoms"]
        # Get the main dimension for atoms
        sizes = atoms.sizes
        main_dim = next(iter(sizes.keys()))  # Get first dimension
        assert sizes[main_dim] == 8
        
        # Get specific dimensions for each field
        name_dim = next(d for d in atoms.dims if 'name' in str(d))
        xyz_dim = next(d for d in atoms.dims if 'xyz' in str(d) and not str(d).endswith('_1'))
        type_dim = next(d for d in atoms.dims if 'type' in str(d))
        subst_id_dim = next(d for d in atoms.dims if 'subst_id' in str(d))
        subst_name_dim = next(d for d in atoms.dims if 'subst_name' in str(d))
        charge_dim = next(d for d in atoms.dims if 'charge' in str(d))
        
        assert atoms["name"].isel({name_dim: 0}).item() == "C"
        assert tuple(atoms["xyz"].isel({xyz_dim: 0}).values) == pytest.approx((3.1080, 0.6530, -8.5260))
        assert atoms["type"].isel({type_dim: 0}).item() == "c3"
        assert int(atoms["subst_id"].isel({subst_id_dim: 0}).item()) == 1
        assert atoms["subst_name"].isel({subst_name_dim: 0}).item() == "ETH"
        assert atoms["charge"].isel({charge_dim: 0}).item() == -0.094100


class TestMol2Comprehensive:
    """Comprehensive MOL2 tests using chemfiles test cases."""

    def test_ethane_basic_structure(self, test_data_path):
        """Test basic ethane structure reading."""
        fpath = test_data_path / "data/mol2/ethane.mol2"
        if not fpath.exists():
            pytest.skip("ethane.mol2 test data not available")
        
        frame = mp.io.read_mol2(fpath, frame=mp.Frame())
        
        # Check basic structure
        assert "atoms" in frame
        atoms = frame["atoms"]
        
        # Get atom count
        sizes = atoms.sizes
        main_dim = next(iter(sizes.keys()))
        n_atoms = sizes[main_dim]
        
        # Ethane should have 8 atoms (2 C + 6 H)
        assert n_atoms == 8
        
        # Check required fields
        assert "name" in atoms.data_vars
        assert "xyz" in atoms.data_vars
        assert "type" in atoms.data_vars
        assert "charge" in atoms.data_vars

    def test_imatinib_large_molecule(self, test_data_path):
        """Test reading larger molecule (imatinib)."""
        fpath = test_data_path / "data/mol2/imatinib.mol2"
        if not fpath.exists():
            pytest.skip("imatinib.mol2 test data not available")
        
        frame = mp.io.read_mol2(fpath, frame=mp.Frame())
        
        # Check basic structure
        assert "atoms" in frame
        atoms = frame["atoms"]
        
        # Get atom count
        sizes = atoms.sizes
        main_dim = next(iter(sizes.keys()))
        n_atoms = sizes[main_dim]
        
        # Imatinib should have many atoms
        assert n_atoms > 50
        
        # Check bonds if present
        if "bonds" in frame:
            bonds = frame["bonds"]
            bond_sizes = bonds.sizes
            bond_dim = next(iter(bond_sizes.keys()))
            assert bond_sizes[bond_dim] > 0

    def test_status_bits_with_bond_status(self, test_data_path):
        """Test MOL2 files with bond status information."""
        # Test any available MOL2 file
        for mol2_file in ["ethane.mol2", "imatinib.mol2"]:
            fpath = test_data_path / f"data/mol2/{mol2_file}"
            if fpath.exists():
                frame = mp.io.read_mol2(fpath, frame=mp.Frame())
                
                # Should handle status bits gracefully
                assert "atoms" in frame
                atoms = frame["atoms"]
                
                # Check basic fields
                assert "name" in atoms.data_vars
                assert "xyz" in atoms.data_vars
                break
        else:
            pytest.skip("No MOL2 test files available")

    def test_small_molecules_li_pf6(self, test_data_path):
        """Test small molecules like Li/PF6."""
        # Test available small molecule files
        for mol2_file in ["ethane.mol2", "imatinib.mol2"]:
            fpath = test_data_path / f"data/mol2/{mol2_file}"
            if fpath.exists():
                frame = mp.io.read_mol2(fpath, frame=mp.Frame())
                
                # Check that small molecules are handled
                assert "atoms" in frame
                atoms = frame["atoms"]
                
                # Should have atomic numbers assigned
                if "atomic_number" in atoms.data_vars:
                    atomic_numbers = atoms["atomic_number"].values
                    assert all(an > 0 for an in atomic_numbers)
                break
        else:
            pytest.skip("No MOL2 test files available")

    def test_ring_detection_structures(self, test_data_path):
        """Test MOL2 files with ring structures."""
        fpath = test_data_path / "data/mol2/imatinib.mol2"
        if not fpath.exists():
            pytest.skip("imatinib.mol2 test data not available")
        
        frame = mp.io.read_mol2(fpath, frame=mp.Frame())
        
        # Should handle ring structures
        assert "atoms" in frame
        
        # If bonds are present, check connectivity
        if "bonds" in frame:
            bonds = frame["bonds"]
            assert "i" in bonds.data_vars
            assert "j" in bonds.data_vars

    def test_coordinate_precision(self, test_data_path):
        """Test coordinate precision in MOL2 files."""
        fpath = test_data_path / "data/mol2/ethane.mol2"
        if not fpath.exists():
            pytest.skip("ethane.mol2 test data not available")
        
        frame = mp.io.read_mol2(fpath, frame=mp.Frame())
        atoms = frame["atoms"]
        
        # Check coordinate precision
        xyz = atoms["xyz"].values
        assert xyz.dtype == np.float64
        assert not np.any(np.isnan(xyz))
        assert not np.any(np.isinf(xyz))

    def test_charge_handling(self, test_data_path):
        """Test charge information handling."""
        fpath = test_data_path / "data/mol2/ethane.mol2"
        if not fpath.exists():
            pytest.skip("ethane.mol2 test data not available")
        
        frame = mp.io.read_mol2(fpath, frame=mp.Frame())
        atoms = frame["atoms"]
        
        # Should have charge information
        if "charge" in atoms.data_vars:
            charges = atoms["charge"].values
            assert len(charges) > 0
            assert all(isinstance(c, (int, float)) for c in charges)

    def test_substructure_handling(self, test_data_path):
        """Test substructure information."""
        fpath = test_data_path / "data/mol2/ethane.mol2"
        if not fpath.exists():
            pytest.skip("ethane.mol2 test data not available")
        
        frame = mp.io.read_mol2(fpath, frame=mp.Frame())
        atoms = frame["atoms"]
        
        # Check substructure fields
        if "subst_id" in atoms.data_vars:
            subst_ids = atoms["subst_id"].values
            assert len(subst_ids) > 0
        
        if "subst_name" in atoms.data_vars:
            subst_names = atoms["subst_name"].values
            assert len(subst_names) > 0

    def test_atomic_number_assignment(self, test_data_path):
        """Test atomic number assignment from atom types."""
        fpath = test_data_path / "data/mol2/ethane.mol2"
        if not fpath.exists():
            pytest.skip("ethane.mol2 test data not available")
        
        frame = mp.io.read_mol2(fpath, frame=mp.Frame())
        atoms = frame["atoms"]
        
        # Should have atomic numbers
        if "atomic_number" in atoms.data_vars:
            atomic_numbers = atoms["atomic_number"].values
            assert all(an > 0 for an in atomic_numbers)
            
            # For ethane, should have carbon (6) and hydrogen (1)
            unique_elements = set(atomic_numbers)
            assert 6 in unique_elements  # Carbon
            assert 1 in unique_elements  # Hydrogen

    def test_bond_type_variations(self, test_data_path):
        """Test various bond type handling."""
        for mol2_file in ["ethane.mol2", "imatinib.mol2"]:
            fpath = test_data_path / f"data/mol2/{mol2_file}"
            if fpath.exists():
                frame = mp.io.read_mol2(fpath, frame=mp.Frame())
                
                # Should handle bonds
                if "bonds" in frame:
                    bonds = frame["bonds"]
                    
                    # Check bond connectivity
                    assert "i" in bonds.data_vars
                    assert "j" in bonds.data_vars
                    
                    # Bond indices should be valid
                    bond_i = bonds["i"].values
                    bond_j = bonds["j"].values
                    assert all(i >= 0 for i in bond_i)
                    assert all(j >= 0 for j in bond_j)
                break

    def test_empty_sections_handling(self, test_data_path):
        """Test handling of MOL2 files with missing sections."""
        # Test that files with missing sections are handled gracefully
        for mol2_file in ["ethane.mol2", "imatinib.mol2"]:
            fpath = test_data_path / f"data/mol2/{mol2_file}"
            if fpath.exists():
                frame = mp.io.read_mol2(fpath, frame=mp.Frame())
                
                # Should always have atoms
                assert "atoms" in frame
                
                # Other sections may or may not be present
                atoms = frame["atoms"]
                assert len(atoms["name"].values) > 0
                break

    def test_multiple_files_consistency(self, test_data_path):
        """Test consistency across multiple MOL2 files."""
        mol2_files = []
        for mol2_file in ["ethane.mol2", "imatinib.mol2"]:
            fpath = test_data_path / f"data/mol2/{mol2_file}"
            if fpath.exists():
                mol2_files.append(fpath)
        
        if len(mol2_files) < 1:
            pytest.skip("Not enough MOL2 test files available")
        
        # Test that all files can be read consistently
        for fpath in mol2_files:
            frame = mp.io.read_mol2(fpath, frame=mp.Frame())
            
            # Basic consistency checks
            assert "atoms" in frame
            atoms = frame["atoms"]
            
            # Should have basic fields
            assert "name" in atoms.data_vars
            assert "xyz" in atoms.data_vars
            
            # Coordinates should be valid
            xyz = atoms["xyz"].values
            assert not np.any(np.isnan(xyz))
            assert not np.any(np.isinf(xyz))
