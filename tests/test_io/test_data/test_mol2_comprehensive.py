"""Comprehensive MOL2 format tests using chemfiles-testcases data."""

import pytest
import molpy as mp
from pathlib import Path


class TestMol2Comprehensive:
    """Comprehensive tests for MOL2 format using real chemfiles-testcases data."""
    
    def test_ethane_basic_structure(self, test_data_path):
        """Test basic ethane molecule structure and properties."""
        frame = mp.Frame()
        mol2_path = test_data_path / "data/mol2/ethane.mol2"
        result = mp.io.read_mol2(mol2_path, frame)
        
        atoms = result["atoms"]
        bonds = result["bonds"]
        
        # Check atom count
        atom_dim = next(iter(atoms.sizes.keys()))
        assert atoms.sizes[atom_dim] == 8
        
        # Check bond count
        bond_dim = next(iter(bonds.sizes.keys()))
        assert bonds.sizes[bond_dim] == 7
        
        # Verify specific atom properties
        name_dim = next(d for d in atoms.dims if 'name' in str(d))
        xyz_dim = next(d for d in atoms.dims if 'xyz' in str(d) and not str(d).endswith('_1'))
        type_dim = next(d for d in atoms.dims if 'type' in str(d))
        charge_dim = next(d for d in atoms.dims if 'charge' in str(d))
        
        # Test first atom (carbon)
        assert atoms["name"].isel({name_dim: 0}).item() == "C"
        assert atoms["type"].isel({type_dim: 0}).item() == "c3"
        assert atoms["charge"].isel({charge_dim: 0}).item() == pytest.approx(-0.094100)
        
        # Test hydrogen atoms
        assert atoms["name"].isel({name_dim: 2}).item() == "H"
        assert atoms["type"].isel({type_dim: 2}).item() == "hc"
        assert atoms["charge"].isel({charge_dim: 2}).item() == pytest.approx(0.031700)
        
        # Test coordinates
        coords_0 = tuple(atoms["xyz"].isel({xyz_dim: 0}).values)
        assert coords_0 == pytest.approx((3.1080, 0.6530, -8.5260))
        
        # Test bond properties
        i_dim = next(d for d in bonds["i"].dims)
        j_dim = next(d for d in bonds["j"].dims)
        type_dim = next(d for d in bonds["type"].dims)
        
        # Test first bond
        assert bonds["i"].isel({i_dim: 0}).item() == 1
        assert bonds["j"].isel({j_dim: 0}).item() == 2
        assert bonds["type"].isel({type_dim: 0}).item() == "1"
    
    def test_imatinib_large_molecule(self, test_data_path):
        """Test large molecule (imatinib) with multiple atom types."""
        frame = mp.Frame()
        mol2_path = test_data_path / "data/mol2/imatinib.mol2"
        result = mp.io.read_mol2(mol2_path, frame)
        
        atoms = result["atoms"]
        bonds = result["bonds"]
        
        # Check atom and bond counts
        atom_dim = next(iter(atoms.sizes.keys()))
        bond_dim = next(iter(bonds.sizes.keys()))
        assert atoms.sizes[atom_dim] == 68
        assert bonds.sizes[bond_dim] == 72
        
        # Check diverse atom types are present
        type_dim = next(d for d in atoms.dims if 'type' in str(d))
        atom_types = set()
        for i in range(atoms.sizes[atom_dim]):
            atom_types.add(atoms["type"].isel({type_dim: i}).item())
        
        # Should have various atom types like o, n3, nb, c3, ca, etc.
        assert len(atom_types) > 5
        assert "o" in atom_types
        assert "n3" in atom_types
        assert "c3" in atom_types
        assert "ca" in atom_types
    
    def test_status_bits_with_bond_status(self, test_data_path):
        """Test MOL2 file with bond status bits."""
        frame = mp.Frame()
        mol2_path = test_data_path / "data/mol2/status-bits.mol2"
        result = mp.io.read_mol2(mol2_path, frame)
        
        atoms = result["atoms"]
        bonds = result["bonds"]
        
        # Check structure
        atom_dim = next(iter(atoms.sizes.keys()))
        bond_dim = next(iter(bonds.sizes.keys()))
        assert atoms.sizes[atom_dim] == 18
        assert bonds.sizes[bond_dim] == 18  # 18 bonds listed in this test case
        
        # Check aromatic carbons
        type_dim = next(d for d in atoms.dims if 'type' in str(d))
        name_dim = next(d for d in atoms.dims if 'name' in str(d))
        
        # First atom should be aromatic carbon
        assert atoms["name"].isel({name_dim: 0}).item() == "C1"
        assert atoms["type"].isel({type_dim: 0}).item() == "C.ar"
        
        # Check bond types include aromatic
        bond_type_dim = next(d for d in bonds["type"].dims)
        bond_types = set()
        for i in range(bonds.sizes[bond_dim]):
            bond_types.add(bonds["type"].isel({bond_type_dim: i}).item())
        
        assert "ar" in bond_types  # aromatic bond
        assert "1" in bond_types   # single bond
    
    def test_small_molecules_li_pf6(self, test_data_path):
        """Test very small molecules (ions)."""
        # Test lithium ion
        frame = mp.Frame()
        li_path = test_data_path / "data/mol2/li.mol2"
        result = mp.io.read_mol2(li_path, frame)
        
        atoms = result["atoms"]
        atom_dim = next(iter(atoms.sizes.keys()))
        assert atoms.sizes[atom_dim] == 1
        
        # Check lithium properties
        name_dim = next(d for d in atoms.dims if 'name' in str(d))
        assert atoms["name"].isel({name_dim: 0}).item() == "Li"
        
        # Check atomic number assignment
        number_dim = next(d for d in atoms.dims if 'number' in str(d))
        assert atoms["number"].isel({number_dim: 0}).item() == 3  # Li atomic number
    
    def test_ring_detection_structures(self, test_data_path):
        """Test ring and fused ring structures."""
        # Test simple ring
        frame = mp.Frame()
        ring_path = test_data_path / "data/mol2/ring.mol2"
        result = mp.io.read_mol2(ring_path, frame)
        
        atoms = result["atoms"]
        bonds = result["bonds"]
        
        # Should have atoms and bonds for ring structure
        atom_dim = next(iter(atoms.sizes.keys()))
        bond_dim = next(iter(bonds.sizes.keys()))
        assert atoms.sizes[atom_dim] > 0
        assert bonds.sizes[bond_dim] > 0
        
        # Test fused rings
        frame2 = mp.Frame()
        fused_path = test_data_path / "data/mol2/fused.mol2"
        result2 = mp.io.read_mol2(fused_path, frame2)
        
        atoms2 = result2["atoms"]
        bonds2 = result2["bonds"]
        
        atom_dim2 = next(iter(atoms2.sizes.keys()))
        bond_dim2 = next(iter(bonds2.sizes.keys()))
        assert atoms2.sizes[atom_dim2] > 0
        assert bonds2.sizes[bond_dim2] > 0
    
    def test_coordinate_precision(self, test_data_path):
        """Test coordinate precision handling."""
        frame = mp.Frame()
        mol2_path = test_data_path / "data/mol2/ethane.mol2"
        result = mp.io.read_mol2(mol2_path, frame)
        
        atoms = result["atoms"]
        xyz_dim = next(d for d in atoms.dims if 'xyz' in str(d) and not str(d).endswith('_1'))
        
        # Check that coordinates maintain precision
        coords = atoms["xyz"].isel({xyz_dim: 0}).values
        assert abs(coords[0] - 3.1080) < 1e-4
        assert abs(coords[1] - 0.6530) < 1e-4
        assert abs(coords[2] + 8.5260) < 1e-4
    
    def test_charge_handling(self, test_data_path):
        """Test charge field handling including zero charges."""
        frame = mp.Frame()
        
        # Test file with charges
        ethane_path = test_data_path / "data/mol2/ethane.mol2"
        result = mp.io.read_mol2(ethane_path, frame)
        
        atoms = result["atoms"]
        charge_dim = next(d for d in atoms.dims if 'charge' in str(d))
        
        # Check that charges are properly parsed
        charge_0 = atoms["charge"].isel({charge_dim: 0}).item()
        assert charge_0 == pytest.approx(-0.094100)
        
        # Test file with zero charges
        frame2 = mp.Frame()
        status_path = test_data_path / "data/mol2/status-bits.mol2"
        result2 = mp.io.read_mol2(status_path, frame2)
        
        atoms2 = result2["atoms"]
        charge_dim2 = next(d for d in atoms2.dims if 'charge' in str(d))
        
        # All charges should be 0.0 in status-bits.mol2
        for i in range(atoms2.sizes[next(iter(atoms2.sizes.keys()))]):
            charge = atoms2["charge"].isel({charge_dim2: i}).item()
            assert charge == pytest.approx(0.0)
    
    def test_substructure_handling(self, test_data_path):
        """Test substructure ID and name handling."""
        frame = mp.Frame()
        mol2_path = test_data_path / "data/mol2/ethane.mol2"
        result = mp.io.read_mol2(mol2_path, frame)
        
        atoms = result["atoms"]
        subst_id_dim = next(d for d in atoms.dims if 'subst_id' in str(d))
        subst_name_dim = next(d for d in atoms.dims if 'subst_name' in str(d))
        
        # All atoms in ethane should belong to substructure 1 named "ETH"
        atom_dim = next(iter(atoms.sizes.keys()))
        for i in range(atoms.sizes[atom_dim]):
            subst_id = atoms["subst_id"].isel({subst_id_dim: i}).item()
            subst_name = atoms["subst_name"].isel({subst_name_dim: i}).item()
            assert subst_id == 1
            assert subst_name == "ETH"
    
    def test_atomic_number_assignment(self, test_data_path):
        """Test atomic number assignment from atom names and types."""
        frame = mp.Frame()
        mol2_path = test_data_path / "data/mol2/ethane.mol2"
        result = mp.io.read_mol2(mol2_path, frame)
        
        atoms = result["atoms"]
        name_dim = next(d for d in atoms.dims if 'name' in str(d))
        number_dim = next(d for d in atoms.dims if 'number' in str(d))
        
        # Check carbon atoms
        carbon_indices = []
        hydrogen_indices = []
        atom_dim = next(iter(atoms.sizes.keys()))
        
        for i in range(atoms.sizes[atom_dim]):
            name = atoms["name"].isel({name_dim: i}).item()
            if name == "C":
                carbon_indices.append(i)
            elif name == "H":
                hydrogen_indices.append(i)
        
        # Check atomic numbers
        for i in carbon_indices:
            atomic_num = atoms["number"].isel({number_dim: i}).item()
            assert atomic_num == 6  # Carbon
        
        for i in hydrogen_indices:
            atomic_num = atoms["number"].isel({number_dim: i}).item()
            assert atomic_num == 1  # Hydrogen
    
    def test_bond_type_variations(self, test_data_path):
        """Test different bond type representations."""
        frame = mp.Frame()
        
        # Test standard single bonds
        ethane_path = test_data_path / "data/mol2/ethane.mol2"
        result = mp.io.read_mol2(ethane_path, frame)
        
        bonds = result["bonds"]
        type_dim = next(d for d in bonds["type"].dims)
        
        # All bonds in ethane should be single bonds
        bond_dim = next(iter(bonds.sizes.keys()))
        for i in range(bonds.sizes[bond_dim]):
            bond_type = bonds["type"].isel({type_dim: i}).item()
            assert bond_type == "1"
        
        # Test aromatic bonds
        frame2 = mp.Frame()
        status_path = test_data_path / "data/mol2/status-bits.mol2"
        result2 = mp.io.read_mol2(status_path, frame2)
        
        bonds2 = result2["bonds"]
        type_dim2 = next(d for d in bonds2["type"].dims)
        
        # Should have aromatic bonds
        bond_types = set()
        bond_dim2 = next(iter(bonds2.sizes.keys()))
        for i in range(bonds2.sizes[bond_dim2]):
            bond_types.add(bonds2["type"].isel({type_dim2: i}).item())
        
        assert "ar" in bond_types
    
    def test_empty_sections_handling(self, test_data_path):
        """Test handling of files with missing or empty sections."""
        # Test lithium file (single atom, no bonds)
        frame = mp.Frame()
        li_path = test_data_path / "data/mol2/li.mol2"
        result = mp.io.read_mol2(li_path, frame)
        
        atoms = result["atoms"]
        atom_dim = next(iter(atoms.sizes.keys()))
        assert atoms.sizes[atom_dim] == 1
        
        # Should handle missing bonds gracefully
        if "bonds" in result:
            bonds = result["bonds"]
            bond_dim = next(iter(bonds.sizes.keys()))
            # Should have 0 bonds for single atom
            assert bonds.sizes[bond_dim] == 0
    
    def test_multiple_files_consistency(self, test_data_path):
        """Test that multiple files can be read consistently."""
        mol2_files = [
            "ethane.mol2",
            "li.mol2", 
            "ring.mol2",
            "status-bits.mol2"
        ]
        
        results = []
        for mol2_file in mol2_files:
            frame = mp.Frame()
            mol2_path = test_data_path / f"data/mol2/{mol2_file}"
            if mol2_path.exists():
                result = mp.io.read_mol2(mol2_path, frame)
                results.append((mol2_file, result))
        
        # All files should be readable
        assert len(results) == len(mol2_files)
        
        # Each result should have atoms
        for filename, result in results:
            assert "atoms" in result
            atoms = result["atoms"]
            atom_dim = next(iter(atoms.sizes.keys()))
            assert atoms.sizes[atom_dim] > 0
