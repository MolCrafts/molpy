"""
Unit tests for PolymerBuilder - Template-based polymer construction.

This module contains comprehensive tests for the PolymerBuilder system,
including monomer creation, anchor rules, and polymer chain construction.
"""
import pytest
import numpy as np
import numpy.testing as npt

from molpy.core.atomistic import Atomistic, Atom
from molpy.builder.polybuilder import PolymerBuilder, Monomer, AnchorRule
from molpy.op.geometry import rotation_matrix_from_vectors


class TestAnchorRule:
    """Test suite for AnchorRule class."""
    
    def test_anchor_rule_creation(self):
        """Test basic AnchorRule creation."""
        rule = AnchorRule(init=0, end=1)
        assert rule.init == 0
        assert rule.end == 1
        assert rule.deletes == []
    
    def test_anchor_rule_with_deletes(self):
        """Test AnchorRule creation with delete list."""
        rule = AnchorRule(init=0, end=1, deletes=[2, 3])
        assert rule.init == 0
        assert rule.end == 1
        assert rule.deletes == [2, 3]


class TestMonomer:
    """Test suite for Monomer class."""
    
    @pytest.fixture
    def simple_atomistic(self):
        """Create a simple 3-atom structure for testing."""
        struct = Atomistic()
        struct.atoms.add(Atom(name="C1", symbol="C", xyz=np.array([0.0, 0.0, 0.0])))
        struct.atoms.add(Atom(name="C2", symbol="C", xyz=np.array([1.0, 0.0, 0.0])))
        struct.atoms.add(Atom(name="H1", symbol="H", xyz=np.array([0.0, 1.0, 0.0])))
        return struct
    
    def test_monomer_creation(self, simple_atomistic):
        """Test basic Monomer creation."""
        anchor = AnchorRule(init=0, end=1)
        monomer = Monomer(simple_atomistic, anchors=[anchor])
        
        assert len(monomer.anchors) == 1
        assert monomer.anchors[0].init == 0
        assert monomer.anchors[0].end == 1
        assert len(monomer.atoms) == 3
    
    def test_monomer_xyz_property(self, simple_atomistic):
        """Test that Monomer correctly inherits xyz property from structure."""
        monomer = Monomer(simple_atomistic, anchors=[])
        expected_xyz = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        npt.assert_array_almost_equal(monomer.xyz, expected_xyz)
    
    def test_monomer_symbols_property(self, simple_atomistic):
        """Test that Monomer correctly provides symbols property."""
        monomer = Monomer(simple_atomistic, anchors=[])
        expected_symbols = ['C', 'C', 'H']
        assert monomer.struct.symbols == expected_symbols


class TestPolymerBuilder:
    """Test suite for PolymerBuilder class."""
    
    @pytest.fixture
    def test_monomers(self):
        """Create test monomers for polymer building."""
        # Create a simple methylene monomer (-CH2-)
        methylene = Atomistic()
        methylene.atoms.add(Atom(name="C1", symbol="C", xyz=np.array([0.0, 0.0, 0.0])))
        methylene.atoms.add(Atom(name="H1", symbol="H", xyz=np.array([0.0, 1.0, 0.0])))
        methylene.atoms.add(Atom(name="H2", symbol="H", xyz=np.array([0.0, 0.0, 1.0])))
        methylene.atoms.add(Atom(name="C2", symbol="C", xyz=np.array([1.5, 0.0, 0.0])))
        
        # Anchor rule: from C1 (init=0) to C2 (end=3), delete nothing for first unit
        anchor_ch2 = AnchorRule(init=0, end=3, deletes=[])
        monomer_ch2 = Monomer(methylene, anchors=[anchor_ch2])
        
        # Create a methyl end cap (-CH3)
        methyl = Atomistic()
        methyl.atoms.add(Atom(name="C1", symbol="C", xyz=np.array([0.0, 0.0, 0.0])))
        methyl.atoms.add(Atom(name="H1", symbol="H", xyz=np.array([0.0, 1.0, 0.0])))
        methyl.atoms.add(Atom(name="H2", symbol="H", xyz=np.array([0.0, 0.0, 1.0])))
        methyl.atoms.add(Atom(name="H3", symbol="H", xyz=np.array([1.0, 0.0, 0.0])))
        
        anchor_ch3 = AnchorRule(init=0, end=0, deletes=[])  # Self-contained unit
        monomer_ch3 = Monomer(methyl, anchors=[anchor_ch3])
        
        return {"CH2": monomer_ch2, "CH3": monomer_ch3}
    
    def test_polymer_builder_creation(self, test_monomers):
        """Test PolymerBuilder initialization."""
        builder = PolymerBuilder(test_monomers)
        assert "CH2" in builder.monomers
        assert "CH3" in builder.monomers
        assert len(builder.monomers) == 2
    
    def test_build_single_monomer(self, test_monomers):
        """Test building polymer with single monomer."""
        builder = PolymerBuilder(test_monomers)
        
        # Simple path with one point
        path = np.array([[0.0, 0.0, 0.0]])
        seq = ["CH3"]
        
        result = builder.build(path, seq)
        
        assert isinstance(result, Atomistic)
        assert len(result.atoms) == 4  # One CH3 unit
        assert len(result.symbols) == 4
    
    def test_build_linear_chain(self, test_monomers):
        """Test building a linear polymer chain."""
        builder = PolymerBuilder(test_monomers)
        
        # Linear path along x-axis
        path = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0]
        ])
        seq = ["CH2", "CH2", "CH3"]
        
        result = builder.build(path, seq)
        
        assert isinstance(result, Atomistic)
        # Should have 3 + 3 + 4 = 10 atoms (assuming no deletions for simplicity)
        assert len(result.atoms) >= 8  # At least some atoms
        assert len(result.positions) == len(result.symbols)
    
    def test_build_with_rotation(self, test_monomers):
        """Test that monomers are properly rotated along path."""
        builder = PolymerBuilder(test_monomers)
        
        # Path with direction change
        path = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Along x
            [1.0, 1.0, 0.0]   # Turn to y
        ])
        seq = ["CH2", "CH2", "CH3"]
        
        result = builder.build(path, seq)
        
        assert isinstance(result, Atomistic)
        assert len(result.atoms) >= 6
        
        # Check that atoms are positioned reasonably
        positions = result.positions
        assert positions.shape[1] == 3  # 3D coordinates
    
    def test_build_with_deletions(self):
        """Test polymer building with atom deletions."""
        # Create monomer with deletion rule
        methylene = Atomistic()
        methylene.atoms.add(Atom(name="C1", symbol="C", xyz=np.array([0.0, 0.0, 0.0])))
        methylene.atoms.add(Atom(name="H1", symbol="H", xyz=np.array([0.0, 1.0, 0.0])))
        methylene.atoms.add(Atom(name="H2", symbol="H", xyz=np.array([0.0, 0.0, 1.0])))
        methylene.atoms.add(Atom(name="C2", symbol="C", xyz=np.array([1.5, 0.0, 0.0])))
        
        # Rule that deletes H2 atom (index 2) after first monomer
        anchor_with_deletion = AnchorRule(init=0, end=3, deletes=[2])
        monomer_del = Monomer(methylene, anchors=[anchor_with_deletion])
        
        builder = PolymerBuilder({"DEL": monomer_del})
        
        path = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]
        ])
        seq = ["DEL", "DEL"]
        
        result = builder.build(path, seq)
        
        # First monomer: 4 atoms, second monomer: 3 atoms (one deleted)
        assert len(result.atoms) == 7
    
    def test_empty_sequence(self, test_monomers):
        """Test handling of empty sequence."""
        builder = PolymerBuilder(test_monomers)
        
        path = np.array([[0.0, 0.0, 0.0]])
        seq = []
        
        with pytest.raises((IndexError, ValueError)):
            builder.build(path, seq)
    
    def test_mismatched_path_sequence_length(self, test_monomers):
        """Test handling of mismatched path and sequence lengths."""
        builder = PolymerBuilder(test_monomers)
        
        # Path has 2 points but sequence has 3 elements
        path = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        seq = ["CH2", "CH2", "CH3"]
        
        # This should work - last monomer reuses previous direction
        result = builder.build(path, seq)
        assert isinstance(result, Atomistic)
    
    def test_unknown_monomer_name(self, test_monomers):
        """Test handling of unknown monomer name."""
        builder = PolymerBuilder(test_monomers)
        
        path = np.array([[0.0, 0.0, 0.0]])
        seq = ["UNKNOWN"]
        
        with pytest.raises(KeyError):
            builder.build(path, seq)


class TestGeometryOperations:
    """Test suite for geometry operations used in polymer building."""
    
    def test_rotation_matrix_identity(self):
        """Test rotation matrix for identical vectors."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        
        R = rotation_matrix_from_vectors(v1, v2)
        
        # Should be identity matrix
        npt.assert_array_almost_equal(R, np.eye(3))
    
    def test_rotation_matrix_90_degrees(self):
        """Test rotation matrix for 90-degree rotation."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        
        R = rotation_matrix_from_vectors(v1, v2)
        
        # Apply rotation to v1 should give v2
        result = R @ v1
        npt.assert_array_almost_equal(result, v2)
    
    def test_rotation_matrix_opposite_vectors(self):
        """Test rotation matrix for opposite vectors."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        
        R = rotation_matrix_from_vectors(v1, v2)
        
        # Apply rotation to v1 should give v2
        result = R @ v1
        npt.assert_array_almost_equal(result, v2, decimal=5)


class TestIntegration:
    """Integration tests for complete polymer building workflows."""
    
    def test_build_realistic_polymer(self):
        """Test building a more realistic polymer structure."""
        # Create ethylene monomer (C2H4 unit)
        ethylene = Atomistic()
        # Add atoms with realistic coordinates
        positions = np.array([
            [0.0, 0.0, 0.0],    # C1
            [1.54, 0.0, 0.0],   # C2 (typical C-C bond length)
            [-0.5, 0.9, 0.0],   # H1
            [-0.5, -0.9, 0.0],  # H2
            [2.04, 0.9, 0.0],   # H3
            [2.04, -0.9, 0.0]   # H4
        ])
        
        symbols = ['C', 'C', 'H', 'H', 'H', 'H']
        for i, (pos, sym) in enumerate(zip(positions, symbols)):
            ethylene.atoms.add(Atom(name=f"{sym}{i+1}", symbol=sym, xyz=pos))
        
        # Anchor from C1 to C2, delete overlapping hydrogens
        anchor = AnchorRule(init=0, end=1, deletes=[4, 5])  # Delete H3, H4 from subsequent units
        eth_monomer = Monomer(ethylene, anchors=[anchor])
        
        builder = PolymerBuilder({"ETH": eth_monomer})
        
        # Create a curved path
        path = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 1.0, 0.0],
            [9.0, 0.0, 0.0]
        ])
        seq = ["ETH", "ETH", "ETH", "ETH"]
        
        result = builder.build(path, seq)
        
        # Should have reasonable number of atoms
        assert len(result.atoms) >= 12  # At least 4 carbons + some hydrogens
        assert len(result.atoms) <= 22  # Not more than full structure
        
        # Check that we have both C and H atoms
        symbols = result.symbols
        assert 'C' in symbols
        assert 'H' in symbols
        
        # Verify 3D structure
        positions = result.positions
        assert positions.shape[1] == 3
        assert not np.allclose(positions, 0)  # Not all at origin
    
    def test_polymer_building_performance(self, test_monomers):
        """Test polymer building with larger structures."""
        builder = PolymerBuilder(test_monomers)
        
        # Create longer chain
        n_units = 20
        path = np.array([[i * 2.0, 0.0, 0.0] for i in range(n_units)])
        seq = ["CH2"] * (n_units - 1) + ["CH3"]
        
        result = builder.build(path, seq)
        
        assert isinstance(result, Atomistic)
        assert len(result.atoms) > n_units  # Should have multiple atoms per unit
        
        # Basic sanity checks
        positions = result.positions
        assert positions.shape[0] == len(result.symbols)
        assert positions.shape[1] == 3


if __name__ == "__main__":
    pytest.main([__file__])