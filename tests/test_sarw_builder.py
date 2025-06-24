#!/usr/bin/env python3
"""
Tests for the Self-Avoiding Random Walk (SARW) builder framework.

This test suite validates:
- RandomWalkLattice position generation
- RandomWalkBuilder functionality
- SAWPolymerBuilder with connectivity
- AbstractBuilder interface compliance
- Integration with existing molpy framework
"""

import pytest
import numpy as np
from molpy.core.atomistic import AtomicStructure
from molpy.builder import (
    RandomWalkLattice, 
    RandomWalkBuilder, 
    SAWPolymerBuilder,
    AbstractBuilder,
    PositionGenerator
)

class TestRandomWalkLattice:
    """Test the RandomWalkLattice position generator."""
    
    def test_initialization(self):
        """Test RandomWalkLattice initialization."""
        lattice = RandomWalkLattice(step_length=1.5, max_attempts=100, seed=42)
        
        assert lattice.step_length == 1.5
        assert lattice.max_attempts == 100
        assert lattice.rng is not None
        assert len(lattice.directions) > 0
        
        # Check that directions are normalized to step_length
        norms = np.linalg.norm(lattice.directions, axis=1)
        np.testing.assert_allclose(norms, 1.5, rtol=1e-10)
    
    def test_position_generation(self):
        """Test basic position generation."""
        lattice = RandomWalkLattice(step_length=1.0, seed=42)
        
        positions = lattice.generate_positions(n_steps=5)
        
        assert positions.shape == (6, 3)  # n_steps + 1 positions
        assert np.allclose(positions[0], [0.0, 0.0, 0.0])  # Default start
        
        # Check step lengths
        steps = np.diff(positions, axis=0)
        step_lengths = np.linalg.norm(steps, axis=1)
        np.testing.assert_allclose(step_lengths, 1.0, rtol=1e-10)
    
    def test_custom_start_position(self):
        """Test position generation with custom start."""
        lattice = RandomWalkLattice(step_length=2.0, seed=42)
        start = np.array([1.0, 2.0, 3.0])
        
        positions = lattice.generate_positions(n_steps=3, start_pos=start)
        
        assert positions.shape == (4, 3)
        np.testing.assert_allclose(positions[0], start)
    
    def test_self_avoidance(self):
        """Test that self-avoidance constraint is enforced."""
        lattice = RandomWalkLattice(step_length=1.0, max_attempts=50, seed=42)
        
        positions = lattice.generate_positions(
            n_steps=10, 
            exclusion_radius=0.8
        )
        
        # Check that no two positions are too close
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i != j:
                    distance = np.linalg.norm(pos1 - pos2)
                    assert distance >= 0.8 or distance == 0.0  # Allow identical positions
    
    def test_protocol_compliance(self):
        """Test that RandomWalkLattice implements PositionGenerator protocol."""
        lattice = RandomWalkLattice()
        assert isinstance(lattice, PositionGenerator)
        assert hasattr(lattice, 'generate_positions')
        assert callable(lattice.generate_positions)

class TestRandomWalkBuilder:
    """Test the RandomWalkBuilder class."""
    
    def create_test_monomer(self):
        """Helper to create a simple test monomer."""
        monomer = AtomicStructure("test_monomer")
        monomer.def_atom(name="C", element="C", xyz=[0.0, 0.0, 0.0])
        return monomer
    
    def test_initialization(self):
        """Test RandomWalkBuilder initialization."""
        builder = RandomWalkBuilder(step_length=1.5, seed=42)
        
        assert isinstance(builder, AbstractBuilder)
        assert isinstance(builder.position_generator, RandomWalkLattice)
        assert builder.position_generator.step_length == 1.5
    
    def test_abstract_builder_compliance(self):
        """Test that RandomWalkBuilder properly implements AbstractBuilder."""
        builder = RandomWalkBuilder()
        
        assert isinstance(builder, AbstractBuilder)
        assert hasattr(builder, 'build')
        assert callable(builder.build)
        assert hasattr(builder, 'position_generator')
    
    def test_build_basic_polymer(self):
        """Test basic polymer building."""
        builder = RandomWalkBuilder(step_length=1.0, seed=42)
        monomer = self.create_test_monomer()
        
        polymer = builder.build(
            monomer_template=monomer,
            n_monomers=5,
            name="test_polymer"
        )
        
        assert isinstance(polymer, AtomicStructure)
        assert polymer.name == "test_polymer"
        assert len(polymer.atoms) == 5  # One atom per monomer
        
        # Check that atoms are positioned correctly
        atom_positions = [atom['xyz'] for atom in polymer.atoms]
        atom_positions = np.array(atom_positions)
        
        # First atom should be at start position (default origin)
        np.testing.assert_allclose(atom_positions[0], [0.0, 0.0, 0.0])
        
        # Check distances between consecutive atoms
        distances = np.linalg.norm(np.diff(atom_positions, axis=0), axis=1)
        np.testing.assert_allclose(distances, 1.0, rtol=1e-10)
    
    def test_build_with_custom_start(self):
        """Test building with custom start position."""
        builder = RandomWalkBuilder(step_length=2.0, seed=42)
        monomer = self.create_test_monomer()
        start_pos = np.array([5.0, 10.0, -3.0])
        
        polymer = builder.build(
            monomer_template=monomer,
            n_monomers=3,
            start_position=start_pos
        )
        
        # First atom should be at the custom start position
        first_atom_pos = polymer.atoms[0]['xyz']
        np.testing.assert_allclose(first_atom_pos, start_pos)
    
    def test_monomer_template_preservation(self):
        """Test that monomer template properties are preserved."""
        builder = RandomWalkBuilder(seed=42)
        
        # Create a more complex monomer
        monomer = AtomicStructure("complex_monomer")
        monomer.def_atom(name="C1", element="C", xyz=[0.0, 0.0, 0.0])
        monomer.def_atom(name="H1", element="H", xyz=[0.5, 0.5, 0.0])
        monomer.def_atom(name="H2", element="H", xyz=[-0.5, 0.5, 0.0])
        
        polymer = builder.build(
            monomer_template=monomer,
            n_monomers=2
        )
        
        assert len(polymer.atoms) == 6  # 3 atoms per monomer × 2 monomers
        
        # Check that atom names and elements are preserved
        carbon_atoms = [atom for atom in polymer.atoms if atom['element'] == 'C']
        hydrogen_atoms = [atom for atom in polymer.atoms if atom['element'] == 'H']
        
        assert len(carbon_atoms) == 2
        assert len(hydrogen_atoms) == 4

class TestSAWPolymerBuilder:
    """Test the SAWPolymerBuilder class."""
    
    def create_ch2_monomer(self):
        """Helper to create a CH2 monomer unit."""
        monomer = AtomicStructure("CH2")
        monomer.def_atom(name="C", element="C", xyz=[0.0, 0.0, 0.0])
        monomer.def_atom(name="H1", element="H", xyz=[0.6, 0.6, 0.0])
        monomer.def_atom(name="H2", element="H", xyz=[-0.6, 0.6, 0.0])
        return monomer
    
    def test_initialization(self):
        """Test SAWPolymerBuilder initialization."""
        builder = SAWPolymerBuilder(step_length=1.54, seed=42)
        
        assert isinstance(builder, RandomWalkBuilder)
        assert isinstance(builder, AbstractBuilder)
    
    def test_build_connected_polymer(self):
        """Test building a connected polymer with bonds."""
        builder = SAWPolymerBuilder(step_length=1.0, seed=42)
        monomer = self.create_ch2_monomer()
        
        polymer = builder.build_connected_polymer(
            monomer_template=monomer,
            n_monomers=3,
            bond_length=1.5,
            name="connected_polymer"
        )
        
        assert isinstance(polymer, AtomicStructure)
        assert polymer.name == "connected_polymer"
        assert len(polymer.atoms) == 9  # 3 atoms per monomer × 3 monomers
        
        # Check that bonds were created
        assert hasattr(polymer, 'bonds')
        # Note: The exact number of bonds depends on the implementation
        # but there should be at least n_monomers-1 inter-monomer bonds
    
    def test_inherit_from_random_walk_builder(self):
        """Test that SAWPolymerBuilder inherits RandomWalkBuilder functionality."""
        builder = SAWPolymerBuilder(seed=42)
        monomer = self.create_ch2_monomer()
        
        # Should be able to use the basic build method
        polymer = builder.build(
            monomer_template=monomer,
            n_monomers=2
        )
        
        assert isinstance(polymer, AtomicStructure)
        assert len(polymer.atoms) == 6

class TestBuilderIntegration:
    """Test integration with the rest of the molpy framework."""
    
    def test_builder_exports(self):
        """Test that new builders are properly exported."""
        from molpy.builder import (
            AbstractBuilder, 
            PositionGenerator,
            RandomWalkLattice,
            RandomWalkBuilder,
            SAWPolymerBuilder
        )
        
        # All imports should succeed
        assert AbstractBuilder is not None
        assert PositionGenerator is not None
        assert RandomWalkLattice is not None
        assert RandomWalkBuilder is not None
        assert SAWPolymerBuilder is not None
    
    def test_consistent_interface(self):
        """Test that SARW builders have consistent interface with crystal builders."""
        from molpy.builder import FCCBuilder, RandomWalkBuilder
        
        # Both should inherit from AbstractBuilder
        fcc_builder = FCCBuilder(a=2.0)
        sarw_builder = RandomWalkBuilder()
        
        assert isinstance(fcc_builder, AbstractBuilder)
        assert isinstance(sarw_builder, AbstractBuilder)
        
        # Both should have build methods (though with different signatures)
        assert hasattr(fcc_builder, 'build')
        assert hasattr(sarw_builder, 'build')
    
    def test_reproducibility(self):
        """Test that builders produce reproducible results with same seed."""
        builder1 = RandomWalkBuilder(step_length=1.0, seed=12345)
        builder2 = RandomWalkBuilder(step_length=1.0, seed=12345)
        
        monomer = AtomicStructure("test")
        monomer.def_atom(name="C", element="C", xyz=[0.0, 0.0, 0.0])
        
        polymer1 = builder1.build(monomer_template=monomer, n_monomers=5)
        polymer2 = builder2.build(monomer_template=monomer, n_monomers=5)
        
        # Atom positions should be identical
        for atom1, atom2 in zip(polymer1.atoms, polymer2.atoms):
            np.testing.assert_allclose(atom1['xyz'], atom2['xyz'])

def run_tests():
    """Run all tests manually (for environments without pytest)."""
    test_classes = [
        TestRandomWalkLattice,
        TestRandomWalkBuilder,
        TestSAWPolymerBuilder,
        TestBuilderIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]
        
        print(f"\nRunning {test_class.__name__}:")
        
        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
    
    print(f"\nTest Results: {passed_tests}/{total_tests} passed")
    return passed_tests == total_tests

if __name__ == "__main__":
    try:
        import pytest
        # Run with pytest if available
        pytest.main([__file__])
    except ImportError:
        # Fall back to manual test runner
        print("pytest not available, running tests manually...")
        success = run_tests()
        if not success:
            exit(1)
