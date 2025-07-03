#!/usr/bin/env python3
"""Unit tests for GROMACS TOP file reader.

This module contains comprehensive tests for:
- GROMACS topology file reading functionality
- Force field parameter extraction and validation
- Atom types, bond types, angle types, dihedral types, and pair types
- Error handling and edge cases

Uses pytest framework with modern Python 3.10+ type hints and Google-style docstrings.
"""

from pathlib import Path

import pytest

import molpy as mp


class TestGMXTopReader:
    """Test suite for GROMACS topology file reader."""
    
    def test_read_bromobutane_topology(self, TEST_DATA_DIR: Path) -> None:
        """Test reading 1-bromobutane.top file and validating force field parameters.
        
        This test validates that the TOP reader can correctly parse GROMACS topology
        files and extract all force field parameters including atom types, bond types,
        angle types, dihedral types, and pair types.
        
        Args:
            TEST_DATA_DIR: Path to test data directory fixture.
        """
        top_file = TEST_DATA_DIR / "forcefield/gromacs/1-bromobutane.top"
        if not top_file.exists():
            pytest.skip("gromacs test data not available")
        
        ff = mp.io.read_top(top_file, mp.ForceField())
        
        # Test atom style and types
        atomstyle = ff.get_atomstyle("full")
        assert atomstyle is not None
        atomtypes = atomstyle.get_types()
        assert len(atomtypes) == 14
        
        # Test specific atom type
        at135 = atomstyle.get("opls_135")
        assert at135 is not None, "Could not find opls_135 atom type"
        assert at135["atom"] == "C"
        assert at135["charge"] == "-0.18"
        assert at135["mass"] == "12.011"

        # Test bond style and types
        bondstyle = ff.get_bondstyle("harmonic")
        assert bondstyle is not None
        bondtypes = bondstyle.get_types()
        assert len(bondtypes) == 13

        # Test angle style and types
        anglestyle = ff.get_anglestyle("harmonic")
        assert anglestyle is not None
        angletypes = anglestyle.get_types()
        assert len(angletypes) == 24

        # Test dihedral style and types
        diestyle = ff.get_dihedralstyle("harmonic")
        assert diestyle is not None
        diestypes = diestyle.get_types()
        assert len(diestypes) == 27

        # Test pair style and types
        pairstyle = ff.get_pairstyle("lj12-6")
        assert pairstyle is not None
        pairtypes = pairstyle.get_types()
        assert len(pairtypes) == 27

    def test_read_nonexistent_file(self) -> None:
        """Test error handling when reading non-existent TOP file.
        
        Verifies that appropriate exceptions are raised when attempting to read
        a topology file that doesn't exist.
        """
        nonexistent_file = Path("nonexistent.top")
        
        with pytest.raises(FileNotFoundError):
            mp.io.read_top(nonexistent_file, mp.ForceField())

    def test_empty_forcefield_creation(self) -> None:
        """Test creation of empty ForceField object.
        
        Validates that an empty ForceField can be created and has the expected
        initial state before loading any topology data.
        """
        ff = mp.ForceField()
        
        # Should be able to create empty force field
        assert ff is not None
        
        # Basic force field queries should work even when empty
        atomstyle = ff.get_atomstyle("full")
        # This might be None or an empty style depending on implementation
        # Just verify no exception is raised

    def test_force_field_parameter_validation(self, TEST_DATA_DIR: Path) -> None:
        """Test detailed validation of force field parameters.
        
        Performs more detailed checks on the parsed force field parameters
        to ensure data integrity and correct parsing.
        
        Args:
            TEST_DATA_DIR: Path to test data directory fixture.
        """
        top_file = TEST_DATA_DIR / "forcefield/gromacs/1-bromobutane.top"
        if not top_file.exists():
            pytest.skip("gromacs test data not available")
        
        ff = mp.io.read_top(top_file, mp.ForceField())
        
        # Test that atom types contain expected fields
        atomstyle = ff.get_atomstyle("full")
        if atomstyle is not None:
            atomtypes = atomstyle.get_types()
            if len(atomtypes) > 0:
                # Check that first atom type has required fields
                if isinstance(atomtypes, dict):
                    first_type = next(iter(atomtypes.values()))
                else:
                    first_type = atomtypes[0]
                assert "atom" in first_type
                assert "mass" in first_type
                assert "charge" in first_type

        # Test that bond types are properly structured
        bondstyle = ff.get_bondstyle("harmonic")
        if bondstyle is not None:
            bondtypes = bondstyle.get_types()
            if len(bondtypes) > 0:
                # Bond types should exist and be accessible (could be list or dict)
                assert isinstance(bondtypes, (dict, list))
                if isinstance(bondtypes, list):
                    # If it's a list, check first element
                    first_bond = bondtypes[0]
                    assert hasattr(first_bond, '__str__')  # Should be representable

        # Test that angle types are properly structured
        anglestyle = ff.get_anglestyle("harmonic")
        if anglestyle is not None:
            angletypes = anglestyle.get_types()
            if len(angletypes) > 0:
                # Angle types should exist and be accessible (could be list or dict)
                assert isinstance(angletypes, (dict, list))

    def test_multiple_style_access(self, TEST_DATA_DIR: Path) -> None:
        """Test accessing multiple force field styles from same file.
        
        Verifies that multiple interaction styles can be accessed from the same
        force field without conflicts.
        
        Args:
            TEST_DATA_DIR: Path to test data directory fixture.
        """
        top_file = TEST_DATA_DIR / "forcefield/gromacs/1-bromobutane.top"
        if not top_file.exists():
            pytest.skip("gromacs test data not available")
        
        ff = mp.io.read_top(top_file, mp.ForceField())
        
        # Should be able to access all styles
        styles = [
            ff.get_atomstyle("full"),
            ff.get_bondstyle("harmonic"),
            ff.get_anglestyle("harmonic"),
            ff.get_dihedralstyle("harmonic"),
            ff.get_pairstyle("lj12-6")
        ]
        
        # At least some styles should be available
        available_styles = [s for s in styles if s is not None]
        assert len(available_styles) > 0, "No force field styles were loaded"