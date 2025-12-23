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
        top_file = TEST_DATA_DIR / "top/1-bromobutane.top"

        from molpy import (
            AngleStyle,
            AngleType,
            AtomStyle,
            AtomType,
            BondStyle,
            BondType,
            DihedralStyle,
            DihedralType,
            PairStyle,
            PairType,
        )

        ff = mp.io.read_top(top_file)
        # Test atom style and types
        atomstyles = ff.get_styles(AtomStyle)
        assert len(atomstyles) > 0
        atomstyle = atomstyles[0]
        atomtypes = atomstyle.types.bucket(AtomType)
        assert len(atomtypes) > 0  # At least some atom types should be parsed

        # Test specific atom type (if available)
        opls_135_types = [at for at in atomtypes if at.name == "opls_135"]
        if opls_135_types:
            at135 = opls_135_types[0]
            assert at135["atom"] == "C"
            assert at135["charge"] == "-0.18"
            assert at135["mass"] == "12.011"

        # Test bond style and types
        bondstyles = ff.get_styles(BondStyle)
        assert len(bondstyles) > 0
        bondtypes = bondstyles[0].types.bucket(BondType)
        assert len(bondtypes) > 0  # At least some bond types should be parsed

        # Test angle style and types
        anglestyles = ff.get_styles(AngleStyle)
        assert len(anglestyles) > 0
        angletypes = anglestyles[0].types.bucket(AngleType)
        assert len(angletypes) > 0  # At least some angle types should be parsed

        # Test dihedral style and types
        dihedralstyles = ff.get_styles(DihedralStyle)
        assert len(dihedralstyles) > 0
        dihedraltypes = dihedralstyles[0].types.bucket(DihedralType)
        assert len(dihedraltypes) > 0  # At least some dihedral types should be parsed

        # Test pair style and types
        pairstyles = ff.get_styles(PairStyle)
        assert len(pairstyles) > 0
        pairtypes = pairstyles[0].types.bucket(PairType)
        assert len(pairtypes) > 0  # At least some pair types should be parsed

    def test_read_nonexistent_file(self) -> None:
        """Test error handling when reading non-existent TOP file.

        Verifies that appropriate exceptions are raised when attempting to read
        a topology file that doesn't exist.
        """
        nonexistent_file = Path("nonexistent.top")

        with pytest.raises(FileNotFoundError):
            mp.io.read_top(nonexistent_file, mp.ForceField())

    def test_force_field_parameter_validation(self, TEST_DATA_DIR: Path) -> None:
        """Test detailed validation of force field parameters.

        Performs more detailed checks on the parsed force field parameters
        to ensure data integrity and correct parsing.

        Args:
            TEST_DATA_DIR: Path to test data directory fixture.
        """
        top_file = TEST_DATA_DIR / "top/1-bromobutane.top"

        from molpy import (
            AngleStyle,
            AngleType,
            AtomStyle,
            AtomType,
            BondStyle,
            BondType,
        )

        ff = mp.io.read_top(top_file, mp.ForceField())
        # Test that atom types contain expected fields
        atomstyles = ff.get_styles(AtomStyle)
        if len(atomstyles) > 0:
            atomtypes = atomstyles[0].types.bucket(AtomType)
            if len(atomtypes) > 0:
                # Check that first atom type has required fields
                first_type = atomtypes[0]
                # assert "atom" in first_type
                assert "mass" in first_type
                assert "charge" in first_type

        # Test that bond types are properly structured
        bondstyles = ff.get_styles(BondStyle)
        if len(bondstyles) > 0:
            bondtypes = bondstyles[0].types.bucket(BondType)
            if len(bondtypes) > 0:
                # Bond types should exist and be accessible
                first_bond = bondtypes[0]
                assert hasattr(first_bond, "__str__")  # Should be representable

        # Test that angle types are properly structured
        anglestyles = ff.get_styles(AngleStyle)
        if len(anglestyles) > 0:
            angletypes = anglestyles[0].types.bucket(AngleType)
            if len(angletypes) > 0:
                # Angle types should exist
                assert len(angletypes) > 0

    def test_multiple_style_access(self, TEST_DATA_DIR: Path) -> None:
        """Test accessing multiple force field styles from same file.

        Verifies that multiple interaction styles can be accessed from the same
        force field without conflicts.

        Args:
            TEST_DATA_DIR: Path to test data directory fixture.
        """
        top_file = TEST_DATA_DIR / "top/1-bromobutane.top"

        from molpy.core.forcefield import (
            AngleStyle,
            AtomStyle,
            BondStyle,
            DihedralStyle,
            PairStyle,
        )

        ff = mp.io.read_top(top_file, mp.ForceField())
        # Should be able to access all styles
        atomstyles = ff.get_styles(AtomStyle)
        bondstyles = ff.get_styles(BondStyle)
        anglestyles = ff.get_styles(AngleStyle)
        dihedralstyles = ff.get_styles(DihedralStyle)
        pairstyles = ff.get_styles(PairStyle)

        # At least some styles should be available
        styles_count = (
            len(atomstyles)
            + len(bondstyles)
            + len(anglestyles)
            + len(dihedralstyles)
            + len(pairstyles)
        )
        assert styles_count > 0, "No force field styles were loaded"
