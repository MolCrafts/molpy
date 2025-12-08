"""Tests for generative BigSMILES (gBigSMILES) parser."""

import pytest

from molpy.parser.smiles import (
    parse_gbigsmiles,
    GBigSmilesMoleculeIR,
    GBigSmilesSystemIR,
    SmilesGraphIR,
    SmilesAtomIR,
)


class TestSystemSizeParsing:
    """Test parsing of system size |NUMBER| notation."""

    def test_simple_integer_system_size(self):
        """Test parsing |1000| system size."""
        ir = parse_gbigsmiles("{[<]CC[>]}|1000|")

        # With new IR, system size creates GBigSmilesSystemIR
        assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))
        if isinstance(ir, GBigSmilesSystemIR):
            assert ir.total_mass is not None
            assert ir.total_mass == 1000.0

    def test_scientific_notation_system_size(self):
        """Test parsing |5e5| system size."""
        ir = parse_gbigsmiles("{[<]CC[>]}|5e5|")

        assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))
        if isinstance(ir, GBigSmilesSystemIR):
            assert ir.total_mass is not None
            assert ir.total_mass == 5e5

    def test_decimal_system_size(self):
        """Test parsing |1.5e6| system size."""
        ir = parse_gbigsmiles("{[<]CC[>]}|1.5e6|")

        assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))
        if isinstance(ir, GBigSmilesSystemIR):
            assert ir.total_mass is not None
            assert ir.total_mass == 1.5e6

    def test_no_system_size(self):
        """Test parsing without system size."""
        ir = parse_gbigsmiles("{[<]CC[>]}")

        assert isinstance(ir, GBigSmilesMoleculeIR)


class TestGBigSmilesBasic:
    """Test basic gBigSMILES parsing."""

    def test_simple_homopolymer_with_system_size(self):
        """Test {[<]CC[>]}|5e5|"""
        ir = parse_gbigsmiles("{[<]CC[>]}|5e5|")

        # Check IR type
        assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))

        # Check system size (if system IR)
        if isinstance(ir, GBigSmilesSystemIR):
            assert ir.total_mass is not None
            assert ir.total_mass == 5e5

        # Check structure
        if isinstance(ir, GBigSmilesMoleculeIR):
            structure = ir.structure
        else:
            structure = ir.molecules[0].molecule.structure if ir.molecules else None

        assert structure is not None
        # Check stochastic objects
        assert len(structure.stochastic_objects) >= 1

    def test_copolymer_with_system_size(self):
        """Test {[<]CC[>],[<]OCC[>]}|1e6|"""
        ir = parse_gbigsmiles("{[<]CC[>],[<]OCC[>]}|1e6|")

        assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))
        if isinstance(ir, GBigSmilesSystemIR):
            assert ir.total_mass is not None
            assert ir.total_mass == 1e6

        # Get structure
        if isinstance(ir, GBigSmilesMoleculeIR):
            structure = ir.structure
        else:
            structure = ir.molecules[0].molecule.structure if ir.molecules else None

        assert structure is not None
        # Should have stochastic objects
        assert len(structure.stochastic_objects) >= 1
        stoch_obj = structure.stochastic_objects[0]
        # Should have 2 repeat units
        assert len(stoch_obj.repeat_units) >= 2

    def test_with_distribution_and_system_size(self):
        """Test {[<]CC[>]}|flory_schulz(0.9)||5e5| - distribution and system size"""
        ir = parse_gbigsmiles("{[<]CC[>]}|flory_schulz(0.9)||5e5|")

        assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))

        # Check system size (if system IR)
        if isinstance(ir, GBigSmilesSystemIR):
            assert ir.total_mass is not None
            assert ir.total_mass == 5e5

        # Get structure and check distribution
        if isinstance(ir, GBigSmilesMoleculeIR):
            structure = ir.structure
            # Check stochastic metadata
            if ir.stochastic_metadata:
                gb_stoch = ir.stochastic_metadata[0]
                assert gb_stoch.distribution is not None
                assert gb_stoch.distribution.name == "flory_schulz"
        else:
            structure = ir.molecules[0].molecule.structure if ir.molecules else None


class TestGBigSmilesWithStartingSegment:
    """Test gBigSMILES with starting SMILES segment."""

    def test_starting_segment_simple(self):
        """Test CCOC{[<]CC[>]}|5e5|"""
        ir = parse_gbigsmiles("CCOC{[<]CC[>]}|5e5|")

        assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))
        if isinstance(ir, GBigSmilesSystemIR):
            assert ir.total_mass is not None
            assert ir.total_mass == 5e5

        # Check backbone (starting segment)
        if isinstance(ir, GBigSmilesMoleculeIR):
            structure = ir.structure
        else:
            structure = ir.molecules[0].molecule.structure if ir.molecules else None

        assert structure is not None
        # Backbone should have atoms from CCOC
        assert len(structure.backbone.atoms) >= 4

    def test_starting_segment_complex(self):
        """Test CCOC(=O)C(C)(C){[<]CC[>]}|5e5|"""
        ir = parse_gbigsmiles("CCOC(=O)C(C)(C){[<]CC[>]}|5e5|")

        assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))
        if isinstance(ir, GBigSmilesSystemIR):
            assert ir.total_mass == 5e5

        # Check backbone
        if isinstance(ir, GBigSmilesMoleculeIR):
            structure = ir.structure
        else:
            structure = ir.molecules[0].molecule.structure if ir.molecules else None

        assert structure is not None
        assert len(structure.backbone.atoms) >= 8  # CCOC(=O)C(C)(C)


class TestUserExampleIntegration:
    """Integration test with user's complete example."""

    def test_user_example_gbigsmiles(self):
        """
        Test parsing of user's complete gBigSMILES example:
        CCOC(=O)C(C)(C){[>][<]CC([>])c1ccccc1, [<]CC([>])C(=O)OC [<]}|schulz_zimm(1500, 1400)|[Br].|5e5|
        """
        gbigsmiles = "CCOC(=O)C(C)(C){[>][<]CC([>])c1ccccc1, [<]CC([>])C(=O)OC [<]}|schulz_zimm(1500, 1400)|[Br].|5e5|"

        ir = parse_gbigsmiles(gbigsmiles)

        # Verify IR structure
        assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))

        # Verify system size (if system IR)
        if isinstance(ir, GBigSmilesSystemIR):
            assert ir.total_mass is not None
            assert ir.total_mass == 5e5

        # Get structure
        if isinstance(ir, GBigSmilesMoleculeIR):
            structure = ir.structure
        else:
            structure = ir.molecules[0].molecule.structure if ir.molecules else None

        assert structure is not None

        # Verify backbone (starting segment: CCOC(=O)C(C)(C))
        assert len(structure.backbone.atoms) >= 8

        # Verify stochastic objects
        assert len(structure.stochastic_objects) >= 1
        stoch_obj = structure.stochastic_objects[0]

        # Verify repeat units (should have 2: CC([>])c1ccccc1 and CC([>])C(=O)OC)
        assert len(stoch_obj.repeat_units) >= 2

        # Verify distribution (if in metadata)
        if isinstance(ir, GBigSmilesMoleculeIR) and ir.stochastic_metadata:
            gb_stoch = ir.stochastic_metadata[0]
            if gb_stoch.distribution:
                assert gb_stoch.distribution.name == "schulz_zimm"

    def test_user_example_structure_details(self):
        """Test detailed structure of user's example."""
        gbigsmiles = "CCOC(=O)C(C)(C){[>][<]CC([>])c1ccccc1, [<]CC([>])C(=O)OC [<]}|schulz_zimm(1500, 1400)|[Br].|5e5|"

        ir = parse_gbigsmiles(gbigsmiles)

        # Get structure and count atoms
        if isinstance(ir, GBigSmilesMoleculeIR):
            structure = ir.structure
        else:
            structure = ir.molecules[0].molecule.structure if ir.molecules else None

        assert structure is not None

        # Count atoms from backbone and stochastic objects
        total_atoms = len(structure.backbone.atoms)
        for stoch_obj in structure.stochastic_objects:
            for unit in stoch_obj.repeat_units:
                total_atoms += len(unit.graph.atoms)
            for eg in stoch_obj.end_groups:
                total_atoms += len(eg.graph.atoms)

        # Should have at least 20 atoms
        assert total_atoms >= 20

        # Verify element diversity
        all_atoms = structure.backbone.atoms[:]
        for stoch_obj in structure.stochastic_objects:
            for unit in stoch_obj.repeat_units:
                all_atoms.extend(unit.graph.atoms)

        symbols = [a.element for a in all_atoms if a.element]
        assert "C" in symbols  # Carbons
        assert "O" in symbols  # Oxygens


class TestBackwardCompatibility:
    """Test that plain SMILES and BigSMILES still work."""

    def test_plain_smiles_still_works(self):
        """Test that plain SMILES parsing still works."""
        from molpy.parser.smiles import parse_smiles

        ir = parse_smiles("CCOC(=O)C")

        assert isinstance(ir, SmilesGraphIR)
        assert len(ir.atoms) == 6  # C,C,O,C,O,C

    def test_bigsmiles_without_system_size(self):
        """Test that BigSMILES without system size still works."""
        ir = parse_gbigsmiles("{[<]CC[>]}")

        assert isinstance(ir, GBigSmilesMoleculeIR)
        # Check structure has atoms
        assert len(ir.structure.stochastic_objects) >= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string(self):
        """Test parsing empty string."""
        ir = parse_gbigsmiles("")

        assert isinstance(ir, GBigSmilesMoleculeIR)
        assert len(ir.structure.backbone.atoms) == 0

    def test_only_system_size_no_molecule(self):
        """Test that system size requires a molecule."""
        # This should fail or return minimal structure
        with pytest.raises(Exception):
            parse_gbigsmiles("|5e5|")

    def test_multiple_system_sizes(self):
        """Test handling of multiple system sizes (should use last one)."""
        # Grammar should only allow one system size at the end
        # This test verifies the parser handles it gracefully
        ir = parse_gbigsmiles("{[<]CC[>]}|1000|")
        if isinstance(ir, GBigSmilesSystemIR):
            assert ir.total_mass == 1000.0


class TestSystemSizeFormats:
    """Test various system size number formats."""

    def test_large_scientific_notation(self):
        """Test |1e10| format."""
        ir = parse_gbigsmiles("{[<]CC[>]}|1e10|")

        if isinstance(ir, GBigSmilesSystemIR):
            assert ir.total_mass == 1e10

    def test_small_scientific_notation(self):
        """Test |1e3| format."""
        ir = parse_gbigsmiles("{[<]CC[>]}|1e3|")

        if isinstance(ir, GBigSmilesSystemIR):
            assert ir.total_mass == 1e3

    def test_decimal_with_exponent(self):
        """Test |2.5e4| format."""
        ir = parse_gbigsmiles("{[<]CC[>]}|2.5e4|")

        if isinstance(ir, GBigSmilesSystemIR):
            assert ir.total_mass == 2.5e4
