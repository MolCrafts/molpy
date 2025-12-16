"""Comprehensive tests for generative BigSMILES (gBigSMILES) parser.

This module tests all gBigSMILES features:
- System size: |NUMBER| notation
- Distributions: flory_schulz, schulz_zimm, gauss, uniform, log_normal, poisson
- Generation weights: bond descriptor reactivity weights
- Dot generation: end group specification
- Stochastic objects: block/random copolymers
"""

import pytest

from molpy.parser.smiles import (
    GBigSmilesMoleculeIR,
    GBigSmilesSystemIR,
    SmilesGraphIR,
    parse_gbigsmiles,
    parse_smiles,
)


# =============================================================================
# Test System Size Parsing
# =============================================================================


class TestSystemSize:
    """Test parsing of system size |NUMBER| notation."""

    @pytest.mark.parametrize(
        "name,gbigsmiles,expected_size",
        [
            ("integer", "{[<]CC[>]}|1000|", 1000.0),
            ("float", "{[<]CC[>]}|1500.5|", 1500.5),
            ("scientific_small", "{[<]CC[>]}|1e3|", 1e3),
            ("scientific_large", "{[<]CC[>]}|5e5|", 5e5),
            ("scientific_very_large", "{[<]CC[>]}|1e10|", 1e10),
            ("decimal_with_exponent", "{[<]CC[>]}|2.5e4|", 2.5e4),
            ("decimal_with_large_exponent", "{[<]CC[>]}|1.5e6|", 1.5e6),
        ],
    )
    def test_system_size_value(self, name, gbigsmiles, expected_size):
        """Test system size is parsed as correct value."""
        ir = parse_gbigsmiles(gbigsmiles)

        assert isinstance(ir, GBigSmilesSystemIR), f"Expected GBigSmilesSystemIR for {name}"
        assert ir.total_mass is not None, f"total_mass is None for {name}"
        assert abs(ir.total_mass - expected_size) < 0.01

    def test_no_system_size(self):
        """Test parsing without system size."""
        ir = parse_gbigsmiles("{[<]CC[>]}")

        assert isinstance(ir, GBigSmilesSystemIR)
        assert len(ir.molecules) == 1
        assert isinstance(ir.molecules[0].molecule, GBigSmilesMoleculeIR)


# =============================================================================
# Test Distribution Parsing
# =============================================================================


class TestDistributions:
    """Test parsing of all 6 distribution function types."""

    @pytest.mark.parametrize(
        "name,gbigsmiles,dist_name,param_count",
        [
            ("flory_schulz", "{[<]CC[>]}|flory_schulz(0.9)|", "flory_schulz", 1),
            ("schulz_zimm", "{[<]CC[>]}|schulz_zimm(1500, 3000)|", "schulz_zimm", 2),
            ("gauss", "{[<]CC[>]}|gauss(100, 20)|", "gauss", 2),
            ("uniform", "{[<]CC[>]}|uniform(50, 150)|", "uniform", 2),
            ("log_normal", "{[<]CC[>]}|log_normal(5.0, 0.5)|", "log_normal", 2),
            ("poisson", "{[<]CC[>]}|poisson(25)|", "poisson", 1),
        ],
    )
    def test_distribution_type(self, name, gbigsmiles, dist_name, param_count):
        """Test each distribution type parses correctly."""
        ir = parse_gbigsmiles(gbigsmiles)

        mol_ir = ir.molecules[0].molecule if isinstance(ir, GBigSmilesSystemIR) else ir

        distribution = next(
            (m.distribution for m in mol_ir.stochastic_metadata if m.distribution),
            None,
        )

        assert distribution is not None, f"Distribution not found for {name}"
        assert distribution.name == dist_name
        assert len(distribution.params) >= param_count

    def test_schulz_zimm_parameter_values(self):
        """Test Schulz-Zimm Mn and Mw parameters are extracted."""
        ir = parse_gbigsmiles("{[<]CC[>]}|schulz_zimm(2000, 4000)|")

        mol_ir = ir.molecules[0].molecule if isinstance(ir, GBigSmilesSystemIR) else ir
        dist = next(
            (m.distribution for m in mol_ir.stochastic_metadata if m.distribution), None
        )

        assert dist is not None
        param_values = sorted([float(v) for v in dist.params.values()])
        assert any(abs(v - 2000.0) < 1 for v in param_values)
        assert any(abs(v - 4000.0) < 1 for v in param_values)

    def test_distribution_and_system_size(self):
        """Test distribution and system size together."""
        ir = parse_gbigsmiles("{[<]CC[>]}|flory_schulz(0.9)||5e5|")

        assert isinstance(ir, GBigSmilesSystemIR)
        assert ir.total_mass == 5e5

        mol_ir = ir.molecules[0].molecule
        dist = next(
            (m.distribution for m in mol_ir.stochastic_metadata if m.distribution), None
        )
        assert dist is not None
        assert dist.name == "flory_schulz"


# =============================================================================
# Test Stochastic Objects
# =============================================================================


class TestStochasticObjects:
    """Test stochastic object parsing variations."""

    def test_homopolymer(self):
        """Test simple homopolymer: {[<]CC[>]}"""
        ir = parse_gbigsmiles("{[<]CC[>]}")

        structure = (
            ir.molecules[0].molecule.structure
            if isinstance(ir, GBigSmilesSystemIR)
            else ir.structure
        )

        assert len(structure.stochastic_objects) == 1
        assert len(structure.stochastic_objects[0].repeat_units) == 1

    def test_copolymer(self):
        """Test copolymer with multiple repeat units."""
        ir = parse_gbigsmiles("{[<]CC[>],[<]OCO[>]}")

        structure = (
            ir.molecules[0].molecule.structure
            if isinstance(ir, GBigSmilesSystemIR)
            else ir.structure
        )

        assert len(structure.stochastic_objects) == 1
        assert len(structure.stochastic_objects[0].repeat_units) >= 2

    def test_block_copolymer(self):
        """Test block copolymer: {[<]CC[>]}{[<]O[>]}"""
        ir = parse_gbigsmiles("{[<]CC[>]}{[<]O[>]}")

        structure = (
            ir.molecules[0].molecule.structure
            if isinstance(ir, GBigSmilesSystemIR)
            else ir.structure
        )

        # Should have 2 stochastic objects (blocks)
        assert len(structure.stochastic_objects) == 2

    def test_with_starting_segment(self):
        """Test with starting SMILES segment: CCOC{[<]CC[>]}"""
        ir = parse_gbigsmiles("CCOC{[<]CC[>]}")

        structure = (
            ir.molecules[0].molecule.structure
            if isinstance(ir, GBigSmilesSystemIR)
            else ir.structure
        )

        # Backbone should have atoms from CCOC
        assert len(structure.backbone.atoms) >= 4

    def test_indexed_descriptors(self):
        """Test indexed bond descriptors: {[<1]CC[>1]}"""
        ir = parse_gbigsmiles("{[<1]CC[>1]}")

        structure = (
            ir.molecules[0].molecule.structure
            if isinstance(ir, GBigSmilesSystemIR)
            else ir.structure
        )

        assert len(structure.stochastic_objects) >= 1

    def test_aa_type_descriptor(self):
        """Test AA-type (symmetric) bond descriptor: [$]"""
        ir = parse_gbigsmiles("{[$]CC[$]}")
        assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))


# =============================================================================
# Test End Groups
# =============================================================================


class TestEndGroups:
    """Test dot generation for end group specification."""

    def test_simple_end_group(self):
        """Test simple end group: [H]."""
        ir = parse_gbigsmiles("{[<]CC[>]}[H].")
        assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))

    def test_end_group_with_system_size(self):
        """Test end group with system size: [H].|5e5|"""
        ir = parse_gbigsmiles("{[<]CC[>]}[H].|5e5|")
        assert isinstance(ir, GBigSmilesSystemIR)
        assert ir.total_mass == 5e5

    def test_halogen_end_group(self):
        """Test halogen end group: [Br]."""
        ir = parse_gbigsmiles("{[<]CC[>]}[Br].")
        assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))


# =============================================================================
# Test Complex Examples
# =============================================================================


class TestComplexExamples:
    """Test complex gBigSMILES examples from papers and real use cases."""

    def test_peo_ps_block_copolymer(self):
        """Test PEO-PS block copolymer from paper."""
        gbigsmiles = "{[<]OCCOCCOCCOCCO[>]}{[<]CCc1ccccc1[>]}|schulz_zimm(1500, 3000)|[H].|1e4|"
        ir = parse_gbigsmiles(gbigsmiles)

        assert isinstance(ir, GBigSmilesSystemIR)
        assert ir.total_mass == 1e4

        structure = ir.molecules[0].molecule.structure
        assert len(structure.stochastic_objects) == 2

    def test_initiator_polymer_endgroup(self):
        """Test initiator-polymer-endgroup pattern."""
        gbigsmiles = "CCOC(=O)C(C)(C){[<]CC[>]}|flory_schulz(0.9)|[Br].|5e5|"
        ir = parse_gbigsmiles(gbigsmiles)

        assert isinstance(ir, GBigSmilesSystemIR)
        assert ir.total_mass == 5e5

        structure = ir.molecules[0].molecule.structure
        assert len(structure.backbone.atoms) >= 8

    def test_user_example_complete(self):
        """Test complete user example with all features."""
        gbigsmiles = (
            "CCOC(=O)C(C)(C){[>][<]CC([>])c1ccccc1, [<]CC([>])C(=O)OC [<]}"
            "|schulz_zimm(1500, 1400)|[Br].|5e5|"
        )
        ir = parse_gbigsmiles(gbigsmiles)

        assert isinstance(ir, GBigSmilesSystemIR)
        assert ir.total_mass == 5e5

        mol_ir = ir.molecules[0].molecule
        dist = next(
            (m.distribution for m in mol_ir.stochastic_metadata if m.distribution), None
        )
        assert dist is not None
        assert dist.name == "schulz_zimm"


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and backward compatibility."""

    def test_empty_string(self):
        """Test empty string returns valid IR."""
        ir = parse_gbigsmiles("")
        assert isinstance(ir, GBigSmilesSystemIR)

    def test_plain_smiles_compatibility(self):
        """Test plain SMILES still parses via gBigSMILES parser."""
        ir = parse_gbigsmiles("CCOC")

        structure = (
            ir.molecules[0].molecule.structure
            if isinstance(ir, GBigSmilesSystemIR)
            else ir.structure
        )

        assert len(structure.backbone.atoms) >= 4

    def test_smiles_parser_still_works(self):
        """Test that plain SMILES parser still works."""
        ir = parse_smiles("CCOC(=O)C")
        assert isinstance(ir, SmilesGraphIR)
        assert len(ir.atoms) == 6

    def test_only_system_size_no_molecule(self):
        """Test that system size requires a molecule."""
        with pytest.raises(Exception):
            parse_gbigsmiles("|5e5|")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
