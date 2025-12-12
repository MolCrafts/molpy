"""Unit tests for GBigSMILES distribution parsing."""

import pytest

from molpy.parser.smiles import parse_gbigsmiles
from molpy.parser.smiles.gbigsmiles_ir import GBigSmilesMoleculeIR, GBigSmilesSystemIR


def test_parse_schulz_zimm_at_chain_level():
    """Test parsing schulz_zimm distribution at chain level."""
    gbigsmiles = (
        "{[<]OCCOCCOCCOCCO[>]}{[<]CCc1ccccc1[>]}|schulz_zimm(1500, 3000)|[H].|1e4|"
    )

    ir = parse_gbigsmiles(gbigsmiles)

    # Should parse successfully
    assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))

    # Get the molecule IR
    if isinstance(ir, GBigSmilesSystemIR):
        mol_ir = ir.molecules[0].molecule
    else:
        mol_ir = ir

    # Check that distribution was extracted
    assert mol_ir.stochastic_metadata is not None
    assert len(mol_ir.stochastic_metadata) > 0

    # Find distribution in metadata
    distribution = None
    for meta in mol_ir.stochastic_metadata:
        if meta.distribution:
            distribution = meta.distribution
            break

    # Distribution must be found
    assert distribution is not None, "Distribution should be extracted by parser"
    assert distribution.name == "schulz_zimm"
    assert len(distribution.params) > 0

    # Check parameters
    # Params might be stored as "0"/"1" or "Mn"/"Mw" or positional values
    param_values = list(distribution.params.values())
    assert (
        len(param_values) >= 2
    ), f"Expected at least 2 params, got {distribution.params}"

    # First param should be Mn (1500), second should be Mw (3000)
    # Order might vary, so check both
    values = sorted([float(v) for v in param_values[:2]])
    assert values[0] == 1500.0, f"Expected Mn=1500, got {values[0]}"
    assert values[1] == 3000.0, f"Expected Mw=3000, got {values[1]}"


def test_parse_schulz_zimm_different_values():
    """Test parsing schulz_zimm with different Mn/Mw values."""
    gbigsmiles = "{[<]CC[>]}|schulz_zimm(2000, 4000)|[H]."

    ir = parse_gbigsmiles(gbigsmiles)
    assert isinstance(ir, (GBigSmilesMoleculeIR, GBigSmilesSystemIR))

    # Get the molecule IR
    if isinstance(ir, GBigSmilesSystemIR):
        mol_ir = ir.molecules[0].molecule
    else:
        mol_ir = ir

    # Find distribution
    distribution = None
    for meta in mol_ir.stochastic_metadata:
        if meta.distribution:
            distribution = meta.distribution
            break

    assert distribution is not None
    assert distribution.name == "schulz_zimm"

    # Check parameters
    param_values = list(distribution.params.values())
    assert len(param_values) >= 2
    values = sorted([float(v) for v in param_values[:2]])
    assert values[0] == 2000.0
    assert values[1] == 4000.0


def test_parse_schulz_zimm_with_system_size():
    """Test parsing schulz_zimm with system size."""
    gbigsmiles = "{[<]CC[>]}|schulz_zimm(1000, 2000)|[H].|5e4|"

    ir = parse_gbigsmiles(gbigsmiles)
    assert isinstance(ir, GBigSmilesSystemIR)
    assert ir.total_mass == 50000.0

    # Check distribution
    mol_ir = ir.molecules[0].molecule
    distribution = None
    for meta in mol_ir.stochastic_metadata:
        if meta.distribution:
            distribution = meta.distribution
            break

    assert distribution is not None
    assert distribution.name == "schulz_zimm"


def test_parse_multiple_stochastic_objects_with_distribution():
    """Test that distribution applies to first stochastic_object when multiple exist."""
    gbigsmiles = (
        "{[<]OCCOCCOCCOCCO[>]}{[<]CCc1ccccc1[>]}|schulz_zimm(1500, 3000)|[H].|1e4|"
    )

    ir = parse_gbigsmiles(gbigsmiles)

    if isinstance(ir, GBigSmilesSystemIR):
        mol_ir = ir.molecules[0].molecule
    else:
        mol_ir = ir

    # Should have multiple stochastic objects
    structure = mol_ir.structure
    assert len(structure.stochastic_objects) >= 2

    # Distribution should be in metadata for at least one stochastic object
    distributions_found = [
        meta.distribution
        for meta in mol_ir.stochastic_metadata
        if meta.distribution is not None
    ]

    assert len(distributions_found) > 0, "At least one distribution should be found"
    assert distributions_found[0].name == "schulz_zimm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

