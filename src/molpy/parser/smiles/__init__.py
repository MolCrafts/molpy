"""SMILES, BigSMILES, and GBigSMILES parsers.

This module provides three explicit parser APIs:
- parse_smiles: Parse pure SMILES strings
- parse_bigsmiles: Parse BigSMILES strings
- parse_gbigsmiles: Parse GBigSMILES strings

Each parser uses its own dedicated grammar and transformer.
"""

from .bigsmiles_ir import (
    BigSmilesMoleculeIR,
    BigSmilesSubgraphIR,
    BondingDescriptorIR,
    EndGroupIR,
    RepeatUnitIR,
    StochasticObjectIR,
    TerminalDescriptorIR,
)
from .bigsmiles_parser import BigSmilesParserImpl
from .gbigsmiles_ir import (
    DistributionIR,
    GBBondingDescriptorIR,
    GBigSmilesComponentIR,
    GBigSmilesMoleculeIR,
    GBigSmilesSystemIR,
    GBStochasticObjectIR,
)
from .gbigsmiles_parser import GBigSmilesParserImpl
from .smiles_ir import SmilesAtomIR, SmilesBondIR, SmilesGraphIR
from .smiles_parser import SmilesParserImpl

# Parser instances (singleton pattern)
_smiles_parser = SmilesParserImpl()
_bigsmiles_parser = BigSmilesParserImpl()
_gbigsmiles_parser = GBigSmilesParserImpl()


def parse_smiles(src: str) -> SmilesGraphIR | list[SmilesGraphIR]:
    """
    Parse a SMILES string into SmilesGraphIR or list of SmilesGraphIR.

    This parser only accepts pure SMILES syntax. It will reject
    BigSMILES or GBigSMILES constructs.

    For dot-separated SMILES (e.g., "C.C", "CC.O"), returns a list of
    SmilesGraphIR, one for each disconnected component.

    Args:
        src: SMILES string (may contain dots for mixtures)

    Returns:
        SmilesGraphIR for single molecule, or list[SmilesGraphIR] for mixtures

    Raises:
        ValueError: if syntax errors detected or unclosed rings

    Examples:
        >>> ir = parse_smiles("CCO")
        >>> len(ir.atoms)
        3
        >>> irs = parse_smiles("C.C")
        >>> len(irs)
        2
    """
    return _smiles_parser.parse(src)


def parse_bigsmiles(src: str) -> BigSmilesMoleculeIR:
    """
    Parse a BigSMILES string into BigSmilesMoleculeIR.

    This parser accepts BigSMILES syntax including stochastic objects,
    bond descriptors, and repeat units. It does NOT accept GBigSMILES
    annotations.

    Args:
        src: BigSMILES string

    Returns:
        BigSmilesMoleculeIR containing backbone and stochastic objects

    Raises:
        ValueError: if syntax errors detected

    Examples:
        >>> ir = parse_bigsmiles("{[<]CC[>]}")
        >>> len(ir.stochastic_objects)
        1
    """
    return _bigsmiles_parser.parse(src)


def parse_gbigsmiles(src: str) -> GBigSmilesSystemIR:
    """
    Parse a GBigSMILES string into GBigSmilesSystemIR.

    This parser accepts GBigSMILES syntax including all BigSMILES
    features plus system size specifications and other generative
    annotations. Always returns GBigSmilesSystemIR, wrapping single
    molecules in a system structure.

    Args:
        src: GBigSMILES string

    Returns:
        GBigSmilesSystemIR containing the parsed system

    Raises:
        ValueError: if syntax errors detected

    Examples:
        >>> ir = parse_gbigsmiles("{[<]CC[>]}|5e5|")
        >>> isinstance(ir, GBigSmilesSystemIR)
        True
    """
    return _gbigsmiles_parser.parse(src)


# Import conversion functions from converter module
from .converter import (
    bigsmilesir_to_monomer,
    bigsmilesir_to_polymerspec,
    PolymerSpec,
    PolymerSegment,
)


def parse_gbigsmiles_to_polymerspec(src: str) -> PolymerSpec:
    """
    Parse GBigSMILES string directly to PolymerSpec.

    Convenience function that combines parsing and conversion.

    Args:
        src: GBigSMILES string

    Returns:
        PolymerSpec containing all monomers, end groups, topology, and original IR

    Examples:
        >>> spec = parse_gbigsmiles_to_polymerspec("{[<]CC[>]}")
        >>> len(spec.segments)
        1
        >>> len(spec.segments[0].repeat_units_ir)
        1
    """
    ir = parse_gbigsmiles(src)
    # Extract BigSmilesMoleculeIR from gBigSMILES IR
    if not ir.molecules:
        raise ValueError("Cannot convert empty gBigSMILES IR to PolymerSpec")
    # Use first molecule's structure
    return bigsmilesir_to_polymerspec(ir.molecules[0].molecule.structure)


# Backward compatibility: SmilesParser class
class SmilesParser:
    """
    Backward compatibility wrapper for the old SmilesParser API.

    This class provides the same interface as the old SmilesParser,
    but uses the new parser implementations under the hood.

    Examples:
        >>> parser = SmilesParser()
        >>> ir = parser.parse_bigsmiles("{[<]CC[>]}")
        >>> ir = parser.parse_smiles("CCO")
    """

    def parse_smiles(self, smiles: str) -> SmilesGraphIR:
        """Parse SMILES string into SmilesGraphIR."""
        return parse_smiles(smiles)

    def parse_bigsmiles(self, text: str) -> BigSmilesMoleculeIR:
        """Parse BigSMILES string into BigSmilesMoleculeIR."""
        return parse_bigsmiles(text)

    def parse_gbigsmiles(self, text: str) -> GBigSmilesSystemIR:
        """Parse GBigSMILES string into GBigSmilesSystemIR."""
        return parse_gbigsmiles(text)


__all__ = [
    "parse_smiles",
    "parse_bigsmiles",
    "parse_gbigsmiles",
    "bigsmilesir_to_monomer",
    "bigsmilesir_to_polymerspec",
    "PolymerSpec",
    "PolymerSegment",
    "SmilesParser",
    # SMILES IR
    "SmilesGraphIR",
    "SmilesAtomIR",
    "SmilesBondIR",
    # BigSMILES IR
    "BigSmilesMoleculeIR",
    "BigSmilesSubgraphIR",
    "BondingDescriptorIR",
    "StochasticObjectIR",
    "TerminalDescriptorIR",
    "RepeatUnitIR",
    "EndGroupIR",
    # gBigSMILES IR
    "GBigSmilesMoleculeIR",
    "GBigSmilesSystemIR",
    "GBigSmilesComponentIR",
    "GBBondingDescriptorIR",
    "GBStochasticObjectIR",
    "DistributionIR",
]
