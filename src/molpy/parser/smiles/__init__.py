"""SMILES, BigSMILES, GBigSMILES, and CGSmiles parsers.

This module provides four explicit parser APIs:
- parse_smiles: Parse pure SMILES strings
- parse_bigsmiles: Parse BigSMILES strings
- parse_gbigsmiles: Parse GBigSMILES strings
- parse_cgsmiles: Parse CGSmiles strings

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
from .cgsmiles_ir import (
    CGSmilesBondIR,
    CGSmilesFragmentIR,
    CGSmilesGraphIR,
    CGSmilesIR,
    CGSmilesNodeIR,
)
from .cgsmiles_parser import CGSmilesParserImpl, parse_cgsmiles

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

__all__ = [
    "parse_smiles",
    "parse_bigsmiles",
    "parse_gbigsmiles",
    "parse_cgsmiles",
    "bigsmilesir_to_monomer",
    "bigsmilesir_to_polymerspec",
    "PolymerSpec",
    "PolymerSegment",
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
    # CGSmiles IR
    "CGSmilesIR",
    "CGSmilesGraphIR",
    "CGSmilesNodeIR",
    "CGSmilesBondIR",
    "CGSmilesFragmentIR",
]
