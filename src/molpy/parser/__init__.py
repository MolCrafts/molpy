"""Unified parser API for SMILES, BigSMILES, GBigSMILES, CGSmiles, and SMARTS.

Convenience wrappers live here so downstream code can do::

    from molpy.parser import parse_molecule, parse_polymer, parse_smarts
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .smarts import SmartsParser
from .smiles import (
    PolymerSpec,
    PolymerSegment,
    bigsmilesir_to_monomer,
    bigsmilesir_to_polymerspec,
    parse_bigsmiles,
    parse_cgsmiles,
    parse_gbigsmiles,
    parse_smiles,
    smilesir_to_atomistic,
)

if TYPE_CHECKING:
    from molpy.core.atomistic import Atomistic
    from .smarts import SmartsIR


# ---------------------------------------------------------------------------
# Singleton parser instances
# ---------------------------------------------------------------------------
_smarts_parser = SmartsParser()


# ---------------------------------------------------------------------------
# Free-function wrappers
# ---------------------------------------------------------------------------


def parse_smarts(pattern: str) -> "SmartsIR":
    """Parse a SMARTS pattern string into :class:`SmartsIR`.

    This is a thin wrapper around ``SmartsParser().parse_smarts(pattern)``.

    Args:
        pattern: SMARTS string.

    Returns:
        Parsed :class:`SmartsIR` representation.
    """
    return _smarts_parser.parse_smarts(pattern)


def parse_molecule(smiles: str) -> "Atomistic":
    """Parse a SMILES string and return a single :class:`Atomistic` structure.

    Args:
        smiles: SMILES string for a single molecule (no dots).

    Returns:
        :class:`Atomistic` structure.
    """
    ir = parse_smiles(smiles)
    if isinstance(ir, list):
        # Dot-separated: take the first component
        ir = ir[0]
    return smilesir_to_atomistic(ir)


def parse_mixture(smiles: str) -> "list[Atomistic]":
    """Parse a (possibly dot-separated) SMILES string into a list of molecules.

    Args:
        smiles: SMILES string, components separated by ``'.'``.

    Returns:
        List of :class:`Atomistic` structures (always a list, even for one).
    """
    ir = parse_smiles(smiles)
    if not isinstance(ir, list):
        ir = [ir]
    return [smilesir_to_atomistic(component) for component in ir]


def parse_monomer(bigsmiles: str) -> "Atomistic":
    """Parse a BigSMILES string and return the first monomer as :class:`Atomistic`.

    Args:
        bigsmiles: BigSMILES string.

    Returns:
        Monomer :class:`Atomistic` structure with port annotations.
    """
    ir = parse_bigsmiles(bigsmiles)
    return bigsmilesir_to_monomer(ir)


def parse_polymer(bigsmiles: str) -> PolymerSpec:
    """Parse a BigSMILES string and return a :class:`PolymerSpec`.

    Args:
        bigsmiles: BigSMILES string.

    Returns:
        :class:`PolymerSpec` describing segments, topology, and monomers.
    """
    ir = parse_bigsmiles(bigsmiles)
    return bigsmilesir_to_polymerspec(ir)


__all__ = [
    # Parser classes
    "SmartsParser",
    # Free functions
    "parse_smiles",
    "parse_bigsmiles",
    "parse_gbigsmiles",
    "parse_cgsmiles",
    "parse_smarts",
    "parse_molecule",
    "parse_mixture",
    "parse_monomer",
    "parse_polymer",
    # Conversion helpers
    "smilesir_to_atomistic",
    "bigsmilesir_to_monomer",
    "bigsmilesir_to_polymerspec",
    # Data classes
    "PolymerSpec",
    "PolymerSegment",
]
