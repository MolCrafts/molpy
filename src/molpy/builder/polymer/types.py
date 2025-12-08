"""
Type definitions for polymer builder.

This module provides dataclasses for structured return values,
replacing the ad-hoc notes/dict approach.
"""

from dataclasses import dataclass, field
from typing import Any

from molpy.core.atomistic import Atomistic, Atom


@dataclass
class ConnectionMetadata:
    """Metadata about a single monomer connection step.

    Attributes:
        port_L: Name of the port used on the left monomer
        port_R: Name of the port used on the right monomer
        reaction_name: Name of the reaction used
        formed_bonds: List of newly formed bonds
        new_angles: List of newly created angles
        new_dihedrals: List of newly created dihedrals
        modified_atoms: Set of atoms whose types may have changed
        requires_retype: Whether retypification is needed
        entity_maps: List of entity mappings for port remapping
    """

    port_L: str
    port_R: str
    reaction_name: str
    formed_bonds: list[Any] = field(default_factory=list)
    new_angles: list[Any] = field(default_factory=list)
    new_dihedrals: list[Any] = field(default_factory=list)
    modified_atoms: set[Atom] = field(default_factory=set)
    requires_retype: bool = False
    entity_maps: list[dict[Atom, Atom]] = field(default_factory=list)


@dataclass
class ConnectionResult:
    """Result of connecting two monomers.

    Attributes:
        product: The resulting Atomistic assembly after connection
        metadata: Metadata about the connection
    """

    product: Atomistic
    metadata: ConnectionMetadata


@dataclass
class PolymerBuildResult:
    """Result of building a polymer.

    Attributes:
        polymer: The assembled Atomistic structure
        connection_history: List of connection metadata for each step
        total_steps: Total number of connection steps performed
    """

    polymer: Atomistic
    connection_history: list[ConnectionMetadata] = field(default_factory=list)
    total_steps: int = 0
