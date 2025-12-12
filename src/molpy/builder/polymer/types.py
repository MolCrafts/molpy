"""
Type definitions for polymer builder.

This module provides dataclasses for structured return values,
replacing the ad-hoc notes/dict approach.
"""

from dataclasses import dataclass, field
from typing import Any

from molpy.core.atomistic import Atom, Atomistic


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


# ============================================================================
# G-BigSMILES Stochastic Growth Data Structures
# ============================================================================


@dataclass
class PortDescriptor:
    """Descriptor for a reactive port on a monomer template.
    
    Attributes:
        descriptor_id: Unique ID within template (e.g., 0, 1, 2)
        port_name: Port name on atom (e.g., "<", ">", "branch")
        role: Port role (e.g., "left", "right", "branch")
        bond_kind: Bond type (e.g., "-", "=", "#")
        compat: Compatibility set for port matching
    """
    
    descriptor_id: int
    port_name: str
    role: str | None = None
    bond_kind: str | None = None
    compat: set[str] | None = None


@dataclass
class MonomerTemplate:
    """Template for a monomer with port descriptors and metadata.
    
    This represents a monomer type that can be instantiated multiple times
    during stochastic growth. Each instantiation creates a fresh copy of
    the structure.
    
    Attributes:
        label: Monomer label (e.g., "EO2", "PS")
        structure: Base Atomistic structure (will be copied on instantiation)
        port_descriptors: Mapping from descriptor_id to PortDescriptor
        mass: Molecular weight (g/mol)
        metadata: Additional metadata (optional)
    """
    
    label: str
    structure: Atomistic
    port_descriptors: dict[int, PortDescriptor]
    mass: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def instantiate(self) -> Atomistic:
        """Create a fresh copy of the structure.
        
        Returns:
            New Atomistic instance with independent atoms and bonds
        """
        return self.structure.copy()
    
    def get_port_by_descriptor(self, descriptor_id: int) -> PortDescriptor | None:
        """Get port descriptor for a specific descriptor ID.
        
        Args:
            descriptor_id: Descriptor ID to look up
            
        Returns:
            PortDescriptor if found, None otherwise
        """
        return self.port_descriptors.get(descriptor_id)
    
    def get_all_descriptors(self) -> list[PortDescriptor]:
        """Get all port descriptors for this template.
        
        Returns:
            List of all PortDescriptor objects
        """
        return list(self.port_descriptors.values())


@dataclass
class MonomerPlacement:
    """Decision for next monomer placement during stochastic growth.
    
    Attributes:
        template: MonomerTemplate to add
        target_descriptor_id: Which port descriptor on the new monomer to connect
    """
    
    template: "MonomerTemplate"
    target_descriptor_id: int


@dataclass
class StochasticChain:
    """Result of stochastic BFS growth.
    
    Attributes:
        polymer: The assembled Atomistic structure
        dp: Degree of polymerization (number of monomers added)
        mass: Total molecular weight (g/mol)
        growth_history: Metadata for each monomer addition step
    """
    
    polymer: Atomistic
    dp: int
    mass: float
    growth_history: list[dict[str, Any]] = field(default_factory=list)
