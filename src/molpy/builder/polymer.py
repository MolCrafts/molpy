"""
Flexible PolymerBuilder for molpy - Template-based polymer construction.

This module provides a modular and extensible builder system for constructing
polymers using reusable monomer templates with context-aware anchor matching.
"""

from typing import Callable, Dict, List, Optional, Union, Any, Tuple, Literal
from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np
from numpy.typing import ArrayLike

from ..core.atomistic import AtomicStructure, Atom, Bond
from ..core.wrapper import Wrapper


@dataclass
class AnchorRule:
    """
    Context-aware anchor matching rule for polymer construction.

    Defines how an anchor atom should behave based on the context
    of neighboring monomers in the polymer chain.
    """

    name: str
    anchor: str | int
    deletes: Optional[List[str|int]] = field(default_factory=list)


@dataclass
class Monomer(Wrapper):
    """
    Template for a monomer unit with anchor definitions.

    Inherits from Wrapper to enable composable functionality.
    Contains the structural information and anchor rules needed
    to construct and connect monomers in polymer chains.
    """

    anchors: dict[str, AnchorRule]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        struct: Union[AtomicStructure, Wrapper],
        anchors: list[AnchorRule] = [],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Monomer with struct, anchors, and metadata."""
        super().__init__(struct)
        self.anchors = {anchor.name: anchor for anchor in anchors}
        self.metadata = metadata or {}

    def clone(self) -> AtomicStructure:
        """Create a deep copy of the underlying structure."""
        return deepcopy(self.unwrap())

    def transformed(
        self,
        position: Optional[ArrayLike] = None,
        rotation: Optional[Tuple[float, ArrayLike]] = None,
        name: Optional[str] = None,
    ) -> AtomicStructure:
        """
        Create a transformed copy of the monomer.

        Args:
            position: Translation vector [x, y, z]
            rotation: Tuple of (angle_radians, axis_vector)
            name: New name for the instance

        Returns:
            Transformed copy of the underlying structure
        """
        inst = self.clone()
        if name:
            inst["name"] = name
        if position:
            # Apply translation directly to atom coordinates
            for atom in inst.atoms:
                current_xyz = atom.xyz
                atom.xyz = current_xyz + np.array(position)
        if rotation:
            # Apply rotation directly to atom coordinates
            angle, axis = rotation
            axis = np.array(axis)
            axis = axis / np.linalg.norm(axis)  # normalize
            for atom in inst.atoms:
                current_xyz = atom.xyz
                # Simple rotation around axis (could be improved with proper rotation matrix)
                # For now, just apply a placeholder transformation
                atom.xyz = current_xyz
        return inst


class PolymerBuilder:
    """
    Flexible builder for constructing polymers from monomer templates.

    Supports context-aware anchor matching and modular monomer registration.
    """

    def __init__(self, factory: Optional[Callable[..., AtomicStructure]] = None):
        """
        Initialize the polymer builder.

        Args:
            factory: Factory function to create empty structures (defaults to AtomicStructure)
        """
        self.monomers: Dict[str, Monomer] = {}
        self.factory = factory or AtomicStructure

    def register_monomer(
        self, name: str, template: Monomer
    ) -> "PolymerBuilder":
        """
        Register a monomer template with the builder.

        Args:
            name: Monomer identifier
            template: Monomer instance

        Returns:
            Self for method chaining
        """
        self.monomers[name] = template
        return self

    def build_linear(self, sequence: List[str], **kwargs) -> AtomicStructure:
        """
        Build a linear polymer from a sequence of monomer names.

        Args:
            sequence: List of monomer names in order
            **kwargs: Additional build parameters

        Returns:
            AtomicStructure containing the built polymer
        """
        if not sequence:
            return self.factory()

        # Create polymer structure
        polymer = self.factory(name=f"polymer_{'-'.join(sequence)}")

        # Add monomers one by one
        for i, monomer_name in enumerate(sequence):
            if monomer_name not in self.monomers:
                raise ValueError(f"Unknown monomer: {monomer_name}")

            template = self.monomers[monomer_name]

            # Determine context for anchor selection
            prev_monomer = sequence[i - 1] if i > 0 else None
            next_monomer = sequence[i + 1] if i < len(sequence) - 1 else None

            # Create monomer instance
            monomer_inst = template.transformed(name=f"{monomer_name}_{i}")

            # Add to polymer (simplified - in practice would handle bonding)
            polymer.add_struct(monomer_inst)

        return polymer

    def get_anchor_atoms(
        self,
        template: Monomer,
        anchor_name: str,
        prev_monomer: Optional[str] = None,
        next_monomer: Optional[str] = None,
    ) -> List[Atom]:
        """
        Get anchor atoms based on context-aware rules.

        Args:
            template: Monomer to get anchors from
            anchor_name: Name of the anchor type
            prev_monomer: Previous monomer in sequence (for context)
            next_monomer: Next monomer in sequence (for context)

        Returns:
            List of Atom objects that match the context
        """
        if anchor_name not in template.anchors:
            return []

        rules = template.anchors[anchor_name]
        matching_atoms = []

        for rule in rules:
            if rule.matches_context(prev_monomer, next_monomer):
                # Find atoms matching the rule
                struct = template.unwrap()
                for atom in struct.atoms:
                    if atom.get("name") == rule.anchor_atom:
                        # Apply patch if specified
                        if rule.patch:
                            rule.patch(atom)
                        matching_atoms.append(atom)

        return matching_atoms

    def build_branched(
        self, backbone_sequence: List[str], branches: Dict[int, List[str]], **kwargs
    ) -> AtomicStructure:
        """
        Build a branched polymer with side chains.

        Args:
            backbone_sequence: Main chain monomer sequence
            branches: Dict mapping backbone positions to branch sequences
            **kwargs: Additional build parameters

        Returns:
            AtomicStructure containing the branched polymer
        """
        # Start with linear backbone
        polymer = self.build_linear(backbone_sequence, **kwargs)

        # Add branches (simplified implementation)
        for position, branch_sequence in branches.items():
            if 0 <= position < len(backbone_sequence):
                branch = self.build_linear(branch_sequence)
                # In practice, would connect branch to backbone at specific position
                polymer.add_struct(branch)

        return polymer

    def validate_sequence(self, sequence: List[str]) -> bool:
        """
        Validate that all monomers in sequence are registered.

        Args:
            sequence: List of monomer names

        Returns:
            True if all monomers are available
        """
        return all(name in self.monomers for name in sequence)

    def available_monomers(self) -> List[str]:
        """Get list of available monomer names."""
        return list(self.monomers.keys())

    def __repr__(self):
        return f"PolymerBuilder(monomers={list(self.monomers.keys())})"
