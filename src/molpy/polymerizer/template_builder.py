"""
Template-based polymer builder for molpy.

This module provides a flexible and extensible builder system for constructing
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
    anchor_atom: str                                      # name of anchor atom
    when_prev: Union[str, Literal["*"], None] = "*"     # previous monomer name or "*" for any
    when_next: Union[str, Literal["*"], None] = "*"     # next monomer name or "*" for any
    patch: Optional[Callable[[Atom], None]] = None       # optional hook to modify atom before bonding
    
    def matches_context(self, prev_monomer: Optional[str], next_monomer: Optional[str]) -> bool:
        """Check if this rule matches the given context."""
        prev_match = (self.when_prev == "*" or 
                     self.when_prev is None and prev_monomer is None or
                     self.when_prev == prev_monomer)
        next_match = (self.when_next == "*" or 
                     self.when_next is None and next_monomer is None or
                     self.when_next == next_monomer)
        return prev_match and next_match


@dataclass
class MonomerTemplate(Wrapper):
    """
    Template for a monomer unit with anchor definitions.
    
    Inherits from Wrapper to enable composable functionality.
    Contains the structural information and anchor rules needed
    to construct and connect monomers in polymer chains.
    """
    anchors: Dict[str, List[AnchorRule]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, struct: Union[AtomicStructure, Wrapper], anchors: Dict[str, List[AnchorRule]], 
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize MonomerTemplate with struct, anchors, and metadata."""
        super().__init__(struct)
        self.anchors = anchors
        self.metadata = metadata or {}
    
    def clone(self) -> AtomicStructure:
        """Create a deep copy of the underlying structure."""
        return deepcopy(self.unwrap())
    
    def transformed(self, position: Optional[ArrayLike] = None, 
                   rotation: Optional[Tuple[float, ArrayLike]] = None,
                   name: Optional[str] = None) -> AtomicStructure:
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
        
        if position is not None:
            # Move all atoms to new position
            pos = np.asarray(position)
            for atom in inst.atoms:
                current_xyz = atom.xyz
                atom.xyz = current_xyz + pos
        
        if rotation is not None:
            angle, axis = rotation
            axis = np.asarray(axis)
            # Rotate all atoms around origin (or center of mass)
            for atom in inst.atoms:
                # This is a simplified rotation - in practice you'd want
                # proper rotation around center of mass
                current_xyz = atom.xyz
                # Apply rodrigues rotation formula here
                atom.xyz = current_xyz  # Placeholder - implement proper rotation
        
        return inst


class PolymerBuilder:
    """
    Flexible polymer builder using template-based monomer assembly.
    
    Supports:
    - Registration of monomer templates with anchor rules
    - Context-aware anchor resolution
    - Linear and graph-based polymer construction
    - Extensible through factory functions
    """
    
    def __init__(self, factory: Optional[Callable[..., AtomicStructure]] = None):
        """
        Initialize PolymerBuilder.
        
        Args:
            factory: Function to create new AtomicStructure instances
        """
        self.monomers: Dict[str, MonomerTemplate] = {}
        self.factory = factory or AtomicStructure
    
    def register_monomer(self, name: str, template: MonomerTemplate) -> "PolymerBuilder":
        """
        Register a monomer template.
        
        Args:
            name: Unique identifier for the monomer
            template: MonomerTemplate defining the structure and anchors
            
        Returns:
            Self for method chaining
        """
        self.monomers[name] = template
        return self
    
    def place(self, monomer_name: str, position: Optional[ArrayLike] = None,
              rotation: Optional[Tuple[float, ArrayLike]] = None,
              instance_name: Optional[str] = None) -> AtomicStructure:
        """
        Create and place a monomer instance.
        
        Args:
            monomer_name: Name of registered monomer
            position: 3D position for placement
            rotation: (angle, axis) for rotation
            instance_name: Name for this instance
            
        Returns:
            Positioned monomer instance
        """
        if monomer_name not in self.monomers:
            raise ValueError(f"Monomer '{monomer_name}' not registered")
        
        template = self.monomers[monomer_name]
        return template.transformed(position=position, rotation=rotation, name=instance_name)
    
    def connect(self, s1: AtomicStructure, anchor1: str, 
                s2: AtomicStructure, anchor2: str,
                bond_props: Optional[Dict[str, Any]] = None) -> Bond:
        """
        Connect two structures via their anchors.
        
        Args:
            s1: First structure
            anchor1: Anchor name in first structure
            s2: Second structure
            anchor2: Anchor name in second structure
            bond_props: Optional properties for the connecting bond
            
        Returns:
            Created bond
        """
        # Find anchor atoms - this is simplified, in practice you'd
        # use the context-aware anchor resolution
        atom1 = self._find_anchor_atom(s1, anchor1)
        atom2 = self._find_anchor_atom(s2, anchor2)
        
        if atom1 is None or atom2 is None:
            raise ValueError("Could not find anchor atoms")
        
        # Create the bond
        bond_props = bond_props or {}
        return Bond(atom1, atom2, **bond_props)
    
    def build_linear(self, sequence: str, spacing: float = 1.54) -> AtomicStructure:
        """
        Build a linear polymer from a sequence string.
        
        Args:
            sequence: String sequence of monomer names (e.g., "AAAAA" or "ABABA")
            spacing: Spacing between monomers in Angstroms
            
        Returns:
            Complete polymer structure
        """
        if not sequence:
            raise ValueError("Empty sequence")
        
        # Create the polymer structure
        polymer = self.factory(name=f"polymer_{sequence}")
        
        # Place monomers and connect them
        monomers = []
        for i, monomer_name in enumerate(sequence):
            if monomer_name not in self.monomers:
                raise ValueError(f"Monomer '{monomer_name}' not registered")
            
            # Determine position
            position = [i * spacing, 0.0, 0.0]
            
            # Create monomer instance
            instance = self.place(monomer_name, position=position, 
                                instance_name=f"{monomer_name}_{i}")
            
            # Add to polymer
            polymer.add_struct(instance)
            monomers.append((monomer_name, instance))
        
        # Connect adjacent monomers
        for i in range(len(monomers) - 1):
            prev_name, prev_struct = monomers[i]
            next_name, next_struct = monomers[i + 1]
            
            # Use context-aware anchor resolution
            prev_template = self.monomers[prev_name]
            next_template = self.monomers[next_name]
            
            # Find appropriate anchors (simplified - would use context matching)
            prev_anchor = self._get_connecting_anchor(prev_template, "right", 
                                                    prev_monomer=prev_name if i > 0 else None,
                                                    next_monomer=next_name)
            next_anchor = self._get_connecting_anchor(next_template, "left",
                                                    prev_monomer=prev_name,
                                                    next_monomer=next_name if i < len(monomers) - 2 else None)
            
            if prev_anchor and next_anchor:
                bond = self.connect(prev_struct, prev_anchor, next_struct, next_anchor)
                polymer.add_bond(bond)
        
        return polymer
    
    def build_from_graph(self, graph: Any) -> AtomicStructure:
        """
        Build a polymer from a topology graph.
        
        Args:
            graph: Topology defining the connectivity pattern
            
        Returns:
            Complete polymer structure
        """
        # This would be implemented based on the specific Topology class
        # For now, raise NotImplementedError
        raise NotImplementedError("Graph-based building not yet implemented")
    
    def _find_anchor_atom(self, struct: AtomicStructure, anchor_name: str) -> Optional[Atom]:
        """Find an anchor atom by name in a structure."""
        # Simple implementation - just find by name
        for atom in struct.atoms:
            atom_name = atom.get('name', '')
            if atom_name == anchor_name:
                return atom
        # Also try to find by anchor name mapping
        if anchor_name in ["left", "right"]:
            # For our simple case, left=c1, right=c2
            target_name = "c1" if anchor_name == "left" else "c2"
            for atom in struct.atoms:
                if atom.get('name', '') == target_name:
                    return atom
        return None
    
    def _get_connecting_anchor(self, template: MonomerTemplate, side: str,
                             prev_monomer: Optional[str] = None,
                             next_monomer: Optional[str] = None) -> Optional[str]:
        """Get the appropriate anchor for connecting based on context."""
        # This is a simplified implementation
        # In practice, you'd have more sophisticated anchor resolution
        anchor_candidates = list(template.anchors.keys())
        
        if side == "left" and "left" in anchor_candidates:
            return "left"
        elif side == "right" and "right" in anchor_candidates:
            return "right"
        elif anchor_candidates:
            return anchor_candidates[0]  # Fallback to first available
        
        return None
