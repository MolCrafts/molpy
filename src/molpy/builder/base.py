import numpy as np
from itertools import product
from typing import Sequence, Iterable, Protocol, runtime_checkable, Optional, Union
from abc import ABC, abstractmethod
from molpy.core.atomistic import Atomistic

class BaseSiteProvider(Protocol):
    """
    Protocol for a position generator that provides positions for placing templates.
    
    Classes implementing this protocol should define a method `generate_positions`
    that returns an iterable of positions as numpy arrays.
    """
    
    @abstractmethod
    def gen_site(self, *args, **kwargs) -> np.ndarray:
        """Generate positions for placing templates."""

class BaseBuilder(ABC):
    """
    Abstract base class for all molecular structure builders.
    
    This class defines the common interface for builders that generate
    molecular structures by placing templates at generated positions.
    """
    
class StructBuilder(BaseBuilder):
    
    def __init__(self, site_provider: BaseSiteProvider):
        """
        Initialize builder with a position generator.
        
        Args:
            site_provider: Object that implements generate_positions method
        """
        self.site_provider = site_provider
    
    @abstractmethod
    def build(self, *args, **kwargs):
        """Build and return the molecular structure."""
        pass
    
    def _place_template_at_positions(
        self, 
        positions: np.ndarray, 
        template: Atomistic, 
        name: str = "structure"
    ) -> Atomistic:
        """
        Helper method to place template at given positions.
        
        Args:
            positions: Array of shape (n_positions, 3)
            template: Template structure to replicate
            name: Name for the resulting structure
            
        Returns:
            Combined structure with template placed at all positions
        """
        if len(positions) == 0:
            return Atomistic(name)
            
        replicas = []
        for i, pos in enumerate(positions):
            # Create a copy of the template
            replica = self._copy_structure(template)
            
            # Translate all atoms by the position offset
            for atom in replica.atoms:
                current_xyz = atom.get("xyz", np.array([0.0, 0.0, 0.0]))
                atom["xyz"] = current_xyz + pos
            
            replicas.append(replica)
        
        return Atomistic.concat(name, replicas)
    
    def _copy_structure(self, struct: Atomistic) -> Atomistic:
        """Create a deep copy of a structure."""
        new_struct = Atomistic(struct.get("name", "copy"))
        
        # Copy atoms
        for atom in struct.atoms:
            new_atom = atom.clone()
            new_struct.atoms.add(new_atom)
        
        # Copy bonds (if any)
        if hasattr(struct, 'bonds') and len(struct.bonds) > 0:
            atom_map = {id(old): new for old, new in zip(struct.atoms, new_struct.atoms)}
            for bond in struct.bonds:
                if id(bond.itom) in atom_map and id(bond.jtom) in atom_map:
                    new_bond = bond.clone()
                    new_bond.itom = atom_map[id(bond.itom)]
                    new_bond.jtom = atom_map[id(bond.jtom)]
                    new_struct.bonds.add(new_bond)
        
        return new_struct
