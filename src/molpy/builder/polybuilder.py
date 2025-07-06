"""
Flexible PolymerBuilder for molpy - Digital Proportional Divider for Molecular Construction.

This module implements a digital equivalent of a proportional divider (比例卡尺) 
for constructing polymers. The core principle mirrors the classical mechanical tool:

PROPORTIONAL DIVIDER ANALOGY:
┌─────────────────────────────────────┐
│  Input Arm ---- Pivot ---- Output   │
│  (Template)     (Path)     (Result) │
│                                     │
│  Monomer  -->  Transform -->  Chain │
│  Template       (R+T)        Segment│
└─────────────────────────────────────┘

Key Correspondences:
- Pivot Points    ↔ Path coordinates path[i]  
- Input Arm       ↔ Monomer templates
- Output Arm      ↔ Placed monomers
- Proportional    ↔ Rotation + Translation
  Transformation    transformation matrix

This approach provides:
1. Intuitive geometric control through path definition
2. Automatic orientation alignment along path direction  
3. Consistent proportional scaling across the structure
4. Modular monomer templates with reusable anchor rules
"""
from dataclasses import dataclass, field
import numpy as np
import copy
from typing import Callable
from ..core.atomistic import Atomistic
from ..op.geometry import rotation_matrix_from_vectors


@dataclass
class AnchorRule:
    """
    Context-aware anchor matching rule for polymer construction.

    Defines how an anchor atom should behave based on the context
    of neighboring monomers in the polymer chain.
    """

    init: int
    end: int
    deletes: list[int] = field(default_factory=list)


class Monomer(Atomistic):
    """
    Template for a monomer unit with anchor definitions.

    Inherits from  to enable composable functionality.
    Contains the structural information and anchor rules needed
    to construct and connect monomers in polymer chains.
    """

    def __init__(
        self,
        anchors: list[AnchorRule] | None = None,
    ):
        """Initialize Monomer with struct, anchors, and metadata."""
        super().__init__()
        # Store anchors as direct attribute instead of going through wrapper
        object.__setattr__(self, '_anchors', anchors or [])

    @property
    def anchors(self) -> list[AnchorRule]:
        """Get the anchor rules for this monomer."""
        return getattr(self, '_anchors', [])

    def __call__(self) -> 'Monomer':
        """Create a deep copy of this monomer."""
        # Atomistic's __call__ already returns a Monomer instance with all structural data copied
        copy = super().__call__()
        
        # Just copy the anchors attribute
        object.__setattr__(copy, '_anchors', self.anchors.copy())
        
        return copy


class PolymerBuilder:
    """
    Digital Proportional Divider for polymer construction.
    
    This class implements the mathematical equivalent of a proportional divider,
    a classical mechanical tool used for scaling and transferring measurements.
    
    WORKING PRINCIPLE:
    ==================
    1. Path Definition: Acts as the "track" or "scale" of the proportional divider
    2. Pivot Points: Each path[i] serves as a transformation center (支点)
    3. Input Templates: Monomer structures define the "input arm" geometry  
    4. Output Placement: Transformed monomers form the "output arm" result
    5. Proportional Transform: Rotation + Translation preserves geometric relationships
    
    TRANSFORMATION SEQUENCE:
    =======================
    For each monomer placement:
    1. Identify pivot point: path[i] 
    2. Calculate native direction: template orientation
    3. Calculate target direction: path segment direction
    4. Apply rotation: align native → target direction
    5. Apply translation: anchor point → pivot point
    6. Handle connectivity: apply deletion rules for chain formation
    
    This approach ensures that:
    - Geometric proportions are preserved
    - Orientations follow the path naturally
    - Chemical connectivity is maintained
    - Complex geometries (linear, curved, 3D) are supported seamlessly
    """

    def __init__(self, monomers: dict[str, Callable[[], Monomer]]):
        """
        Initialize the digital proportional divider with monomer templates.

        Parameters
        ----------
        monomers : dict[str, Callable[[], Monomer]]
            Dictionary mapping monomer names to factory functions that create Monomer instances.
            Each template defines the "input arm" geometry for the proportional divider.
        """
        self.monomers = monomers

    def build(
        self,
        path: np.ndarray,
        seq: list[str],
    ) -> Atomistic:
        """
        Build a polymer chain along a given path with specified sequence.
        
        This method implements the digital equivalent of a proportional divider:
        - Path points act as "pivot points" (支点)
        - Monomer templates are the "input arm" (输入端)
        - Placed monomers are the "output arm" (输出端)
        - The transformation preserves proportions and orientations
        
        Parameters
        ----------
        path : np.ndarray
            Array of shape (N, 3) defining the backbone path (proportional divider track)
        seq : list[str]
            Sequence of monomer names to place along the path

        Returns
        -------
        Atomistic
            The constructed polymer structure
        """
        placed_monomers = []

        for i, monomer_name in enumerate(seq):
            monomer_template = self.monomers[monomer_name]()  # Get the original monomer template
            monomer = monomer_template()  # Clone it
            placed_monomer = self._apply_proportional_transformation(
                monomer, monomer_template.anchors, path, i, len(seq)
            )
            
            # Apply deletion rules for chain connectivity
            if i > 0:
                placed_monomer = self._apply_deletion_rules(placed_monomer)
            
            placed_monomers.append(placed_monomer)

        # Assemble final structure by concatenating Atomistic objects
        return self._assemble_polymer(placed_monomers)

    def _apply_proportional_transformation(
        self, 
        monomer: Monomer, 
        anchors: list[AnchorRule],
        path: np.ndarray, 
        position_index: int,
        total_count: int
    ) -> Monomer:
        """
        Apply proportional divider transformation to place a monomer.
        
        This is the core of the proportional divider algorithm:
        1. Determine pivot point (支点)
        2. Calculate native orientation (输入臂方向)
        3. Calculate target orientation (输出臂方向)
        4. Apply rotation transformation (角度传递)
        5. Apply translation to pivot (位置对齐)
        
        Parameters
        ----------
        monomer : Monomer
            The monomer template to transform
        path : np.ndarray
            The path defining pivot points
        position_index : int
            Current position along the path
        total_count : int
            Total number of monomers in sequence
            
        Returns
        -------
        Monomer
            Transformed monomer structure
        """
        if not anchors:
            raise ValueError(f"Monomer has no anchor rules defined")
        
        anchor_rule = anchors[0]
        
        # Get current positions
        positions = monomer.positions.copy()
        
        # Step 1: Determine pivot point (比例卡尺的支点)
        pivot_point = self._get_pivot_point(path, position_index)
        
        # Step 2: Calculate native direction (输入臂的天然方向)
        native_direction = self._get_native_direction(positions, anchor_rule)
        
        # Step 3: Calculate target direction (输出臂的目标方向)
        target_direction = self._get_target_direction(path, position_index)
        
        # Step 4: Apply proportional divider transformation
        # 4a) Rotation transformation (角度对齐)
        rotation_matrix = rotation_matrix_from_vectors(native_direction, target_direction)
        positions = positions.dot(rotation_matrix.T)
        
        # 4b) Translation to pivot point (位置对齐到支点)
        anchor_position = positions[anchor_rule.init]
        translation_vector = pivot_point - anchor_position
        positions += translation_vector
        
        # Update monomer positions
        for i, atom in enumerate(monomer.atoms):
            atom['xyz'] = positions[i]
        
        # Store anchor rule for deletion processing
        monomer['anchor_rule'] = anchor_rule
        
        return monomer
    
    def _get_pivot_point(self, path: np.ndarray, index: int) -> np.ndarray:
        """Get the pivot point for proportional divider transformation."""
        # Handle case where sequence is longer than path
        pivot_index = min(index, len(path) - 1)
        return path[pivot_index]
    
    def _get_native_direction(self, positions: np.ndarray, anchor_rule: AnchorRule) -> np.ndarray:
        """Calculate the native direction vector of the monomer template."""
        direction_vector = positions[anchor_rule.end] - positions[anchor_rule.init]
        
        # Normalize with safety check
        norm = np.linalg.norm(direction_vector)
        if norm > 1e-10:
            return direction_vector / norm
        else:
            # Default direction if atoms are coincident
            return np.array([1.0, 0.0, 0.0])
    
    def _get_target_direction(self, path: np.ndarray, index: int) -> np.ndarray:
        """Calculate the target direction from the path geometry."""
        if index < len(path) - 1:
            # Normal case: use next path segment
            direction_vector = path[index + 1] - path[index]
        else:
            # Last monomer: reuse previous direction
            if index > 0 and len(path) >= 2:
                prev_index = min(index - 1, len(path) - 2)
                curr_index = min(index, len(path) - 1)
                direction_vector = path[curr_index] - path[prev_index]
            else:
                # Single point case: use default direction
                direction_vector = np.array([1.0, 0.0, 0.0])
        
        # Normalize with safety check
        norm = np.linalg.norm(direction_vector)
        if norm > 1e-10:
            return direction_vector / norm
        else:
            return np.array([1.0, 0.0, 0.0])
    
    def _apply_deletion_rules(self, placed_monomer: Monomer) -> Monomer:
        """Apply deletion rules to avoid atomic overlaps in chain construction."""
        anchor_rule = placed_monomer.get('anchor_rule')
        
        if anchor_rule and anchor_rule.deletes:
            # Get atoms list before deletion (in reverse order to avoid index shifting)
            atoms_list = list(placed_monomer.atoms)
            
            # Remove atoms in reverse order to avoid index issues
            for atom_index in sorted(anchor_rule.deletes, reverse=True):
                if atom_index < len(atoms_list):
                    placed_monomer.atoms.remove(atoms_list[atom_index])
        
        return placed_monomer
    
    def _assemble_polymer(self, placed_monomers: list[Monomer]) -> Atomistic:
        """Assemble the final polymer structure from all placed monomers."""
        return Atomistic.concat(placed_monomers)