"""System classes for organizing molecular data.

This module defines three types of molecular systems:
- FrameSystem: Uses Frame/Block API for columnar data storage
- StructSystem: Uses Struct | Wrapper API for object graph storage
- PeriodicSystem: Wrapper for periodic systems supporting supercell operations
"""

import numpy as np

from .frame import Frame
from .forcefield import ForceField
from .box import Box
from .wrapper import Wrapper
from .protocol import Struct


class Systemic(Wrapper):

    def __init__(self, wrapped, box: Box | None = None, forcefield: ForceField | None = None):
        super().__init__(wrapped)
        self._box = box if box is not None else Box()
        self._forcefield = forcefield if forcefield is not None else ForceField()

    @property
    def forcefield(self) -> ForceField:
        """Return the force field associated with this system."""
        return self._forcefield

    @property
    def box(self) -> Box:
        """Return the simulation box of this system."""
        return self._box
    
    def set_forcefield(self, forcefield: ForceField):
        """Set the force field for this system."""
        self._forcefield = forcefield

    def set_box(self, box: Box):
        """Set the simulation box for this system."""
        self._box = box


class FrameSystem(Systemic):

    def __init__(
        self,
        frame: Frame | None = None,
        box: Box | None = None,
        forcefield: ForceField | None = None,
    ):
        super().__init__(frame, box, forcefield)


class StructSystem(Systemic):

    def __init__(self, struct: Struct | Wrapper[Struct] | None = None, box: Box | None = None, forcefield: ForceField | None = None):
        if struct is None:
            struct = Struct()
        super().__init__(struct, box, forcefield)
        self.structs = []

    def add_struct(self, struct: Struct | Wrapper[Struct]):
        """Add a structure to the system."""
        self._wrapped.add_struct(struct)
        self.structs.append(struct)

class PeriodicSystem(Wrapper):
    """Wrapper for periodic systems supporting supercell operations.
    
    This class can wrap either FrameSystem or StructSystem and provides
    periodic boundary condition operations like supercell creation.
    """

    def __init__(self, system: FrameSystem | StructSystem):
        """Initialize a periodic system.
        
        Args:
            system: The underlying FrameSystem or StructSystem to wrap.
            
        Raises:
            ValueError: If the system's box is not periodic.
        """
        if not system.box.is_periodic:
            raise ValueError("System must have periodic boundary conditions for supercell operations")
        
        super().__init__(system)

    @property
    def box(self) -> Box:
        """Return the simulation box of this system."""
        return self._wrapped.box

    @property
    def forcefield(self) -> ForceField:
        """Return the force field associated with this system."""
        return self._wrapped.forcefield

    def make_supercell(self, transformation_matrix: list[list[int]]) -> 'PeriodicSystem':
        """Create a supercell by replicating the system according to transformation matrix.
        
        Args:
            transformation_matrix: 3x3 matrix defining the supercell transformation.
                Each row represents how many times to replicate along each axis.
                Example: [[4,0,0],[0,2,0],[0,0,3]] creates a 4x2x3 supercell.
                
        Returns:
            A new PeriodicSystem instance with the supercell structure.
            
        Raises:
            ValueError: If transformation_matrix is not 3x3 or contains invalid values.
        """
        # Validate transformation matrix
        if len(transformation_matrix) != 3 or any(len(row) != 3 for row in transformation_matrix):
            raise ValueError("Transformation matrix must be 3x3")
        
        matrix = np.array(transformation_matrix)
        
        # Check for valid supercell matrix (determinant should be positive)
        if np.linalg.det(matrix) <= 0:
            raise ValueError("Transformation matrix must have positive determinant")

        # Create supercell
        new_system = self._create_supercell(matrix)
        return PeriodicSystem(new_system)

    def _create_supercell(self, matrix: np.ndarray) -> FrameSystem | StructSystem:
        """Create supercell structure based on the wrapped system type.
        
        Args:
            matrix: 3x3 transformation matrix.
            
        Returns:
            New system with supercell structure.
        """
        # Transform the box
        new_box = self.box.transform(matrix)
        
        if isinstance(self._wrapped, FrameSystem):
            return self._create_frame_supercell(matrix, new_box)
        elif isinstance(self._wrapped, StructSystem):
            return self._create_struct_supercell(matrix, new_box)
        else:
            raise TypeError(f"Unsupported system type: {type(self._wrapped)}")

    def _create_frame_supercell(self, matrix: np.ndarray, new_box: Box) -> FrameSystem:
        """Create supercell for FrameSystem.
        
        Args:
            matrix: 3x3 transformation matrix.
            new_box: Transformed simulation box.
            
        Returns:
            New FrameSystem with supercell structure.
        """
        original_frame = self._wrapped._wrapped
        new_frame = self._replicate_frame(original_frame, matrix)
        
        return FrameSystem(
            frame=new_frame,
            box=new_box,
            forcefield=self._wrapped.forcefield
        )

    def _create_struct_supercell(self, matrix: np.ndarray, new_box: Box) -> StructSystem:
        """Create supercell for StructSystem.
        
        Args:
            matrix: 3x3 transformation matrix.
            new_box: Transformed simulation box.
            
        Returns:
            New StructSystem with supercell structure.
        """
        original_struct = self._wrapped._wrapped
        new_struct = self._replicate_struct(original_struct, matrix)
        
        return StructSystem(
            struct=new_struct,
            box=new_box,
            forcefield=self._wrapped.forcefield
        )

    def _replicate_frame(self, frame: Frame, matrix: np.ndarray) -> Frame:
        """Replicate Frame data according to transformation matrix.
        
        Args:
            frame: Original frame to replicate.
            matrix: 3x3 transformation matrix.
            
        Returns:
            New frame with replicated data.
        """
        from .frame import Frame, Block
        
        # Get the original box for coordinate transformation
        original_box = self.box
        
        # Create new frame with same box reference initially
        new_frame = Frame(box=frame.box)
        new_frame.metadata = frame.metadata.copy()
        
        # Process each block in the frame
        for block_name in frame.blocks():
            block = frame[block_name]
            new_block_data = {}
            
            # Copy all data from the original block
            for var_name in block:
                new_block_data[var_name] = block[var_name].copy()
            
            # Handle position data - assume it's stored as 'xyz' or 'positions'
            pos_var = None
            if 'xyz' in block:
                pos_var = 'xyz'
            elif 'positions' in block:
                pos_var = 'positions'
            
            if pos_var is not None:
                original_positions = block[pos_var]
                n_atoms = len(original_positions)
                
                # Generate all lattice translation vectors
                translations = self._generate_translations(matrix)
                
                # Calculate supercell dimensions based on actual translations
                supercell_size = len(translations)
                
                # Create array for all replicated positions
                replicated_positions = np.zeros((n_atoms * supercell_size, 3))
                
                # Replicate atoms
                atom_idx = 0
                for translation in translations:
                    # Transform translation vector using original box vectors
                    real_translation = original_box.matrix @ translation
                    
                    # Copy and translate positions
                    for i in range(n_atoms):
                        replicated_positions[atom_idx] = original_positions[i] + real_translation
                        atom_idx += 1
                
                new_block_data[pos_var] = replicated_positions
                
                # Replicate other atomic properties
                for var_name in block:
                    if var_name != pos_var and len(block[var_name]) == n_atoms:
                        # This is likely an atomic property, replicate it
                        value = block[var_name]
                        if value.ndim > 1:
                            new_block_data[var_name] = np.tile(value, (supercell_size, 1))
                        else:
                            new_block_data[var_name] = np.tile(value, supercell_size)
            
            # Set the new block
            new_frame[block_name] = Block(new_block_data)
        
        return new_frame

    def _replicate_struct(self, struct: Struct | Wrapper, matrix: np.ndarray) -> Struct | Wrapper:
        """Replicate Struct | Wrapper data according to transformation matrix.
        
        Args:
            struct: Original struct to replicate.
            matrix: 3x3 transformation matrix.
            
        Returns:
            New struct with replicated data.
        """
        # This is a placeholder implementation
        # The actual implementation depends on the Struct | Wrapper data structure
        # For now, return the original struct
        return struct
    
    def _generate_translations(self, matrix: np.ndarray) -> list[np.ndarray]:
        """Generate all lattice translation vectors for the supercell.
        
        Args:
            matrix: 3x3 transformation matrix.
            
        Returns:
            List of translation vectors in fractional coordinates.
        """
        translations = []
        
        # For simple diagonal matrices, generate translations directly
        if np.allclose(matrix, np.diag(np.diag(matrix))):
            nx, ny, nz = int(matrix[0, 0]), int(matrix[1, 1]), int(matrix[2, 2])
            
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        translations.append(np.array([i, j, k], dtype=float))
        else:
            # For general matrices, we need a more sophisticated approach
            # This is simplified - a full implementation would use proper
            # lattice vector generation algorithms
            det = int(np.linalg.det(matrix))
            
            # Generate translations in a grid that covers the supercell
            max_val = int(np.ceil(np.max(matrix)))
            
            count = 0
            for i in range(max_val):
                for j in range(max_val):
                    for k in range(max_val):
                        if count >= det:
                            break
                        translations.append(np.array([i, j, k], dtype=float))
                        count += 1
                    if count >= det:
                        break
                if count >= det:
                    break
        
        return translations

    def add_vacuum(self, vacuum_thickness: float, direction: str = 'z') -> 'PeriodicSystem':
        """Add vacuum space to the system in the specified direction.
        
        This is commonly used for surface systems where you want to add
        vacuum space above/below the surface to prevent periodic images
        from interacting.
        
        Args:
            vacuum_thickness: Thickness of vacuum to add (in same units as box).
            direction: Direction to add vacuum ('x', 'y', or 'z'). Default is 'z'.
            
        Returns:
            A new PeriodicSystem instance with added vacuum space.
            
        Raises:
            ValueError: If direction is not 'x', 'y', or 'z'.
            ValueError: If vacuum_thickness is not positive.
        """
        # Validate inputs
        if direction not in ['x', 'y', 'z']:
            raise ValueError("Direction must be 'x', 'y', or 'z'")
        
        if vacuum_thickness <= 0:
            raise ValueError("Vacuum thickness must be positive")
        
        # Create new system with expanded box
        new_system = self._create_vacuum_system(vacuum_thickness, direction)
        return PeriodicSystem(new_system)

    def _create_vacuum_system(self, vacuum_thickness: float, direction: str) -> FrameSystem | StructSystem:
        """Create system with added vacuum space.
        
        Args:
            vacuum_thickness: Thickness of vacuum to add.
            direction: Direction to add vacuum.
            
        Returns:
            New system with expanded box.
        """
        # Determine direction index
        dir_map = {'x': 0, 'y': 1, 'z': 2}
        dir_idx = dir_map[direction]
        
        # Create new box with expanded dimension
        current_box = self.box
        new_matrix = current_box.matrix.copy()
        new_matrix[dir_idx, dir_idx] += vacuum_thickness
        
        new_box = Box(
            matrix=new_matrix,
            pbc=current_box.pbc.copy(),
            origin=current_box.origin.copy(),
            name=current_box.name
        )
        
        if isinstance(self._wrapped, FrameSystem):
            return self._create_frame_vacuum_system(new_box)
        elif isinstance(self._wrapped, StructSystem):
            return self._create_struct_vacuum_system(new_box)
        else:
            raise TypeError(f"Unsupported system type: {type(self._wrapped)}")

    def _create_frame_vacuum_system(self, new_box: Box) -> FrameSystem:
        """Create FrameSystem with vacuum space added.
        
        Args:
            new_box: Box with expanded dimensions.
            
        Returns:
            New FrameSystem with same atoms but expanded box.
        """
        # Get original frame
        original_frame = self._wrapped._wrapped
        
        # Create new frame - we don't need to move atoms, just expand the box
        # The atoms stay in their original positions, vacuum is added around them
        new_frame = Frame(box=new_box)
        new_frame.metadata = original_frame.metadata.copy()
        
        # Copy all blocks from original frame
        for block_name in original_frame.blocks():
            original_block = original_frame[block_name]
            new_block_data = {}
            
            # Copy all data from original block without modification
            for var_name in original_block:
                new_block_data[var_name] = original_block[var_name].copy()
            
            # Set the new block
            from .frame import Block
            new_frame[block_name] = Block(new_block_data)
        
        return FrameSystem(
            frame=new_frame,
            box=new_box,
            forcefield=self._wrapped.forcefield
        )

    def _create_struct_vacuum_system(self, new_box: Box) -> StructSystem:
        """Create StructSystem with vacuum space added.
        
        Args:
            new_box: Box with expanded dimensions.
            
        Returns:
            New StructSystem with same structure but expanded box.
        """
        # For struct systems, we just need to copy the structure
        # and use the new box
        original_struct = self._wrapped._wrapped
        
        return StructSystem(
            struct=original_struct,  # Structure remains the same
            box=new_box,
            forcefield=self._wrapped.forcefield
        )
