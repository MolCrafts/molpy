"""
XSF (XCrySDen Structure File) format reader and writer.

XSF is a format for crystal structure visualization, supporting both periodic and non-periodic structures.
It can contain atomic coordinates, unit cell parameters, and other structural information.
"""

import numpy as np
from pathlib import Path
from typing import Any

import molpy as mp
from molpy.core import Element, Frame, Box, Block

from .base import DataReader, DataWriter


class XsfReader(DataReader):
    """
    Parse an XSF file into a Frame.
    
    XSF format supports both crystal structures (with unit cell) and molecular structures.
    The format can contain:
    - CRYSTAL or MOLECULE keyword
    - PRIMVEC or CONVVEC for unit cell vectors
    - PRIMCOORD for atomic coordinates
    - Optional comment lines starting with #
    """
    
    def __init__(self, file: str | Path):
        super().__init__(Path(file))
        self._file = Path(file)
    
    def read(self) -> mp.FrameSystem:
        """
        Read XSF file and return FrameSystem.
        
        Returns
        -------
        FrameSystem
            FrameSystem with:
            - frame containing atomic data
            - box: unit cell for CRYSTAL, Free Box for MOLECULE
        """
        frame = mp.Frame()
        
        lines = self.read_lines()
        lines = [line.strip() for line in lines if line.strip()]
        
        # Remove comment lines
        lines = [line for line in lines if not line.startswith('#')]
        
        if not lines:
            raise ValueError("Empty XSF file")
        
        # Parse structure type
        structure_type = None
        primvec_matrix = None
        convvec_matrix = None
        atoms_data = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if line.upper() == "CRYSTAL":
                structure_type = "CRYSTAL"
                i += 1
                continue
            elif line.upper() == "MOLECULE":
                structure_type = "MOLECULE"
                i += 1
                continue
            elif line.upper() == "PRIMVEC":
                # Read 3 lines of primitive vectors
                primvec_matrix = self._parse_vectors(lines[i+1:i+4])
                i += 4
                continue
            elif line.upper() == "CONVVEC":
                # Read 3 lines of conventional vectors
                convvec_matrix = self._parse_vectors(lines[i+1:i+4])
                i += 4
                continue
            elif line.upper() == "PRIMCOORD":
                # Read atomic coordinates
                if i + 1 >= len(lines):
                    raise ValueError("PRIMCOORD section incomplete")
                
                # Next line should contain number of atoms and multiplicity
                coord_info = lines[i+1].split()
                if len(coord_info) < 2:
                    raise ValueError("Invalid PRIMCOORD header")
                
                n_atoms = int(coord_info[0])
                multiplicity = int(coord_info[1])
                
                # Read atomic coordinates
                atoms_data = self._parse_atoms(lines[i+2:i+2+n_atoms])
                i += 2 + n_atoms
                continue
            else:
                i += 1
        
        # Create atoms block
        if atoms_data:
            atoms_dict = {
                'atomic_number': [atom['atomic_number'] for atom in atoms_data],
                'xyz': [atom['xyz'] for atom in atoms_data],
                'element': [atom['element'] for atom in atoms_data],
                'x': [atom['xyz'][0] for atom in atoms_data],
                'y': [atom['xyz'][1] for atom in atoms_data],
                'z': [atom['xyz'][2] for atom in atoms_data]
            }
            frame["atoms"] = Block(atoms_dict)
        
        # Set up box and create system
        if structure_type == "CRYSTAL":
            if primvec_matrix is not None:
                box = Box(primvec_matrix)
            elif convvec_matrix is not None:
                box = Box(convvec_matrix)
            else:
                # Default box if vectors not specified
                box = Box(np.eye(3))
        else:
            # For molecular structures, use free box
            box = Box()  # Free box for non-periodic molecules
        
        system = mp.FrameSystem(frame=frame, box=box)
        return system
    
    def _parse_vectors(self, lines: list[str]) -> np.ndarray:
        """Parse 3 lines of box vectors."""
        if len(lines) < 3:
            raise ValueError("Incomplete vector specification")
        
        matrix = np.zeros((3, 3))
        for i, line in enumerate(lines[:3]):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid vector line: {line}")
            
            try:
                matrix[i] = [float(parts[j]) for j in range(3)]
            except ValueError as e:
                raise ValueError(f"Invalid vector coordinates: {line}") from e
        
        return matrix
    
    def _parse_atoms(self, lines: list[str]) -> list[dict[str, Any]]:
        """Parse atomic coordinate lines."""
        atoms = []
        
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid atom line: {line}")
            
            try:
                atomic_number = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                
                # Get element symbol from atomic number
                element = Element(atomic_number).symbol
                
                atoms.append({
                    'atomic_number': atomic_number,
                    'xyz': np.array([x, y, z]),
                    'element': element
                })
            except ValueError as e:
                raise ValueError(f"Invalid atom data: {line}") from e
        
        return atoms


class XsfWriter(DataWriter):
    """
    Write Frame to XSF format.
    
    Features:
    - Supports both CRYSTAL and MOLECULE structures
    - Writes PRIMVEC for unit cell vectors
    - Writes PRIMCOORD for atomic coordinates
    - Automatically determines structure type based on presence of box
    """
    
    def __init__(self, file: str | Path):
        super().__init__(Path(file))
        self._file = Path(file)
    
    def write(self, system: mp.FrameSystem) -> None:
        """
        Write FrameSystem to XSF file.
        
        Parameters
        ----------
        system : FrameSystem
            FrameSystem containing atomic data and optional box information
        """
        frame = system._wrapped
        box = system.box
        
        with open(self._file, 'w') as f:
            # Write header comment
            f.write("# XSF file generated by molpy\n")
            
            # Determine structure type
            has_box = (box is not None and 
                      hasattr(box, 'matrix') and 
                      box.matrix is not None and 
                      not np.allclose(box.matrix, 0))
            
            if has_box:
                f.write("CRYSTAL\n")
                
                # Write primitive vectors
                f.write("PRIMVEC\n")
                matrix = box.matrix
                for i in range(3):
                    f.write(f"    {matrix[i, 0]:12.8f}    {matrix[i, 1]:12.8f}    {matrix[i, 2]:12.8f}\n")
                
                # Write conventional vectors (same as primitive for now)
                f.write("CONVVEC\n")
                for i in range(3):
                    f.write(f"    {matrix[i, 0]:12.8f}    {matrix[i, 1]:12.8f}    {matrix[i, 2]:12.8f}\n")
            else:
                f.write("MOLECULE\n")
            
            # Write atomic coordinates
            if "atoms" in frame:
                atoms = frame["atoms"]
                atomic_numbers = atoms["atomic_number"]
                xyz = atoms["xyz"]
                n_atoms = len(atomic_numbers)
                
                f.write("PRIMCOORD\n")
                f.write(f"       {n_atoms} 1\n")
                
                for i in range(n_atoms):
                    an = atomic_numbers[i]
                    x, y, z = xyz[i]
                    f.write(f"{an:2d}    {x:12.8f}    {y:12.8f}    {z:12.8f}\n")
            else:
                # Empty structure
                f.write("PRIMCOORD\n")
                f.write("        0 1\n")
