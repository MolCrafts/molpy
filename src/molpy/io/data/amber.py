from pathlib import Path

import numpy as np
import molpy as mp
from molpy.core.frame import _dict_to_dataset
from .base import DataReader

class AmberInpcrdReader(DataReader):
    """
    AMBER coordinate (inpcrd) file reader.
    
    Format specification:
    - Line 1: Title line
    - Line 2: Number of atoms, optionally time
    - Following lines: Coordinates (x, y, z) for each atom (12.7 format, 6 per line)
    - Optional: Velocities if present
    - Last line: Box dimensions if periodic
    """

    def __init__(self, file: str | Path):
        super().__init__(Path(file))
        self._file = Path(file)

    @staticmethod
    def sanitizer(line: str) -> str:
        """Clean up line by stripping whitespace."""
        return line.rstrip('\n\r')
    
    def read(self, frame: mp.Frame) -> mp.Frame:
        """Read AMBER coordinate file."""
        lines = []
        with open(self._file, 'r') as f:
            lines = [self.sanitizer(line) for line in f.readlines()]
        
        if len(lines) < 2:
            raise ValueError("Invalid AMBER coordinate file: too few lines")
        
        # Parse header
        title = lines[0]
        header_parts = lines[1].split()
        
        if len(header_parts) < 1:
            raise ValueError("Invalid header line")
        
        natoms = int(header_parts[0])
        time = float(header_parts[1]) if len(header_parts) > 1 else None
        
        # Parse coordinates
        coord_lines = []
        line_idx = 2
        
        # Calculate expected number of coordinate lines
        # 6 coordinates per line (12.7 format), 3 coordinates per atom
        coords_needed = natoms * 3
        lines_needed = (coords_needed + 5) // 6  # Round up
        
        if line_idx + lines_needed > len(lines):
            raise ValueError(f"Not enough coordinate lines. Expected {lines_needed}, got {len(lines) - line_idx}")
        
        for i in range(lines_needed):
            coord_lines.append(lines[line_idx + i])
        
        # Parse coordinates using Fortran format (12.8 or 12.7)
        coords = []
        for line in coord_lines:
            # Each coordinate is 12 characters wide
            for i in range(0, len(line), 12):
                if i + 12 <= len(line):
                    coord_str = line[i:i+12].strip()
                    if coord_str:
                        try:
                            coords.append(float(coord_str))
                        except ValueError:
                            # Skip malformed coordinates
                            pass
        
        if len(coords) < natoms * 3:
            raise ValueError(f"Not enough coordinates. Expected {natoms * 3}, got {len(coords)}")
        
        # Reshape coordinates
        positions = np.array(coords[:natoms * 3]).reshape(natoms, 3)
        
        # Check for velocities
        line_idx += lines_needed
        velocities = None
        
        if line_idx < len(lines) - 1:  # Allow for box line at end
            # Check if next lines contain velocities
            remaining_lines = len(lines) - line_idx
            if remaining_lines >= lines_needed:  # Same number of lines for velocities
                vel_coords = []
                for i in range(lines_needed):
                    line = lines[line_idx + i]
                    for j in range(0, len(line), 12):
                        if j + 12 <= len(line):
                            coord_str = line[j:j+12].strip()
                            if coord_str:
                                vel_coords.append(float(coord_str))
                
                if len(vel_coords) >= natoms * 3:
                    velocities = np.array(vel_coords[:natoms * 3]).reshape(natoms, 3)
                    line_idx += lines_needed
        
        # Check for box information
        box = None
        if line_idx < len(lines):
            box_line = lines[line_idx].strip()
            if box_line:
                box_parts = box_line.split()
                if len(box_parts) >= 3:
                    # Basic orthogonal box
                    a, b, c = float(box_parts[0]), float(box_parts[1]), float(box_parts[2])
                    box_matrix = np.diag([a, b, c])
                    box = mp.Box(matrix=box_matrix)
        
        # Create atom data
        atoms_data = {
            'id': np.arange(1, natoms + 1, dtype=int),
            'name': np.array([f'ATM{i+1}' for i in range(natoms)], dtype='U10'),
            'xyz': positions,
            'atomic_number': np.ones(natoms, dtype=int),  # Default to 1 (unknown)
        }
        
        if velocities is not None:
            atoms_data['vx'] = velocities[:, 0]
            atoms_data['vy'] = velocities[:, 1]
            atoms_data['vz'] = velocities[:, 2]
        
        # Set frame data
        frame["atoms"] = _dict_to_dataset(atoms_data)
        if box is not None:
            frame.box = box
        if time is not None:
            frame.timestep = int(time)
        
        frame['props'] = frame.get('props', {})
        frame['props']['name'] = title
        
        return frame


class AmberRst7Reader(AmberInpcrdReader):
    """Alias for AmberInpcrdReader for restart files."""
    pass


class AmberRst7Writer:
    """AMBER restart file writer."""
    
    def __init__(self, file: str | Path):
        self.path = Path(file)
    
    def write(self, frame: mp.Frame) -> None:
        """Write frame to AMBER restart file."""
        if "atoms" not in frame:
            raise ValueError("Frame must contain atoms data")
        
        atoms = frame["atoms"]
        
        # Get atom count
        sizes = atoms.sizes
        main_dim = next(iter(sizes.keys()))
        natoms = sizes[main_dim]
        
        # Get coordinates
        if "xyz" not in atoms.data_vars:
            raise ValueError("Atoms must have xyz coordinates")
        
        xyz = atoms["xyz"].values
        if xyz.shape != (natoms, 3):
            raise ValueError(f"Expected xyz shape ({natoms}, 3), got {xyz.shape}")
        
        # Check for velocities
        has_velocities = all(v in atoms.data_vars for v in ['vx', 'vy', 'vz'])
        
        with open(self.path, 'w') as f:
            # Write title
            title = frame.get('props', {}).get('name', 'Generated by molpy')
            f.write(f"{title}\n")
            
            # Write header
            if hasattr(frame, 'timestep') and frame.timestep is not None:
                f.write(f"{natoms:5d}{frame.timestep:15.7E}\n")
            else:
                f.write(f"{natoms:5d}\n")
            
            # Write coordinates (12.7 format, 6 per line)
            coords = xyz.flatten()
            for i in range(0, len(coords), 6):
                line_coords = coords[i:i+6]
                line = ""
                for coord in line_coords:
                    line += f"{coord:12.7E}"
                f.write(line + "\n")
            
            # Write velocities if present
            if has_velocities:
                vx = atoms["vx"].values
                vy = atoms["vy"].values
                vz = atoms["vz"].values
                velocities = np.column_stack([vx, vy, vz]).flatten()
                
                for i in range(0, len(velocities), 6):
                    line_vels = velocities[i:i+6]
                    line = ""
                    for vel in line_vels:
                        line += f"{vel:12.7E}"
                    f.write(line + "\n")
            
            # Write box if present
            if hasattr(frame, 'box') and frame.box is not None:
                matrix = frame.box.matrix
                # Extract diagonal elements for orthogonal box
                a, b, c = matrix[0, 0], matrix[1, 1], matrix[2, 2]
                f.write(f"{a:12.7E}{b:12.7E}{c:12.7E}\n")