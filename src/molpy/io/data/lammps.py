"""
Modern LAMMPS data and molecule template file I/O.

This module provides clean, efficient, and maintainable readers and writers
for LAMMPS data files and molecule templates, fully compatible with xarray-based Frame structure.
All operations use xarray.Dataset directly with unified string-based type handling.
No backward compatibility code - all type fields are handled uniformly as strings.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np
import xarray as xr

import molpy as mp
from .base import DataReader, DataWriter


class LammpsDataReader(DataReader):
    """Modern LAMMPS data file reader using xarray exclusively."""
    
    def __init__(self, path: Union[str, Path], atom_style: str = "full"):
        super().__init__(Path(path))  # Convert to Path explicitly
        self.atom_style = atom_style
    
    def read(self, frame: Optional[mp.Frame] = None) -> mp.Frame:
        """Read LAMMPS data file into a Frame."""
        if frame is None:
            frame = mp.Frame()
        
        lines = self._read_and_clean_lines()
        sections = self._parse_sections(lines)
        
        # Parse header information
        counts, box_bounds = self._parse_header(sections.get('header', []))
        
        # Set up box
        if box_bounds:
            frame.box = self._create_box(box_bounds)
        
        # Parse data sections
        if 'Masses' in sections:
            masses = self._parse_masses(sections['Masses'])
        else:
            masses = {}
        
        if 'Atoms' in sections:
            self._parse_atoms(sections['Atoms'], frame, masses)
        
        if 'Bonds' in sections and counts.get('bonds', 0) > 0:
            self._parse_bonds(sections['Bonds'], frame)
        
        if 'Angles' in sections and counts.get('angles', 0) > 0:
            self._parse_angles(sections['Angles'], frame)
        
        if 'Dihedrals' in sections and counts.get('dihedrals', 0) > 0:
            self._parse_dihedrals(sections['Dihedrals'], frame)
        
        if 'Impropers' in sections and counts.get('impropers', 0) > 0:
            self._parse_impropers(sections['Impropers'], frame)
        
        # Store metadata
        frame._meta.update({
            'format': 'lammps_data',
            'atom_style': self.atom_style,
            'counts': counts,
            'source_file': str(self._path)
        })
        
        return frame
    
    def _read_and_clean_lines(self) -> List[str]:
        """Read file and return cleaned lines."""
        with open(self._path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        return [line for line in lines if line and not line.startswith('#')]
    
    def _parse_sections(self, lines: List[str]) -> Dict[str, List[str]]:
        """Parse file into sections."""
        sections = {'header': []}
        current_section = 'header'
        
        for line in lines:
            # Check if line starts with section keywords (case insensitive)
            line_lower = line.lower()
            if (line_lower.startswith('atoms') or 
                line_lower.startswith('masses') or 
                line_lower.startswith('bonds') or 
                line_lower.startswith('angles') or 
                line_lower.startswith('dihedrals') or 
                line_lower.startswith('impropers')):
                # Extract the section name (first word)
                section_name = line.split()[0].capitalize()
                current_section = section_name
                sections[current_section] = []
            elif line.lower().endswith('types') or line.lower().endswith('atoms') or line.lower().endswith('bonds'):
                # Count lines go to header
                sections['header'].append(line)
            elif line.lower().startswith('pair coeffs') or line.lower().startswith('bond coeffs'):
                # Skip coefficient sections for now
                current_section = 'coeffs'
                sections[current_section] = []
            elif 'xlo xhi' in line.lower() or 'ylo' in line.lower() and 'yhi' in line.lower() or 'zlo zhi' in line.lower():
                # Box bounds go to header
                sections['header'].append(line)
            else:
                if current_section not in sections:
                    sections[current_section] = []
                sections[current_section].append(line)
        
        return sections
    
    def _parse_header(self, header_lines: List[str]) -> Tuple[Dict[str, int], Optional[Dict[str, Tuple[float, float]]]]:
        """Parse header information."""
        counts = {}
        box_bounds = {}
        
        for line in header_lines:
            parts = line.split()
            if len(parts) < 2:
                continue
                
            try:
                if 'atoms' in line.lower() and not line.lower().startswith('atoms'):
                    counts['atoms'] = int(parts[0])
                elif 'bonds' in line.lower() and not line.lower().startswith('bonds'):
                    counts['bonds'] = int(parts[0])
                elif 'angles' in line.lower() and not line.lower().startswith('angles'):
                    counts['angles'] = int(parts[0])
                elif 'dihedrals' in line.lower() and not line.lower().startswith('dihedrals'):
                    counts['dihedrals'] = int(parts[0])
                elif 'impropers' in line.lower() and not line.lower().startswith('impropers'):
                    counts['impropers'] = int(parts[0])
                elif 'atom types' in line.lower():
                    counts['atom_types'] = int(parts[0])
                elif 'bond types' in line.lower():
                    counts['bond_types'] = int(parts[0])
                elif 'angle types' in line.lower():
                    counts['angle_types'] = int(parts[0])
                elif 'dihedral types' in line.lower():
                    counts['dihedral_types'] = int(parts[0])
                elif 'improper types' in line.lower():
                    counts['improper_types'] = int(parts[0])
                elif 'xlo xhi' in line.lower():
                    box_bounds['x'] = (float(parts[0]), float(parts[1]))
                elif 'ylo' in line.lower() and 'yhi' in line.lower():
                    # Handle "ylo yhi" with flexible spacing
                    ylo_idx = next(i for i, part in enumerate(parts) if 'ylo' in part.lower())
                    box_bounds['y'] = (float(parts[0]), float(parts[1]))
                elif 'zlo zhi' in line.lower():
                    box_bounds['z'] = (float(parts[0]), float(parts[1]))
            except (ValueError, IndexError):
                # Skip lines that can't be parsed
                continue
        
        return counts, box_bounds if box_bounds else None
    
    def _create_box(self, box_bounds: Dict[str, Tuple[float, float]]) -> mp.Box:
        """Create Box from bounds."""
        # Ensure all three dimensions are present, use default if missing
        default_bounds = (0.0, 10.0)
        
        lengths = np.array([
            box_bounds.get('x', default_bounds)[1] - box_bounds.get('x', default_bounds)[0],
            box_bounds.get('y', default_bounds)[1] - box_bounds.get('y', default_bounds)[0],
            box_bounds.get('z', default_bounds)[1] - box_bounds.get('z', default_bounds)[0]
        ])
        origin = np.array([
            box_bounds.get('x', default_bounds)[0],
            box_bounds.get('y', default_bounds)[0],
            box_bounds.get('z', default_bounds)[0]
        ])
        return mp.Box(lengths, origin=origin)
    
    def _parse_masses(self, mass_lines: List[str]) -> Dict[str, float]:
        """Parse mass section. Returns mapping from type (as string) to mass."""
        masses = {}
        for line in mass_lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    # Always treat type as string for unified handling
                    type_str = parts[0]
                    try:
                        mass = float(parts[1])
                        masses[type_str] = mass
                    except ValueError:
                        # Skip lines where mass can't be parsed
                        continue
        return masses
    
    def _parse_atoms(self, atom_lines: List[str], frame: mp.Frame, masses: Dict[str, float]) -> None:
        """Parse atoms section and add to frame as xarray.Dataset."""
        if not atom_lines:
            return
        
        # Collect atom data
        atom_data = {
            'id': [],
            'type': [],
            'x': [],
            'y': [],
            'z': [],
            'mass': []
        }
        
        # Add fields based on atom style
        if self.atom_style == "full":
            atom_data.update({
                'mol': [],
                'q': []
            })
        elif self.atom_style == "charge":
            atom_data['q'] = []
        
        for line in atom_lines:
            if line.strip():
                parts = line.split()
                atom_id = int(parts[0])
                
                if self.atom_style == "full":
                    mol_id = int(parts[1])
                    atom_type_str = parts[2]  # Always treat as string
                    charge = float(parts[3])
                    x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                    
                    atom_data['mol'].append(mol_id)
                    atom_data['q'].append(charge)
                elif self.atom_style == "charge":
                    atom_type_str = parts[1]  # Always treat as string
                    charge = float(parts[2])
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    
                    atom_data['q'].append(charge)
                else:  # atomic
                    atom_type_str = parts[1]  # Always treat as string
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                
                atom_data['id'].append(atom_id)
                atom_data['type'].append(atom_type_str)
                atom_data['x'].append(x)
                atom_data['y'].append(y)
                atom_data['z'].append(z)
                # Get mass from mapping, default to 1.0 if not found
                atom_data['mass'].append(masses.get(atom_type_str, 1.0))
        
        # Convert to numpy arrays
        for key, values in atom_data.items():
            if key == 'type':
                # Keep type as string array
                atom_data[key] = np.array(values, dtype=str)
            else:
                atom_data[key] = np.array(values)
        
        # Create xyz coordinate array
        xyz = np.column_stack([atom_data['x'], atom_data['y'], atom_data['z']])
        atom_data['xyz'] = xyz
        
        # Remove individual x, y, z
        del atom_data['x'], atom_data['y'], atom_data['z']
        
        # Create xarray Dataset directly
        data_vars = {}
        n_atoms = len(atom_data['id'])
        
        for key, values in atom_data.items():
            if key == 'xyz':
                # 2D coordinate array
                data_vars[key] = (['atoms_id', 'spatial'], values)
            else:
                # 1D arrays
                data_vars[key] = (['atoms_id'], values)
        
        # Create coordinates
        coords = {
            'atoms_id': np.arange(n_atoms),
            'spatial': ['x', 'y', 'z']
        }
        
        atoms_dataset = xr.Dataset(data_vars, coords=coords)
        frame['atoms'] = atoms_dataset
    
    def _parse_bonds(self, bond_lines: List[str], frame: mp.Frame) -> None:
        """Parse bonds section and add to frame as xarray.Dataset."""
        if not bond_lines:
            return
        
        bond_data = {
            'id': [],
            'type': [],
            'atom1': [],
            'atom2': []
        }
        
        for line in bond_lines:
            if line.strip():
                parts = line.split()
                bond_id = int(parts[0])
                bond_type_str = parts[1]  # Always treat as string
                atom1 = int(parts[2])
                atom2 = int(parts[3])
                
                bond_data['id'].append(bond_id)
                bond_data['type'].append(bond_type_str)
                bond_data['atom1'].append(atom1)
                bond_data['atom2'].append(atom2)
        
        # Convert to numpy arrays and create Dataset
        data_vars = {}
        n_bonds = len(bond_data['id'])
        
        for key, values in bond_data.items():
            if key == 'type':
                # Keep type as string array
                data_vars[key] = (['bonds_id'], np.array(values, dtype=str))
            else:
                data_vars[key] = (['bonds_id'], np.array(values))
        
        coords = {'bonds_id': np.arange(n_bonds)}
        bonds_dataset = xr.Dataset(data_vars, coords=coords)
        frame['bonds'] = bonds_dataset
    
    def _parse_angles(self, angle_lines: List[str], frame: mp.Frame) -> None:
        """Parse angles section and add to frame as xarray.Dataset."""
        if not angle_lines:
            return
        
        angle_data = {
            'id': [],
            'type': [],
            'atom1': [],
            'atom2': [],
            'atom3': []
        }
        
        for line in angle_lines:
            if line.strip():
                parts = line.split()
                angle_id = int(parts[0])
                angle_type_str = parts[1]  # Always treat as string
                atom1 = int(parts[2])
                atom2 = int(parts[3])
                atom3 = int(parts[4])
                
                angle_data['id'].append(angle_id)
                angle_data['type'].append(angle_type_str)
                angle_data['atom1'].append(atom1)
                angle_data['atom2'].append(atom2)
                angle_data['atom3'].append(atom3)
        
        # Convert to numpy arrays and create Dataset
        data_vars = {}
        n_angles = len(angle_data['id'])
        
        for key, values in angle_data.items():
            if key == 'type':
                # Keep type as string array
                data_vars[key] = (['angles_id'], np.array(values, dtype=str))
            else:
                data_vars[key] = (['angles_id'], np.array(values))
        
        coords = {'angles_id': np.arange(n_angles)}
        angles_dataset = xr.Dataset(data_vars, coords=coords)
        frame['angles'] = angles_dataset
    
    def _parse_dihedrals(self, dihedral_lines: List[str], frame: mp.Frame) -> None:
        """Parse dihedrals section and add to frame as xarray.Dataset."""
        if not dihedral_lines:
            return
        
        dihedral_data = {
            'id': [],
            'type': [],
            'atom1': [],
            'atom2': [],
            'atom3': [],
            'atom4': []
        }
        
        for line in dihedral_lines:
            if line.strip():
                parts = line.split()
                dihedral_id = int(parts[0])
                dihedral_type_str = parts[1]  # Always treat as string
                atom1 = int(parts[2])
                atom2 = int(parts[3])
                atom3 = int(parts[4])
                atom4 = int(parts[5])
                
                dihedral_data['id'].append(dihedral_id)
                dihedral_data['type'].append(dihedral_type_str)
                dihedral_data['atom1'].append(atom1)
                dihedral_data['atom2'].append(atom2)
                dihedral_data['atom3'].append(atom3)
                dihedral_data['atom4'].append(atom4)
        
        # Convert to numpy arrays and create Dataset
        data_vars = {}
        n_dihedrals = len(dihedral_data['id'])
        
        for key, values in dihedral_data.items():
            if key == 'type':
                # Keep type as string array
                data_vars[key] = (['dihedrals_id'], np.array(values, dtype=str))
            else:
                data_vars[key] = (['dihedrals_id'], np.array(values))
        
        coords = {'dihedrals_id': np.arange(n_dihedrals)}
        dihedrals_dataset = xr.Dataset(data_vars, coords=coords)
        frame['dihedrals'] = dihedrals_dataset
    
    def _parse_impropers(self, improper_lines: List[str], frame: mp.Frame) -> None:
        """Parse impropers section and add to frame as xarray.Dataset."""
        if not improper_lines:
            return
        
        improper_data = {
            'id': [],
            'type': [],
            'atom1': [],
            'atom2': [],
            'atom3': [],
            'atom4': []
        }
        
        for line in improper_lines:
            if line.strip():
                parts = line.split()
                improper_id = int(parts[0])
                improper_type_str = parts[1]  # Always treat as string
                atom1 = int(parts[2])
                atom2 = int(parts[3])
                atom3 = int(parts[4])
                atom4 = int(parts[5])
                
                improper_data['id'].append(improper_id)
                improper_data['type'].append(improper_type_str)
                improper_data['atom1'].append(atom1)
                improper_data['atom2'].append(atom2)
                improper_data['atom3'].append(atom3)
                improper_data['atom4'].append(atom4)
        
        # Convert to numpy arrays and create Dataset
        data_vars = {}
        n_impropers = len(improper_data['id'])
        
        for key, values in improper_data.items():
            if key == 'type':
                # Keep type as string array
                data_vars[key] = (['impropers_id'], np.array(values, dtype=str))
            else:
                data_vars[key] = (['impropers_id'], np.array(values))
        
        coords = {'impropers_id': np.arange(n_impropers)}
        impropers_dataset = xr.Dataset(data_vars, coords=coords)
        frame['impropers'] = impropers_dataset


class LammpsDataWriter(DataWriter):
    """Modern LAMMPS data file writer using xarray exclusively."""
    
    def __init__(self, path: Union[str, Path], atom_style: str = "full"):
        super().__init__(Path(path))  # Convert to Path explicitly
        self.atom_style = atom_style
    
    def write(self, frame: mp.Frame) -> None:
        """Write Frame to LAMMPS data file."""
        lines = []
        
        # Header
        lines.append(f"# LAMMPS data file written by molpy on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Count sections
        n_atoms = len(frame['atoms']['id']) if 'atoms' in frame else 0
        n_bonds = len(frame['bonds']['id']) if 'bonds' in frame else 0
        n_angles = len(frame['angles']['id']) if 'angles' in frame else 0
        n_dihedrals = len(frame['dihedrals']['id']) if 'dihedrals' in frame else 0
        n_impropers = len(frame['impropers']['id']) if 'impropers' in frame else 0
        
        lines.append(f"{n_atoms} atoms")
        if n_bonds > 0:
            lines.append(f"{n_bonds} bonds")
        if n_angles > 0:
            lines.append(f"{n_angles} angles")
        if n_dihedrals > 0:
            lines.append(f"{n_dihedrals} dihedrals")
        if n_impropers > 0:
            lines.append(f"{n_impropers} impropers")
        lines.append("")
        
        # Type counts - handle both numeric and string types
        if 'atoms' in frame:
            atom_types = frame['atoms']['type'].values
            unique_atom_types = np.unique(atom_types)
            
            # Try to convert to int, if all can be converted use max, otherwise use count
            try:
                numeric_types = [int(t) for t in unique_atom_types]
                n_atom_types = max(numeric_types) if numeric_types else 0
            except ValueError:
                # If conversion fails, use count of unique types
                n_atom_types = len(unique_atom_types)
            lines.append(f"{n_atom_types} atom types")
        
        if 'bonds' in frame and n_bonds > 0:
            bond_types = frame['bonds']['type'].values
            unique_bond_types = np.unique(bond_types)
            
            try:
                numeric_types = [int(t) for t in unique_bond_types]
                n_bond_types = max(numeric_types) if numeric_types else 0
            except ValueError:
                n_bond_types = len(unique_bond_types)
            lines.append(f"{n_bond_types} bond types")
        
        if 'angles' in frame and n_angles > 0:
            angle_types = frame['angles']['type'].values
            unique_angle_types = np.unique(angle_types)
            
            try:
                numeric_types = [int(t) for t in unique_angle_types]
                n_angle_types = max(numeric_types) if numeric_types else 0
            except ValueError:
                n_angle_types = len(unique_angle_types)
            lines.append(f"{n_angle_types} angle types")
        
        if 'dihedrals' in frame and n_dihedrals > 0:
            dihedral_types = frame['dihedrals']['type'].values
            unique_dihedral_types = np.unique(dihedral_types)
            
            try:
                numeric_types = [int(t) for t in unique_dihedral_types]
                n_dihedral_types = max(numeric_types) if numeric_types else 0
            except ValueError:
                n_dihedral_types = len(unique_dihedral_types)
            lines.append(f"{n_dihedral_types} dihedral types")
        
        if 'impropers' in frame and n_impropers > 0:
            improper_types = frame['impropers']['type'].values
            unique_improper_types = np.unique(improper_types)
            
            try:
                numeric_types = [int(t) for t in unique_improper_types]
                n_improper_types = max(numeric_types) if numeric_types else 0
            except ValueError:
                n_improper_types = len(unique_improper_types)
            lines.append(f"{n_improper_types} improper types")
        
        lines.append("")
        
        # Box bounds
        if frame.box is not None:
            box = frame.box
            lines.append(f"{box.origin[0]:.6f} {box.origin[0] + box.lengths[0]:.6f} xlo xhi")
            lines.append(f"{box.origin[1]:.6f} {box.origin[1] + box.lengths[1]:.6f} ylo yhi")
            lines.append(f"{box.origin[2]:.6f} {box.origin[2] + box.lengths[2]:.6f} zlo zhi")
        else:
            # Default box if none provided
            lines.append("0.0 10.0 xlo xhi")
            lines.append("0.0 10.0 ylo yhi")
            lines.append("0.0 10.0 zlo zhi")
        
        lines.append("")
        
        # Masses section
        if 'atoms' in frame:
            self._write_masses(lines, frame)
        
        # Atoms section
        if 'atoms' in frame:
            self._write_atoms(lines, frame)
        
        # Bonds section
        if 'bonds' in frame and n_bonds > 0:
            self._write_bonds(lines, frame)
        
        # Angles section
        if 'angles' in frame and n_angles > 0:
            self._write_angles(lines, frame)
        
        # Dihedrals section
        if 'dihedrals' in frame and n_dihedrals > 0:
            self._write_dihedrals(lines, frame)
        
        # Impropers section
        if 'impropers' in frame and n_impropers > 0:
            self._write_impropers(lines, frame)
        
        # Write to file
        with open(self._path, 'w') as f:
            f.write('\n'.join(lines))
    
    def _write_masses(self, lines: List[str], frame: mp.Frame) -> None:
        """Write masses section."""
        lines.append("Masses")
        lines.append("")
        
        atoms_data = frame['atoms']
        
        # Get unique atom types and their masses
        unique_types = np.unique(atoms_data['type'].values)
        
        # Create type mapping - if types are numeric strings, use them as IDs
        # If they are string labels, create sequential IDs
        type_to_id = {}
        try:
            # Try numeric sorting
            sorted_types = sorted(unique_types, key=int)
            for atom_type_str in sorted_types:
                type_to_id[atom_type_str] = int(atom_type_str)
        except ValueError:
            # String labels - assign sequential IDs
            for i, atom_type_str in enumerate(sorted(unique_types), 1):
                type_to_id[atom_type_str] = i
        
        for atom_type_str in sorted(unique_types, key=lambda x: type_to_id[x]):
            # Find first occurrence of this type to get mass
            mask = atoms_data['type'].values == atom_type_str
            mass = atoms_data['mass'].values[mask][0]
            # Use mapped type ID for output
            lines.append(f"{type_to_id[atom_type_str]} {mass:.6f}")
        
        lines.append("")
    
    def _write_atoms(self, lines: List[str], frame: mp.Frame) -> None:
        """Write atoms section."""
        lines.append("Atoms")
        lines.append("")
        
        atoms_data = frame['atoms']
        
        # Create type mapping like in _write_masses
        unique_types = np.unique(atoms_data['type'].values)
        type_to_id = {}
        try:
            # Try numeric sorting
            sorted_types = sorted(unique_types, key=int)
            for atom_type_str in sorted_types:
                type_to_id[atom_type_str] = int(atom_type_str)
        except ValueError:
            # String labels - assign sequential IDs
            for i, atom_type_str in enumerate(sorted(unique_types), 1):
                type_to_id[atom_type_str] = i
        
        for i in range(len(atoms_data['id'])):
            atom_id = int(atoms_data['id'].values[i])
            atom_type_str = atoms_data['type'].values[i]
            atom_type = type_to_id[atom_type_str]  # Convert using mapping
            x, y, z = atoms_data['xyz'].values[i]
            
            if self.atom_style == "full":
                mol_id = int(atoms_data['mol'].values[i]) if 'mol' in atoms_data else 1
                charge = float(atoms_data['q'].values[i]) if 'q' in atoms_data else 0.0
                lines.append(f"{atom_id} {mol_id} {atom_type} {charge:.6f} {x:.6f} {y:.6f} {z:.6f}")
            elif self.atom_style == "charge":
                charge = float(atoms_data['q'].values[i]) if 'q' in atoms_data else 0.0
                lines.append(f"{atom_id} {atom_type} {charge:.6f} {x:.6f} {y:.6f} {z:.6f}")
            else:  # atomic
                lines.append(f"{atom_id} {atom_type} {x:.6f} {y:.6f} {z:.6f}")
        
        lines.append("")
    
    def _write_bonds(self, lines: List[str], frame: mp.Frame) -> None:
        """Write bonds section."""
        lines.append("Bonds")
        lines.append("")
        
        bonds_data = frame['bonds']
        
        # Create bond type to id mapping - always treat as strings
        bond_types = bonds_data['type'].values
        unique_bond_types = list(dict.fromkeys(str(t) for t in bond_types))
        bond_type_to_id = {bond_type: i+1 for i, bond_type in enumerate(unique_bond_types)}
        
        for i in range(len(bonds_data['id'])):
            bond_id = int(bonds_data['id'].values[i])
            bond_type = bond_type_to_id[str(bonds_data['type'].values[i])]
            # Support both 'atom1'/'atom2' and 'i'/'j' naming
            if 'atom1' in bonds_data:
                atom1 = int(bonds_data['atom1'].values[i])
                atom2 = int(bonds_data['atom2'].values[i])
            else:
                atom1 = int(bonds_data['i'].values[i])
                atom2 = int(bonds_data['j'].values[i])
            lines.append(f"{bond_id} {bond_type} {atom1} {atom2}")
        
        lines.append("")
    
    def _write_angles(self, lines: List[str], frame: mp.Frame) -> None:
        """Write angles section."""
        lines.append("Angles")
        lines.append("")
        
        angles_data = frame['angles']
        
        # Create angle type to id mapping - always treat as strings
        angle_types = angles_data['type'].values
        unique_angle_types = list(dict.fromkeys(str(t) for t in angle_types))
        angle_type_to_id = {angle_type: i+1 for i, angle_type in enumerate(unique_angle_types)}
        
        for i in range(len(angles_data['id'])):
            angle_id = int(angles_data['id'].values[i])
            angle_type = angle_type_to_id[str(angles_data['type'].values[i])]
            # Support both 'atom1'/'atom2'/'atom3' and 'i'/'j'/'k' naming
            if 'atom1' in angles_data:
                atom1 = int(angles_data['atom1'].values[i])
                atom2 = int(angles_data['atom2'].values[i])
                atom3 = int(angles_data['atom3'].values[i])
            else:
                atom1 = int(angles_data['i'].values[i])
                atom2 = int(angles_data['j'].values[i])
                atom3 = int(angles_data['k'].values[i])
            lines.append(f"{angle_id} {angle_type} {atom1} {atom2} {atom3}")
        
        lines.append("")
    
    def _write_dihedrals(self, lines: List[str], frame: mp.Frame) -> None:
        """Write dihedrals section."""
        lines.append("Dihedrals")
        lines.append("")
        
        dihedrals_data = frame['dihedrals']
        
        # Create dihedral type to id mapping - always treat as strings
        dihedral_types = dihedrals_data['type'].values
        unique_dihedral_types = list(dict.fromkeys(str(t) for t in dihedral_types))
        dihedral_type_to_id = {dihedral_type: i+1 for i, dihedral_type in enumerate(unique_dihedral_types)}
        
        for i in range(len(dihedrals_data['id'])):
            dihedral_id = int(dihedrals_data['id'].values[i])
            dihedral_type = dihedral_type_to_id[str(dihedrals_data['type'].values[i])]
            # Support both 'atom1'/'atom2'/'atom3'/'atom4' and 'i'/'j'/'k'/'l' naming
            if 'atom1' in dihedrals_data:
                atom1 = int(dihedrals_data['atom1'].values[i])
                atom2 = int(dihedrals_data['atom2'].values[i])
                atom3 = int(dihedrals_data['atom3'].values[i])
                atom4 = int(dihedrals_data['atom4'].values[i])
            else:
                atom1 = int(dihedrals_data['i'].values[i])
                atom2 = int(dihedrals_data['j'].values[i])
                atom3 = int(dihedrals_data['k'].values[i])
                atom4 = int(dihedrals_data['l'].values[i])
            lines.append(f"{dihedral_id} {dihedral_type} {atom1} {atom2} {atom3} {atom4}")
        
        lines.append("")
    
    def _write_impropers(self, lines: List[str], frame: mp.Frame) -> None:
        """Write impropers section."""
        lines.append("Impropers")
        lines.append("")
        
        impropers_data = frame['impropers']
        
        # Create improper type to id mapping - always treat as strings
        improper_types = impropers_data['type'].values
        unique_improper_types = list(dict.fromkeys(str(t) for t in improper_types))
        improper_type_to_id = {improper_type: i+1 for i, improper_type in enumerate(unique_improper_types)}
        
        for i in range(len(impropers_data['id'])):
            improper_id = int(impropers_data['id'].values[i])
            improper_type = improper_type_to_id[str(impropers_data['type'].values[i])]
            atom1 = int(impropers_data['atom1'].values[i])
            atom2 = int(impropers_data['atom2'].values[i])
            atom3 = int(impropers_data['atom3'].values[i])
            atom4 = int(impropers_data['atom4'].values[i])
            lines.append(f"{improper_id} {improper_type} {atom1} {atom2} {atom3} {atom4}")
        
        lines.append("")


class LammpsMoleculeReader(DataReader):
    """LAMMPS molecule template file reader using xarray exclusively."""
    
    def __init__(self, path: Union[str, Path]):
        super().__init__(Path(path))  # Convert to Path explicitly
    
    def read(self, frame: Optional[mp.Frame] = None) -> mp.Frame:
        """Read LAMMPS molecule template file into a Frame."""
        if frame is None:
            frame = mp.Frame()
        
        lines = self._read_and_clean_lines()
        sections = self._parse_sections(lines)
        
        # Parse header
        counts = self._parse_header(sections.get('header', []))
        
        # Parse sections
        if 'Coords' in sections:
            self._parse_coords(sections['Coords'], frame)
        
        if 'Types' in sections:
            self._parse_types(sections['Types'], frame)
        
        if 'Charges' in sections:
            self._parse_charges(sections['Charges'], frame)
        
        if 'Bonds' in sections:
            self._parse_bonds(sections['Bonds'], frame)
        
        if 'Angles' in sections:
            self._parse_angles(sections['Angles'], frame)
        
        if 'Dihedrals' in sections:
            self._parse_dihedrals(sections['Dihedrals'], frame)
        
        if 'Impropers' in sections:
            self._parse_impropers(sections['Impropers'], frame)
        
        # Store metadata
        frame._meta.update({
            'format': 'lammps_molecule',
            'counts': counts,
            'source_file': str(self._path)
        })
        
        return frame
    
    def _read_and_clean_lines(self) -> List[str]:
        """Read file and return cleaned lines."""
        with open(self._path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        return [line for line in lines if line and not line.startswith('#')]
    
    def _parse_sections(self, lines: List[str]) -> Dict[str, List[str]]:
        """Parse file into sections."""
        sections = {'header': []}
        current_section = 'header'
        
        section_keywords = ['coords', 'types', 'charges', 'bonds', 'angles', 'dihedrals', 'impropers']
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in section_keywords):
                current_section = line.capitalize()
                sections[current_section] = []
            else:
                if current_section not in sections:
                    sections[current_section] = []
                sections[current_section].append(line)
        
        return sections
    
    def _parse_header(self, header_lines: List[str]) -> Dict[str, int]:
        """Parse header information."""
        counts = {}
        
        for line in header_lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    count = int(parts[0])
                    if 'atoms' in line.lower():
                        counts['atoms'] = count
                    elif 'bonds' in line.lower():
                        counts['bonds'] = count
                    elif 'angles' in line.lower():
                        counts['angles'] = count
                    elif 'dihedrals' in line.lower():
                        counts['dihedrals'] = count
                    elif 'impropers' in line.lower():
                        counts['impropers'] = count
                except ValueError:
                    continue
        
        return counts
    
    def _parse_coords(self, coord_lines: List[str], frame: mp.Frame) -> None:
        """Parse coordinates section."""
        coords_data = {
            'id': [],
            'x': [],
            'y': [],
            'z': []
        }
        
        for line in coord_lines:
            if line.strip():
                parts = line.split()
                atom_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                
                coords_data['id'].append(atom_id)
                coords_data['x'].append(x)
                coords_data['y'].append(y)
                coords_data['z'].append(z)
        
        if coords_data['id']:
            # Convert to numpy arrays
            for key, values in coords_data.items():
                coords_data[key] = np.array(values)
            
            # Create xyz coordinate array
            xyz = np.column_stack([coords_data['x'], coords_data['y'], coords_data['z']])
            
            # Create or update atoms dataset
            if 'atoms' not in frame:
                data_vars = {
                    'id': (['atoms_id'], coords_data['id']),
                    'xyz': (['atoms_id', 'spatial'], xyz)
                }
                coords_dict = {
                    'atoms_id': np.arange(len(coords_data['id'])),
                    'spatial': ['x', 'y', 'z']
                }
                frame['atoms'] = xr.Dataset(data_vars, coords=coords_dict)
            else:
                # Update existing dataset
                frame['atoms'] = frame['atoms'].assign({
                    'xyz': (['atoms_id', 'spatial'], xyz)
                })
    
    def _parse_types(self, type_lines: List[str], frame: mp.Frame) -> None:
        """Parse types section."""
        types_data = {'id': [], 'type': []}
        
        for line in type_lines:
            if line.strip():
                parts = line.split()
                atom_id = int(parts[0])
                atom_type_str = parts[1]  # Always treat as string
                
                types_data['id'].append(atom_id)
                types_data['type'].append(atom_type_str)
        
        if types_data['id']:
            type_array = np.array(types_data['type'], dtype=str)  # Ensure string array
            
            if 'atoms' not in frame:
                data_vars = {
                    'id': (['atoms_id'], np.array(types_data['id'])),
                    'type': (['atoms_id'], type_array)
                }
                coords_dict = {'atoms_id': np.arange(len(types_data['id']))}
                frame['atoms'] = xr.Dataset(data_vars, coords=coords_dict)
            else:
                frame['atoms'] = frame['atoms'].assign({'type': (['atoms_id'], type_array)})
    
    def _parse_charges(self, charge_lines: List[str], frame: mp.Frame) -> None:
        """Parse charges section."""
        charges_data = {'id': [], 'q': []}
        
        for line in charge_lines:
            if line.strip():
                parts = line.split()
                atom_id = int(parts[0])
                charge = float(parts[1])
                
                charges_data['id'].append(atom_id)
                charges_data['q'].append(charge)
        
        if charges_data['id']:
            charge_array = np.array(charges_data['q'])
            
            if 'atoms' not in frame:
                data_vars = {
                    'id': (['atoms_id'], np.array(charges_data['id'])),
                    'q': (['atoms_id'], charge_array)
                }
                coords_dict = {'atoms_id': np.arange(len(charges_data['id']))}
                frame['atoms'] = xr.Dataset(data_vars, coords=coords_dict)
            else:
                frame['atoms'] = frame['atoms'].assign({'q': (['atoms_id'], charge_array)})
    
    def _parse_bonds(self, bond_lines: List[str], frame: mp.Frame) -> None:
        """Parse bonds section."""
        bond_data = {
            'id': [],
            'type': [],
            'atom1': [],
            'atom2': []
        }
        
        for line in bond_lines:
            if line.strip():
                parts = line.split()
                bond_id = int(parts[0])
                bond_type_str = parts[1]  # Always treat as string
                atom1 = int(parts[2])
                atom2 = int(parts[3])
                
                bond_data['id'].append(bond_id)
                bond_data['type'].append(bond_type_str)
                bond_data['atom1'].append(atom1)
                bond_data['atom2'].append(atom2)
        
        if bond_data['id']:
            # Convert to numpy arrays and create Dataset
            data_vars = {}
            n_bonds = len(bond_data['id'])
            
            for key, values in bond_data.items():
                if key == 'type':
                    # Keep type as string array
                    data_vars[key] = (['bonds_id'], np.array(values, dtype=str))
                else:
                    data_vars[key] = (['bonds_id'], np.array(values))
            
            coords = {'bonds_id': np.arange(n_bonds)}
            frame['bonds'] = xr.Dataset(data_vars, coords=coords)
    
    def _parse_angles(self, angle_lines: List[str], frame: mp.Frame) -> None:
        """Parse angles section - same as LammpsDataReader."""
        if not angle_lines:
            return
        
        angle_data = {
            'id': [],
            'type': [],
            'atom1': [],
            'atom2': [],
            'atom3': []
        }
        
        for line in angle_lines:
            if line.strip():
                parts = line.split()
                angle_id = int(parts[0])
                angle_type_str = parts[1]  # Always treat as string
                atom1 = int(parts[2])
                atom2 = int(parts[3])
                atom3 = int(parts[4])
                
                angle_data['id'].append(angle_id)
                angle_data['type'].append(angle_type_str)
                angle_data['atom1'].append(atom1)
                angle_data['atom2'].append(atom2)
                angle_data['atom3'].append(atom3)
        
        if angle_data['id']:
            # Convert to numpy arrays and create Dataset
            data_vars = {}
            n_angles = len(angle_data['id'])
            
            for key, values in angle_data.items():
                if key == 'type':
                    # Keep type as string array
                    data_vars[key] = (['angles_id'], np.array(values, dtype=str))
                else:
                    data_vars[key] = (['angles_id'], np.array(values))
            
            coords = {'angles_id': np.arange(n_angles)}
            frame['angles'] = xr.Dataset(data_vars, coords=coords)
    
    def _parse_dihedrals(self, dihedral_lines: List[str], frame: mp.Frame) -> None:
        """Parse dihedrals section - same as LammpsDataReader."""
        if not dihedral_lines:
            return
        
        dihedral_data = {
            'id': [],
            'type': [],
            'atom1': [],
            'atom2': [],
            'atom3': [],
            'atom4': []
        }
        
        for line in dihedral_lines:
            if line.strip():
                parts = line.split()
                dihedral_id = int(parts[0])
                dihedral_type_str = parts[1]  # Always treat as string
                atom1 = int(parts[2])
                atom2 = int(parts[3])
                atom3 = int(parts[4])
                atom4 = int(parts[5])
                
                dihedral_data['id'].append(dihedral_id)
                dihedral_data['type'].append(dihedral_type_str)
                dihedral_data['atom1'].append(atom1)
                dihedral_data['atom2'].append(atom2)
                dihedral_data['atom3'].append(atom3)
                dihedral_data['atom4'].append(atom4)
        
        if dihedral_data['id']:
            # Convert to numpy arrays and create Dataset
            data_vars = {}
            n_dihedrals = len(dihedral_data['id'])
            
            for key, values in dihedral_data.items():
                if key == 'type':
                    # Keep type as string array
                    data_vars[key] = (['dihedrals_id'], np.array(values, dtype=str))
                else:
                    data_vars[key] = (['dihedrals_id'], np.array(values))
            
            coords = {'dihedrals_id': np.arange(n_dihedrals)}
            frame['dihedrals'] = xr.Dataset(data_vars, coords=coords)
    
    def _parse_impropers(self, improper_lines: List[str], frame: mp.Frame) -> None:
        """Parse impropers section - same as LammpsDataReader."""
        if not improper_lines:
            return
        
        improper_data = {
            'id': [],
            'type': [],
            'atom1': [],
            'atom2': [],
            'atom3': [],
            'atom4': []
        }
        
        for line in improper_lines:
            if line.strip():
                parts = line.split()
                improper_id = int(parts[0])
                improper_type_str = parts[1]  # Always treat as string
                atom1 = int(parts[2])
                atom2 = int(parts[3])
                atom3 = int(parts[4])
                atom4 = int(parts[5])
                
                improper_data['id'].append(improper_id)
                improper_data['type'].append(improper_type_str)
                improper_data['atom1'].append(atom1)
                improper_data['atom2'].append(atom2)
                improper_data['atom3'].append(atom3)
                improper_data['atom4'].append(atom4)
        
        if improper_data['id']:
            # Convert to numpy arrays and create Dataset
            data_vars = {}
            n_impropers = len(improper_data['id'])
            
            for key, values in improper_data.items():
                if key == 'type':
                    # Keep type as string array
                    data_vars[key] = (['impropers_id'], np.array(values, dtype=str))
                else:
                    data_vars[key] = (['impropers_id'], np.array(values))
            
            coords = {'impropers_id': np.arange(n_impropers)}
            frame['impropers'] = xr.Dataset(data_vars, coords=coords)


class LammpsMoleculeWriter(DataWriter):
    """LAMMPS molecule template file writer using xarray exclusively."""
    
    def __init__(self, path: Union[str, Path]):
        super().__init__(Path(path))  # Convert to Path explicitly
    
    def write(self, frame: mp.Frame) -> None:
        """Write Frame to LAMMPS molecule template file."""
        lines = []
        
        # Header comment
        lines.append(f"# LAMMPS molecule template written by molpy on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Count lines
        n_atoms = len(frame['atoms']['id']) if 'atoms' in frame else 0
        n_bonds = len(frame['bonds']['id']) if 'bonds' in frame else 0
        n_angles = len(frame['angles']['id']) if 'angles' in frame else 0
        n_dihedrals = len(frame['dihedrals']['id']) if 'dihedrals' in frame else 0
        n_impropers = len(frame['impropers']['id']) if 'impropers' in frame else 0
        
        lines.append(f"{n_atoms} atoms")
        if n_bonds > 0:
            lines.append(f"{n_bonds} bonds")
        if n_angles > 0:
            lines.append(f"{n_angles} angles")
        if n_dihedrals > 0:
            lines.append(f"{n_dihedrals} dihedrals")
        if n_impropers > 0:
            lines.append(f"{n_impropers} impropers")
        lines.append("")
        
        # Coords section
        if 'atoms' in frame and 'xyz' in frame['atoms']:
            self._write_coords(lines, frame)
        
        # Types section
        if 'atoms' in frame and 'type' in frame['atoms']:
            self._write_types(lines, frame)
        
        # Charges section
        if 'atoms' in frame and 'q' in frame['atoms']:
            self._write_charges(lines, frame)
        
        # Bonds section
        if 'bonds' in frame and n_bonds > 0:
            self._write_bonds(lines, frame)
        
        # Angles section
        if 'angles' in frame and n_angles > 0:
            self._write_angles(lines, frame)
        
        # Dihedrals section
        if 'dihedrals' in frame and n_dihedrals > 0:
            self._write_dihedrals(lines, frame)
        
        # Impropers section
        if 'impropers' in frame and n_impropers > 0:
            self._write_impropers(lines, frame)
        
        # Write to file
        with open(self._path, 'w') as f:
            f.write('\n'.join(lines))
    
    def _write_coords(self, lines: List[str], frame: mp.Frame) -> None:
        """Write coordinates section."""
        lines.append("Coords")
        lines.append("")
        
        atoms_data = frame['atoms']
        
        for i in range(len(atoms_data['id'])):
            atom_id = int(atoms_data['id'].values[i])
            x, y, z = atoms_data['xyz'].values[i]
            lines.append(f"{atom_id} {x:.6f} {y:.6f} {z:.6f}")
        
        lines.append("")
    
    def _write_types(self, lines: List[str], frame: mp.Frame) -> None:
        """Write types section."""
        lines.append("Types")
        lines.append("")
        
        atoms_data = frame['atoms']
        atom_ids = atoms_data['id'].values.tolist()
        atom_types = atoms_data['type'].values.tolist()
        for i in range(len(atoms_data['id'])):
            atom_id = atom_ids[i]
            atom_type = atom_types[i]
            lines.append(f"{atom_id} {atom_type}")
        
        lines.append("")
    
    def _write_charges(self, lines: List[str], frame: mp.Frame) -> None:
        """Write charges section."""
        lines.append("Charges")
        lines.append("")
        
        atoms_data = frame['atoms']
        
        for i in range(len(atoms_data['id'])):
            atom_id = int(atoms_data['id'].values[i])
            charge = float(atoms_data['q'].values[i])
            lines.append(f"{atom_id} {charge:.6f}")
        
        lines.append("")
    
    def _write_bonds(self, lines: List[str], frame: mp.Frame) -> None:
        """Write bonds section."""
        lines.append("Bonds")
        lines.append("")
        
        bonds_data = frame['bonds']
    
        for i in range(len(bonds_data['id'])):
            bond_id = int(bonds_data['id'].values[i])
            bond_type = str(bonds_data['type'].values[i])
            atom1 = int(bonds_data['i'].values[i]) + 1
            atom2 = int(bonds_data['j'].values[i]) + 1
            lines.append(f"{bond_id} {bond_type} {atom1} {atom2}")
        
        lines.append("")
    
    def _write_angles(self, lines: List[str], frame: mp.Frame) -> None:
        """Write angles section."""
        lines.append("Angles")
        lines.append("")
        
        angles_data = frame['angles']

        for i in range(len(angles_data['id'])):
            angle_id = int(angles_data['id'].values[i])
            angle_type = str(angles_data['type'].values[i])
            atom1 = int(angles_data['i'].values[i]) + 1
            atom2 = int(angles_data['j'].values[i]) + 1
            atom3 = int(angles_data['k'].values[i]) + 1
            lines.append(f"{angle_id} {angle_type} {atom1} {atom2} {atom3}")
        
        lines.append("")
    
    def _write_dihedrals(self, lines: List[str], frame: mp.Frame) -> None:
        """Write dihedrals section."""
        lines.append("Dihedrals")
        lines.append("")
        
        dihedrals_data = frame['dihedrals']
        
        for i in range(len(dihedrals_data['id'])):
            dihedral_id = int(dihedrals_data['id'].values[i])
            dihedral_type = str(dihedrals_data['type'].values[i]) 
            atom1 = int(dihedrals_data['i'].values[i]) + 1
            atom2 = int(dihedrals_data['j'].values[i]) + 1
            atom3 = int(dihedrals_data['k'].values[i]) + 1
            atom4 = int(dihedrals_data['l'].values[i]) + 1
            lines.append(f"{dihedral_id} {dihedral_type} {atom1} {atom2} {atom3} {atom4}")
        
        lines.append("")
    
    def _write_impropers(self, lines: List[str], frame: mp.Frame) -> None:
        """Write impropers section."""
        lines.append("Impropers")
        lines.append("")
        
        impropers_data = frame['impropers']
        
        for i in range(len(impropers_data['id'])):
            improper_id = int(impropers_data['id'].values[i])
            improper_type = impropers_data['type'].values[i]
            atom1 = int(impropers_data['i'].values[i]) + 1
            atom2 = int(impropers_data['j'].values[i]) + 1
            atom3 = int(impropers_data['k'].values[i]) + 1
            atom4 = int(impropers_data['l'].values[i]) + 1
            lines.append(f"{improper_id} {improper_type} {atom1} {atom2} {atom3} {atom4}")
        
        lines.append("")
