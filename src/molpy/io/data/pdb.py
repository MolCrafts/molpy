from pathlib import Path
import re
from .base import DataWriter, DataReader
import numpy as np

import molpy as mp
from collections import defaultdict
from molpy.core.frame import _dict_to_dataset


class PDBReader(DataReader):
    """
    Robust PDB file reader that handles various PDB formats and edge cases.
    
    Features:
    - Parses ATOM and HETATM records
    - Handles CRYST1 records for unit cell information
    - Parses CONECT records for bond information
    - Handles missing fields gracefully
    - Processes duplicate atom names
    - Supports both standard and non-standard PDB formats
    """

    def __init__(self, file: Path):
        super().__init__(path=file)

    @staticmethod
    def sanitizer(line: str) -> str:
        """Clean up line by stripping whitespace but preserving PDB formatting."""
        return line.rstrip('\n\r')

    def _parse_cryst1(self, line: str) -> dict:
        """Parse CRYST1 record for unit cell parameters."""
        try:
            # CRYST1 record format: CRYST1 a b c alpha beta gamma spacegroup z
            parts = line.split()
            if len(parts) >= 7:
                a, b, c = float(parts[1]), float(parts[2]), float(parts[3])
                alpha, beta, gamma = float(parts[4]), float(parts[5]), float(parts[6])
                
                # Convert to box matrix (simplified orthogonal case)
                if abs(alpha - 90.0) < 1e-6 and abs(beta - 90.0) < 1e-6 and abs(gamma - 90.0) < 1e-6:
                    # Orthogonal box
                    matrix = np.diag([a, b, c])
                else:
                    # For non-orthogonal, we need proper conversion
                    # Simplified implementation - use orthogonal approximation
                    matrix = np.diag([a, b, c])
                
                return {
                    'matrix': matrix,
                    'a': a, 'b': b, 'c': c,
                    'alpha': alpha, 'beta': beta, 'gamma': gamma
                }
        except (ValueError, IndexError):
            pass
        
        return None

    def _parse_atom_line(self, line: str) -> dict | None:
        """Parse ATOM or HETATM record according to PDB v3.3 format specification.
        
        PDB Format v3.3 ATOM/HETATM Record:
        COLUMNS        DATA TYPE    FIELD          DEFINITION
        --------------------------------------------------------------------------------
         1 -  6        Record name  "ATOM  " or "HETATM"
         7 - 11        Integer      serial         Atom serial number.
        13 - 16        Atom         name           Atom name.
        17             Character    altLoc         Alternate location indicator.
        18 - 20        Residue name resName        Residue name.
        22             Character    chainID        Chain identifier.
        23 - 26        Integer      resSeq         Residue sequence number.
        27             AChar        iCode          Code for insertion of residues.
        31 - 38        Real(8.3)    x              Orthogonal coordinates for X in Angstroms.
        39 - 46        Real(8.3)    y              Orthogonal coordinates for Y in Angstroms.
        47 - 54        Real(8.3)    z              Orthogonal coordinates for Z in Angstroms.
        55 - 60        Real(6.2)    occupancy      Occupancy.
        61 - 66        Real(6.2)    tempFactor     Temperature factor.
        77 - 78        LString(2)   element        Element symbol, right-justified.
        79 - 80        LString(2)   charge         Charge on the atom.
        """
        if len(line) < 54:  # Minimum required length for coordinates
            return None
            
        try:
            # Record name (1-6)
            record_type = line[0:6].strip()
            
            # Serial number (7-11)
            serial_str = line[6:11].strip()
            serial = int(serial_str) if serial_str else 0
            
            # Atom name (13-16) - note: column 12 is space
            name = line[12:16].strip()
            
            # Alternate location indicator (17)
            altLoc = line[16:17] if len(line) > 16 else ' '
            
            # Residue name (18-20)
            resName = line[17:20].strip() if len(line) > 19 else ''
            
            # Chain identifier (22) - note: column 21 is space
            chainID = line[21:22] if len(line) > 21 else ' '
            
            # Residue sequence number (23-26)
            resSeq_str = line[22:26].strip() if len(line) > 25 else ''
            resSeq = int(resSeq_str) if resSeq_str and resSeq_str.isdigit() else 0
            
            # Insertion code (27)
            iCode = line[26:27] if len(line) > 26 else ' '
            
            # Coordinates (31-38, 39-46, 47-54) - note: columns 28-30 are spaces
            x_str = line[30:38].strip() if len(line) > 37 else '0.0'
            y_str = line[38:46].strip() if len(line) > 45 else '0.0'
            z_str = line[46:54].strip() if len(line) > 53 else '0.0'
            
            x = float(x_str) if x_str else 0.0
            y = float(y_str) if y_str else 0.0
            z = float(z_str) if z_str else 0.0
            
            # Occupancy (55-60)
            occupancy_str = line[54:60].strip() if len(line) > 59 else '1.0'
            occupancy = float(occupancy_str) if occupancy_str else 1.0
            
            # Temperature factor (61-66)
            tempFactor_str = line[60:66].strip() if len(line) > 65 else '0.0'
            tempFactor = float(tempFactor_str) if tempFactor_str else 0.0
            
            # Element symbol (77-78) - note: columns 67-76 are spaces or other data
            element = line[76:78].strip() if len(line) > 77 else ''
            
            # Charge (79-80)
            charge = line[78:80].strip() if len(line) > 79 else ''
            
            # If element is empty, try to guess from atom name
            if not element and name:
                # Extract alphabetic part from atom name
                import re
                element_match = re.match(r'([A-Za-z]+)', name)
                if element_match:
                    element_guess = element_match.group(1)
                    # Take first 1-2 characters as element
                    if len(element_guess) >= 2 and element_guess[:2] in ['BR', 'CL', 'FE', 'MG', 'CA', 'ZN', 'NI', 'CU']:
                        element = element_guess[:2].upper()
                    else:
                        element = element_guess[0].upper()
            
            return {
                'record_type': record_type,
                'id': serial,
                'name': name,
                'altLoc': altLoc,
                'resName': resName,
                'chainID': chainID,
                'resSeq': resSeq,
                'iCode': iCode,
                'x': x,
                'y': y,
                'z': z,
                'occupancy': occupancy,
                'tempFactor': tempFactor,
                'element': element,
                'charge': charge
            }
            
        except (ValueError, IndexError) as e:
            # Handle malformed lines gracefully
            return None

    def _parse_conect_line(self, line: str) -> list:
        """Parse CONECT record for bond information."""
        try:
            parts = line.split()
            if len(parts) < 3:
                return []
            
            bonds = []
            atom_i = int(parts[1])
            
            # CONECT records can have multiple bonded atoms
            for j_str in parts[2:]:
                try:
                    atom_j = int(j_str)
                    # Add bond (ensure i < j for consistency)
                    if atom_i < atom_j:
                        bonds.append([atom_i, atom_j])
                    else:
                        bonds.append([atom_j, atom_i])
                except ValueError:
                    continue
            
            return bonds
            
        except (ValueError, IndexError):
            return []

    def read(self, frame):
        """Read PDB file and populate frame."""
        
        atoms_data = {
            "id": [],
            "name": [],
            "altLoc": [],
            "resName": [],
            "chainID": [],
            "resSeq": [],
            "iCode": [],
            "xyz": [],
            "occupancy": [],
            "tempFactor": [],
            "element": [],
            "charge": [],
            "record_type": []
        }
        
        bonds = []
        box_info = None
        
        try:
            with open(self._path, "r") as f:
                for line in f:
                    line = self.sanitizer(line)
                    
                    if not line:
                        continue
                    
                    if line.startswith("CRYST1"):
                        box_info = self._parse_cryst1(line)
                    
                    elif line.startswith(("ATOM", "HETATM")):
                        atom_data = self._parse_atom_line(line)
                        if atom_data:
                            atoms_data["id"].append(atom_data["id"])
                            atoms_data["name"].append(atom_data["name"])
                            atoms_data["altLoc"].append(atom_data["altLoc"])
                            atoms_data["resName"].append(atom_data["resName"])
                            atoms_data["chainID"].append(atom_data["chainID"])
                            atoms_data["resSeq"].append(atom_data["resSeq"])
                            atoms_data["iCode"].append(atom_data["iCode"])
                            atoms_data["xyz"].append([atom_data["x"], atom_data["y"], atom_data["z"]])
                            atoms_data["occupancy"].append(atom_data["occupancy"])
                            atoms_data["tempFactor"].append(atom_data["tempFactor"])
                            atoms_data["element"].append(atom_data["element"])
                            atoms_data["charge"].append(atom_data["charge"])
                            atoms_data["record_type"].append(atom_data["record_type"])
                    
                    elif line.startswith("CONECT"):
                        bond_list = self._parse_conect_line(line)
                        bonds.extend(bond_list)
        
        except FileNotFoundError:
            # Re-raise FileNotFoundError for proper handling
            raise
        except Exception as e:
            # Handle file reading errors gracefully
            print(f"Warning: Error reading PDB file {self._path}: {e}")
        
        # Handle empty file case
        if not atoms_data["id"]:
            atoms_data = {"id": [], "name": [], "xyz": [], "element": []}
        
        # Remove duplicates from bonds and convert to numpy array
        if bonds:
            bonds = np.unique(np.array(bonds), axis=0)
        
        # Handle duplicate atom names
        if len(atoms_data["id"]) > 0 and len(set(atoms_data["name"])) != len(atoms_data["name"]):
            atom_name_counter = defaultdict(int)
            for i, name in enumerate(atoms_data["name"]):
                atom_name_counter[name] += 1
                if atom_name_counter[name] > 1:
                    atoms_data["name"][i] = f"{name}{atom_name_counter[name]}"
        
        # Convert lists to numpy arrays for consistency
        for key, values in atoms_data.items():
            if values:  # Only convert non-empty lists
                if key == "xyz":
                    atoms_data[key] = np.array(values, dtype=float)
                elif key in ["id", "resSeq"]:
                    atoms_data[key] = np.array(values, dtype=int)
                elif key in ["occupancy", "tempFactor"]:
                    atoms_data[key] = np.array(values, dtype=float)
                else:
                    # For string data, use proper string dtype
                    max_len = max(len(str(v)) for v in values) if values else 10
                    atoms_data[key] = np.array(values, dtype=f'U{max_len}')
        
        # Create dataset
        frame["atoms"] = _dict_to_dataset(atoms_data)
        
        # Set box information
        if box_info:
            frame.box = mp.Box(matrix=box_info['matrix'])
        else:
            frame.box = mp.Box()  # Default box
        
        # Add bonds if present
        if len(bonds) > 0:
            frame["bonds"] = _dict_to_dataset({
                "i": bonds[:, 0] - 1,  # Convert to 0-based indexing
                "j": bonds[:, 1] - 1
            })
        
        return frame


class PDBWriter(DataWriter):
    """
    Robust PDB file writer that creates properly formatted PDB files.
    
    Features:
    - Writes ATOM/HETATM records with proper formatting
    - Handles missing fields with sensible defaults
    - Writes CRYST1 records from box information
    - Writes CONECT records for bonds
    - Ensures PDB format compliance
    """

    def __init__(self, path: Path):
        super().__init__(path=path)

    def _format_atom_line(self, serial: int, atom_data: dict) -> str:
        """Format a single ATOM/HETATM line according to PDB v3.3 specifications.
        
        PDB Format v3.3 ATOM/HETATM Record:
        COLUMNS        DATA TYPE    FIELD          DEFINITION
        --------------------------------------------------------------------------------
         1 -  6        Record name  "ATOM  " or "HETATM"
         7 - 11        Integer      serial         Atom serial number.
        13 - 16        Atom         name           Atom name.
        17             Character    altLoc         Alternate location indicator.
        18 - 20        Residue name resName        Residue name.
        22             Character    chainID        Chain identifier.
        23 - 26        Integer      resSeq         Residue sequence number.
        27             AChar        iCode          Code for insertion of residues.
        31 - 38        Real(8.3)    x              Orthogonal coordinates for X in Angstroms.
        39 - 46        Real(8.3)    y              Orthogonal coordinates for Y in Angstroms.
        47 - 54        Real(8.3)    z              Orthogonal coordinates for Z in Angstroms.
        55 - 60        Real(6.2)    occupancy      Occupancy.
        61 - 66        Real(6.2)    tempFactor     Temperature factor.
        77 - 78        LString(2)   element        Element symbol, right-justified.
        79 - 80        LString(2)   charge         Charge on the atom.
        """
        
        # Extract data with defaults
        record_type = atom_data.get("record_type", "ATOM")
        atom_name = atom_data.get("name", "UNK")
        alt_loc = atom_data.get("altLoc", " ")
        res_name = atom_data.get("resName", "UNK")
        chain_id = atom_data.get("chainID", " ")
        res_seq = atom_data.get("resSeq", 1)
        i_code = atom_data.get("iCode", " ")
        
        # Coordinates
        x = atom_data.get("x", 0.0)
        y = atom_data.get("y", 0.0) 
        z = atom_data.get("z", 0.0)
        
        # Optional fields
        occupancy = atom_data.get("occupancy", 1.0)
        temp_factor = atom_data.get("tempFactor", 0.0)
        element = atom_data.get("element", "")
        charge = atom_data.get("charge", "")
        
        # Format according to PDB v3.3 specification
        # Columns 1-6: Record name, left-justified
        line = f"{record_type:<6s}"
        
        # Columns 7-11: Serial number, right-justified
        line += f"{serial:>5d}"
        
        # Column 12: Space
        line += " "
        
        # Columns 13-16: Atom name
        # For atom names: if 1 character, start at column 14; if 2+ characters, start at column 13
        if len(atom_name) == 1:
            line += f" {atom_name:<3s}"  # Space + 1 char + 2 spaces
        elif len(atom_name) <= 4:
            line += f"{atom_name:<4s}"   # Up to 4 characters
        else:
            line += f"{atom_name[:4]:<4s}"  # Truncate to 4 characters
        
        # Column 17: Alternate location indicator
        line += f"{alt_loc[0] if alt_loc else ' ':1s}"
        
        # Columns 18-20: Residue name, left-justified
        line += f"{res_name[:3]:<3s}"
        
        # Column 21: Space
        line += " "
        
        # Column 22: Chain identifier
        line += f"{chain_id[0] if chain_id else ' ':1s}"
        
        # Columns 23-26: Residue sequence number, right-justified
        line += f"{res_seq:>4d}"
        
        # Column 27: Insertion code
        line += f"{i_code[0] if i_code else ' ':1s}"
        
        # Columns 28-30: Spaces
        line += "   "
        
        # Columns 31-38: X coordinate, right-justified, 8.3 format
        line += f"{x:>8.3f}"
        
        # Columns 39-46: Y coordinate, right-justified, 8.3 format  
        line += f"{y:>8.3f}"
        
        # Columns 47-54: Z coordinate, right-justified, 8.3 format
        line += f"{z:>8.3f}"
        
        # Columns 55-60: Occupancy, right-justified, 6.2 format
        line += f"{occupancy:>6.2f}"
        
        # Columns 61-66: Temperature factor, right-justified, 6.2 format
        line += f"{temp_factor:>6.2f}"
        
        # Columns 67-76: Spaces (could contain segment identifier, but we'll use spaces)
        line += "          "
        
        # Columns 77-78: Element symbol, right-justified
        if element:
            line += f"{element[:2]:>2s}"
        else:
            line += "  "
        
        # Columns 79-80: Charge
        if charge:
            line += f"{charge[:2]:>2s}"
        else:
            line += "  "
        
        return line

    def _format_cryst1_line(self, box) -> str:
        """Format CRYST1 line from box information."""
        if box is None:
            return "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1"
        
        # Extract box parameters
        matrix = box.matrix
        a = float(matrix[0, 0])
        b = float(matrix[1, 1])
        c = float(matrix[2, 2])
        
        # For now, assume orthogonal (90 degree angles)
        alpha = beta = gamma = 90.0
        
        return f"CRYST1{a:>9.3f}{b:>9.3f}{c:>9.3f}{alpha:>7.2f}{beta:>7.2f}{gamma:>7.2f} P 1           1"

    def _get_atom_data_at_index(self, atoms_dataset, index: int) -> dict:
        """Extract atom data at given index from dataset."""
        atom_data = {}
        
        # Find the main atom dimension (should be 'atoms_id' or similar)
        main_dim = None
        for dim_name in atoms_dataset.dims:
            dim_str = str(dim_name)
            if dim_str.endswith('_id') or 'atom' in dim_str.lower():
                main_dim = dim_name
                break
        
        if main_dim is None:
            # Fallback to first dimension
            main_dim = list(atoms_dataset.dims.keys())[0] if atoms_dataset.dims else None
        
        if main_dim is None:
            return atom_data
            
        # Extract data for this atom
        for var_name in atoms_dataset.data_vars:
            values = atoms_dataset[var_name]
            
            if var_name == "xyz":
                # Handle coordinate array - should be (atoms_id, xyz_dim_1) or similar
                if main_dim in values.dims:
                    coord = values.isel({main_dim: index}).values
                    if len(coord) >= 3:
                        atom_data["x"] = float(coord[0])
                        atom_data["y"] = float(coord[1]) 
                        atom_data["z"] = float(coord[2])
            else:
                # Handle scalar values - use main dimension if present
                if main_dim in values.dims:
                    value = values.isel({main_dim: index}).values
                    if hasattr(value, 'item'):
                        value = value.item()
                    atom_data[var_name] = value
        
        return atom_data

    def write(self, frame):
        """Write frame to PDB file."""
        
        with open(self._path, "w") as f:
            # Write header
            frame_name = frame.get("name", "MOL")
            f.write(f"REMARK  {frame_name}\n")
            
            # Write CRYST1 record if box exists
            if hasattr(frame, 'box') and frame.box is not None:
                f.write(self._format_cryst1_line(frame.box) + "\n")
            else:
                f.write(self._format_cryst1_line(None) + "\n")
            
            # Write atoms
            if "atoms" in frame:
                atoms = frame["atoms"]
                sizes = atoms.sizes
                
                # Find the main atom dimension
                main_dim = None
                for dim_name in sizes.keys():
                    dim_str = str(dim_name)
                    if dim_str.endswith('_id') or 'atom' in dim_str.lower():
                        main_dim = dim_name
                        break
                        
                if main_dim is None:
                    main_dim = next(iter(sizes.keys())) if sizes else None
                    
                if main_dim:
                    n_atoms = sizes[main_dim]
                
                for i in range(n_atoms):
                    atom_data = self._get_atom_data_at_index(atoms, i)
                    serial = i + 1  # PDB uses 1-based indexing
                    
                    # Use atom ID if available, otherwise use serial
                    if "id" in atom_data:
                        display_serial = atom_data["id"]
                    else:
                        display_serial = serial
                    
                    line = self._format_atom_line(display_serial, atom_data)
                    f.write(line + "\n")
            
            # Write bonds as CONECT records
            if "bonds" in frame:
                bonds = frame["bonds"]
                bond_dict = defaultdict(list)
                
                # Find the main bond dimension
                bond_sizes = bonds.sizes
                bond_main_dim = None
                for dim_name in bond_sizes.keys():
                    dim_str = str(dim_name)
                    if dim_str.endswith('_id') or 'bond' in dim_str.lower():
                        bond_main_dim = dim_name
                        break
                        
                if bond_main_dim is None:
                    bond_main_dim = next(iter(bond_sizes.keys())) if bond_sizes else None
                
                if bond_main_dim and "i" in bonds.data_vars and "j" in bonds.data_vars:
                    n_bonds = bond_sizes[bond_main_dim]
                    
                    # Group bonds by first atom (convert back to 1-based)
                    for i in range(n_bonds):
                        atom_i = int(bonds["i"].isel({bond_main_dim: i}).values) + 1
                        atom_j = int(bonds["j"].isel({bond_main_dim: i}).values) + 1
                        
                        bond_dict[atom_i].append(atom_j)
                        bond_dict[atom_j].append(atom_i)
                    
                    # Write CONECT records
                    for atom_i, connected_atoms in bond_dict.items():
                        # Remove duplicates and limit to 4 connections per line
                        connected_atoms = sorted(set(connected_atoms))
                        
                        if len(connected_atoms) > 4:
                            # PDB format limitation - split into multiple CONECT records
                            for chunk_start in range(0, len(connected_atoms), 4):
                                chunk = connected_atoms[chunk_start:chunk_start + 4]
                                conect_line = f"CONECT{atom_i:>5d}" + "".join(f"{j:>5d}" for j in chunk)
                                f.write(conect_line + "\n")
                        else:
                            conect_line = f"CONECT{atom_i:>5d}" + "".join(f"{j:>5d}" for j in connected_atoms)
                            f.write(conect_line + "\n")
            
            # Write END record
            f.write("END\n")
