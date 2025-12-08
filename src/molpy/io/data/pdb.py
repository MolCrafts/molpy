import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from molpy.core import Block, Box, Frame

from .base import DataReader, DataWriter

# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────
_ALNUM_RE = re.compile(r"([A-Za-z]+)")

_TWO_CHR_ELEMENTS = {
    "BR",
    "CL",
    "FE",
    "MG",
    "CA",
    "ZN",
    "NI",
    "CU",
    "NA",
    "SI",
    "CR",
}


def _dict_to_block(data: dict[str, list[Any] | np.ndarray]) -> Block:
    """Convert 'column -> list/array' dict into a Block of ndarrays.

    Args:
        data: Dictionary where values can be lists or numpy arrays

    Returns:
        Block with all values converted to numpy arrays
    """
    blk = Block()
    for k, vals in data.items():
        # If already a numpy array, use it directly (after ensuring it's the right dtype)
        if isinstance(vals, np.ndarray):
            if k == "xyz":
                blk[k] = vals.astype(float)
            elif k in {"id", "resSeq"}:
                blk[k] = vals.astype(int)
            else:
                # Keep as is, Block will handle dtype conversion if needed
                blk[k] = vals
        else:
            # Convert from list
            if k == "xyz":
                blk[k] = np.asarray(vals, dtype=float)
            elif k in {"id", "resSeq"}:
                blk[k] = np.asarray(vals, dtype=int)
            else:
                # variable-width unicode for strings
                # Handle empty list case
                if not vals:
                    blk[k] = np.array([], dtype="U1")
                else:
                    # Convert list to array, handling mixed types
                    try:
                        max_len = (
                            max(len(str(v)) for v in vals if v is not None)
                            if vals
                            else 1
                        )
                        blk[k] = np.asarray(vals, dtype=f"U{max_len}")
                    except (TypeError, ValueError):
                        # Fallback: use object dtype if conversion fails
                        blk[k] = np.asarray(vals, dtype=object)
    return blk


# ──────────────────────────────────────────────────────────────────────
# main reader
# ──────────────────────────────────────────────────────────────────────
class PDBReader(DataReader):
    """
    Minimal-yet-robust PDB reader.

    * ATOM / HETATM parsed per PDB v3.3 fixed columns
    * CRYST1 -> frame.box
    * CONECT -> bond list
    """

    __slots__ = ()

    # ------------------------------------------------------------------
    def __init__(self, file: Path, **kwargs):
        super().__init__(path=file, **kwargs)

    # ------------------------------------------------------------------ private parsers
    @staticmethod
    def _parse_cryst1(line: str) -> np.ndarray | None:
        try:
            a, b, c = map(float, (line[6:15], line[15:24], line[24:33]))
            alpha, beta, gamma = map(float, (line[33:40], line[40:47], line[47:54]))
        except ValueError:
            return None

        # Orthorhombic shortcut
        if all(abs(x - 90) < 1e-3 for x in (alpha, beta, gamma)):
            return np.diag([a, b, c])

        # Triclinic conversion
        alpha_r, beta_r, gamma_r = np.deg2rad([alpha, beta, gamma])
        cos_alpha, cos_beta, cos_gamma = np.cos([alpha_r, beta_r, gamma_r])
        sin_gamma = np.sin(gamma_r)

        v_x = [a, 0.0, 0.0]
        v_y = [b * cos_gamma, b * sin_gamma, 0.0]
        cx = c * cos_beta
        cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
        cz = np.sqrt(c**2 - cx**2 - cy**2)
        v_z = [cx, cy, cz]

        return np.array([v_x, v_y, v_z])

    # ..................................................................
    @staticmethod
    def _parse_atom(line: str) -> tuple[dict[str, Any], np.ndarray]:
        serial = int(line[6:11])
        name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain = line[21:22] or " "
        res_seq = int(line[22:26])
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        occ_str = line[54:60].strip()
        occ = float(occ_str) if occ_str else 1.0
        bfac = float(line[60:66].strip() or 0.0)
        elem = (line[76:78] or "").strip().upper()

        if not elem:
            m = _ALNUM_RE.match(name)
            guess = m.group(1).upper() if m else ""
            elem = guess[:2] if guess[:2] in _TWO_CHR_ELEMENTS else guess[:1]

        return (
            {
                "id": serial,
                "name": name,
                "resName": res_name,
                "chainID": chain,
                "resSeq": res_seq,
                "occupancy": occ,
                "tempFactor": bfac,
                "element": elem,
            },
            xyz,
        )

    # ..................................................................
    @staticmethod
    def _parse_conect(line: str) -> list[tuple[int, int]]:
        center = int(line[6:11])
        bonds = []
        for offset in range(11, len(line), 5):
            seg = line[offset : offset + 5].strip()
            if seg:
                partner = int(seg)
                bonds.append(tuple(sorted((center, partner))))
        return bonds

    # ------------------------------------------------------------------ public read()
    def read(self, frame: Frame | None = None) -> Frame:
        frame = frame or Frame()

        atoms_data: dict[str, list[Any]] = defaultdict(list)
        coords: list[np.ndarray] = []
        unique_bonds: set[tuple[int, int]] = set()
        box_matrix: np.ndarray | None = None

        for raw in self:
            if raw.startswith("CRYST1"):
                box_matrix = self._parse_cryst1(raw)
            elif raw.startswith(("ATOM  ", "HETATM")):
                info, xyz = self._parse_atom(raw)
                for k, v in info.items():
                    atoms_data[k].append(v)
                coords.append(xyz)
            elif raw.startswith("CONECT"):
                unique_bonds.update(self._parse_conect(raw))

        # ---------- commit to frame ------------------------------------
        if coords:
            # Store coordinates as separate x, y, z fields only
            coords_array = np.array(coords)
            atoms_data["x"] = coords_array[:, 0]
            atoms_data["y"] = coords_array[:, 1]
            atoms_data["z"] = coords_array[:, 2]
            frame["atoms"] = _dict_to_block(atoms_data)

        if unique_bonds:
            # Keep atom IDs as 1-based (no conversion)
            # Use atom1, atom2 field names
            bonds_array = np.array(sorted(unique_bonds), dtype=int)
            frame["bonds"] = Block(
                {"atom1": bonds_array[:, 0], "atom2": bonds_array[:, 1]}
            )

        frame.metadata["box"] = (
            Box(matrix=box_matrix) if box_matrix is not None else Box()
        )
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

    def _format_atom_line_fast(
        self,
        serial: int,
        atom_name: str,
        res_name: str,
        chain_id: str,
        res_seq: int,
        x: float,
        y: float,
        z: float,
        occupancy: float,
        temp_factor: float,
        element: str,
    ) -> str:
        """Fast version of _format_atom_line that takes individual parameters."""
        # Format according to PDB v3.3 specification
        line = "ATOM  "  # Columns 1-6: Record name

        # Columns 7-11: Serial number, right-justified
        line += f"{serial:>5d}"

        # Column 12: Space
        line += " "

        # Columns 13-16: Atom name
        if len(atom_name) == 1:
            line += f" {atom_name:<3s}"  # Space + 1 char + 2 spaces
        elif len(atom_name) <= 4:
            line += f"{atom_name:<4s}"  # Up to 4 characters
        else:
            line += f"{atom_name[:4]:<4s}"  # Truncate to 4 characters

        # Column 17: Alternate location indicator (space)
        line += " "

        # Columns 18-20: Residue name, left-justified
        line += f"{res_name[:3]:<3s}"

        # Column 21: Space
        line += " "

        # Column 22: Chain identifier
        line += f"{chain_id[0] if chain_id else ' ':1s}"

        # Columns 23-26: Residue sequence number, right-justified
        line += f"{res_seq:>4d}"

        # Column 27: Insertion code (space)
        line += " "

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

        # Columns 67-76: Spaces
        line += "          "

        # Columns 77-78: Element symbol, right-justified
        elem_str = str(element).upper()[:2] if element else "  "
        line += f"{elem_str:>2s}"

        # Columns 79-80: Charge (optional, use spaces if not provided)
        line += "  "

        # Ensure line is exactly 79 characters before adding newline
        if len(line) < 79:
            line = line.ljust(79)
        elif len(line) > 79:
            line = line[:79]

        # End with newline (total 80 characters: 79 content + 1 newline)
        line += "\n"
        return line

    def _format_atom_line(self, serial: int, atom_data: Block) -> str:
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
        record_type = str(atom_data.get("record_type", "ATOM"))
        atom_name = str(atom_data.get("name", "UNK"))
        alt_loc = str(atom_data.get("altLoc", " "))
        res_name = str(atom_data.get("resName", "UNK"))
        chain_id = str(atom_data.get("chainID", " "))
        res_seq = str(atom_data.get("resSeq", 1))
        i_code = str(atom_data.get("iCode", " "))

        # Coordinates - must use separate x, y, z fields
        x = float(atom_data["x"])
        y = float(atom_data["y"])
        z = float(atom_data["z"])

        # Optional fields
        occupancy = float(atom_data.get("occupancy", 1.0))
        temp_factor = float(atom_data.get("tempFactor", 0.0))
        element = str(atom_data.get("element"))

        # Format according to PDB v3.3 specification
        # Columns 1-6: Record name, left-justified
        line = f"{record_type:<6s}"

        # Columns 7-11: Serial number, right-justified
        line += f"{serial:>5d}"

        # Column 12: Space
        line += " "

        # Columns 13-16: Atom name
        # For atom names: if 1 character, start at column 14; if 2+ characters, start at column 13
        atom_name = str(atom_name)
        if len(atom_name) == 1:
            line += f" {atom_name:<3s}"  # Space + 1 char + 2 spaces
        elif len(atom_name) <= 4:
            line += f"{atom_name:<4s}"  # Up to 4 characters
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
        line += f"{res_seq:>4s}"

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
        line += f"{float(occupancy):>6.2f}"

        # Columns 61-66: Temperature factor, right-justified, 6.2 format
        line += f"{float(temp_factor):>6.2f}"

        # Columns 67-76: Spaces (could contain segment identifier, but we'll use spaces)
        line += "          "

        # Columns 77-78: Element symbol, right-justified (2 characters)
        if element:
            elem_str = str(element)[:2].upper().strip()
            line += f"{elem_str:>2s}"
        else:
            line += "  "

        # Columns 79-80: Charge (optional, use spaces if not provided)
        # Charge is typically not provided, so use spaces (2 characters)
        line += "  "

        # At this point, line should be exactly 79 characters
        # Verify and fix if needed
        if len(line) < 79:
            line = line.ljust(79)
        elif len(line) > 79:
            line = line[:79]

        # Add newline to make total 80 characters
        line += "\n"
        return line

    def _format_cryst1_line(self, box) -> str:
        """Format CRYST1 line from box information."""
        if box is None:
            return (
                "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1"
            )

        # Extract box parameters
        matrix = box.matrix
        a = float(matrix[0, 0])
        b = float(matrix[1, 1])
        c = float(matrix[2, 2])

        # For now, assume orthogonal (90 degree angles)
        alpha = beta = gamma = 90.0

        return f"CRYST1{a:>9.3f}{b:>9.3f}{c:>9.3f}{alpha:>7.2f}{beta:>7.2f}{gamma:>7.2f} P 1           1"

    def write(self, frame):
        """Write frame to PDB file.

        Required fields in frame["atoms"]:
        - x, y, z: coordinates (float, required)
        - id: atom ID (int, optional, defaults to index+1)

        Optional fields in frame["atoms"]:
        - name: atom name (str)
        - resName: residue name (str)
        - element: element symbol (str)
        - resSeq: residue sequence number (int)
        - chainID: chain identifier (str)
        - occupancy: occupancy (float)
        - tempFactor: temperature factor (float)

        Optional metadata:
        - elements: space-separated string of element symbols (one per atom)
        - name: frame name (str)
        - box: Box object for CRYST1 record

        Raises:
            ValueError: If required fields (x, y, z) are missing or contain None
        """
        # Extract elements from metadata if available
        elements_list = None
        if "elements" in frame.metadata:
            elements_str = frame.metadata["elements"]
            if isinstance(elements_str, str):
                elements_list = elements_str.split()

        with open(self._path, "w") as f:
            # Write header
            frame_name = frame.metadata.get("name", "MOL")
            f.write(f"REMARK  {frame_name}\n")

            # Write CRYST1 record if box exists
            if "box" in frame.metadata and frame.metadata["box"] is not None:
                f.write(self._format_cryst1_line(frame.metadata["box"]) + "\n")
            else:
                f.write(self._format_cryst1_line(None) + "\n")

            atoms = frame["atoms"]
            n_atoms = atoms.nrows

            # Validate required fields exist and are not None
            required_fields = ["x", "y", "z"]
            for field in required_fields:
                if field not in atoms:
                    raise ValueError(
                        f"Required field '{field}' is missing in frame['atoms']"
                    )
                # Check if any values are None
                values = atoms[field]
                if values is None:
                    raise ValueError(f"Required field '{field}' contains None")
                # Check for None in array (if object dtype)
                if hasattr(values, "dtype") and values.dtype == object:
                    if any(v is None for v in values):
                        raise ValueError(
                            f"Required field '{field}' contains None values"
                        )

            for i in range(n_atoms):
                # Extract required fields - raise error if None
                x_val = atoms["x"][i]
                y_val = atoms["y"][i]
                z_val = atoms["z"][i]

                if x_val is None or y_val is None or z_val is None:
                    raise ValueError(
                        f"Required coordinate fields contain None at index {i}: "
                        f"x={x_val}, y={y_val}, z={z_val}"
                    )

                x = float(x_val)
                y = float(y_val)
                z = float(z_val)

                # Extract optional fields with defaults
                atom_id = (
                    int(atoms["id"][i])
                    if "id" in atoms and atoms["id"][i] is not None
                    else i + 1
                )

                # Get element from metadata list or atom_data
                element = None
                if elements_list and i < len(elements_list):
                    element = elements_list[i]
                elif "element" in atoms and atoms["element"][i] is not None:
                    element = str(atoms["element"][i])
                elif "symbol" in atoms and atoms["symbol"][i] is not None:
                    element = str(atoms["symbol"][i]).upper()
                else:
                    element = "X"  # Default unknown element

                # Get atom name (use element if not specified)
                atom_name = None
                if "name" in atoms and atoms["name"][i] is not None:
                    atom_name = str(atoms["name"][i])
                else:
                    atom_name = element  # Use element as fallback

                # Get residue name
                res_name = "UNK"
                if "resName" in atoms and atoms["resName"][i] is not None:
                    res_name = str(atoms["resName"][i])

                # Get residue sequence number
                res_seq = 1
                if "resSeq" in atoms and atoms["resSeq"][i] is not None:
                    res_seq = int(atoms["resSeq"][i])

                # Get chain ID
                chain_id = " "
                if "chainID" in atoms and atoms["chainID"][i] is not None:
                    chain_id = str(atoms["chainID"][i])[:1]

                # Get optional fields with defaults
                occupancy = 1.0
                if "occupancy" in atoms and atoms["occupancy"][i] is not None:
                    occupancy = float(atoms["occupancy"][i])

                temp_factor = 0.0
                if "tempFactor" in atoms and atoms["tempFactor"][i] is not None:
                    temp_factor = float(atoms["tempFactor"][i])

                # Format and write atom line
                line = self._format_atom_line_fast(
                    serial=atom_id,
                    atom_name=atom_name,
                    res_name=res_name,
                    chain_id=chain_id,
                    res_seq=res_seq,
                    x=x,
                    y=y,
                    z=z,
                    occupancy=occupancy,
                    temp_factor=temp_factor,
                    element=element,
                )
                f.write(line)
            f.write("\n")

            # Write bonds as CONECT records
            if "bonds" in frame:
                bonds = frame["bonds"]
                if "atom1" in bonds and "atom2" in bonds:
                    connect = defaultdict(list)
                    # atom1, atom2 are stored as atom IDs (1-based), use directly
                    for atom1_id, atom2_id in zip(
                        bonds["atom1"].tolist(), bonds["atom2"].tolist()
                    ):
                        atom1_id_int = int(atom1_id)
                        atom2_id_int = int(atom2_id)
                        connect[atom1_id_int].append(atom2_id_int)
                        connect[atom2_id_int].append(atom1_id_int)
                    for atom1_id, atom2_ids in connect.items():
                        js = [str(atom2_id).rjust(5) for atom2_id in atom2_ids]
                        f.write(f"CONECT{str(atom1_id).rjust(5)}{''.join(js)}\n")
            # Write END record
            f.write("END\n")
