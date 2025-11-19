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


def _dict_to_block(data: dict[str, list[Any]]) -> Block:
    """Convert 'column -> list' dict into a Block of ndarrays."""
    blk = Block()
    for k, vals in data.items():
        if k == "xyz":
            blk[k] = np.asarray(vals, dtype=float)
        elif k in {"id", "resSeq"}:
            blk[k] = np.asarray(vals, dtype=int)
        else:
            # variable-width unicode for strings
            max_len = max(len(str(v)) for v in vals) if vals else 1
            blk[k] = np.asarray(vals, dtype=f"U{max_len}")
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
            atoms_data["xyz"] = coords
            frame["atoms"] = _dict_to_block(atoms_data)

        if unique_bonds:
            ij = np.array(sorted(unique_bonds), dtype=int) - 1  # zero-based
            frame["bonds"] = Block({"i": ij[:, 0], "j": ij[:, 1]})

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

        # Coordinates - try xyz first, then xyz, then x/y/z
        if "xyz" in atom_data:
            xyz = atom_data["xyz"]
            x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        else:
            x = atom_data.get("x")
            y = atom_data.get("y")
            z = atom_data.get("z")

        # Optional fields
        occupancy = atom_data.get("occupancy", 1.0)
        temp_factor = atom_data.get("tempFactor", 0.0)
        element = atom_data.get("element")
        atom_data.get("charge")

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
        line += f"{float(occupancy):>6.2f}"

        # Columns 61-66: Temperature factor, right-justified, 6.2 format
        line += f"{float(temp_factor):>6.2f}"

        # Columns 67-76: Spaces (could contain segment identifier, but we'll use spaces)
        line += "          "

        # Columns 77-78: Element symbol, right-justified
        if element:
            line += f"{element[:2]:>2s}"
        else:
            line += "  "

        # Columns 79-80: Charge
        # if charge:
        #     line += f"{str(charge):2s}"
        # else:
        #     line += "  "

        line += "\n"  # End with newline
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
        """Write frame to PDB file."""

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

            for i in range(n_atoms):
                atom_data = atoms[i]
                display_serial = int(atom_data["id"]) if "id" in atom_data else i + 1
                line = self._format_atom_line(display_serial, atom_data)
                f.write(line)
            f.write("\n")

            # Write bonds as CONECT records
            if "bonds" in frame:
                bonds = frame["bonds"]
                connect = defaultdict(list)
                for i, j in zip(bonds["i"].tolist(), bonds["j"].tolist()):
                    itom = atoms[i]
                    jtom = atoms[j]
                    i_id = itom.get("id", i + 1)
                    j_id = jtom.get("id", j + 1)
                    connect[i_id].append(j_id)
                    connect[j_id].append(i_id)
                for i_id, j_ids in connect.items():
                    js = [str(j_id).rjust(5) for j_id in j_ids]
                    f.write(f"CONECT{str(i_id).rjust(5)}{''.join(js)}\n")
            # Write END record
            f.write("END\n")
