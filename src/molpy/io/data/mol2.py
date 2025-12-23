from pathlib import Path

import numpy as np

from molpy.core.frame import Frame
from molpy.core.element import Element

from .base import DataReader


class Mol2Reader(DataReader):
    """
    Robust MOL2 file reader following TRIPOS MOL2 format specification.

    Features:
    - Parses MOLECULE, ATOM, BOND, and SUBSTRUCTURE sections
    - Handles various atom types and bond types
    - Robust error handling for malformed files
    - Supports partial files with missing sections
    - Assigns atomic numbers from atom names/types
    """

    def __init__(self, file: str | Path):
        super().__init__(Path(file))
        self._file = Path(file)
        self.molecule_info = {}

    @staticmethod
    def sanitizer(line: str) -> str:
        """Clean up line by stripping whitespace."""
        return line.strip()

    def read(self, frame: Frame) -> Frame:
        """Read MOL2 file and populate frame."""

        try:
            with open(self._file) as f:
                lines = f.readlines()
        except OSError as e:
            raise ValueError(f"Cannot read MOL2 file {self._file}: {e}")

        self.atoms = []
        self.bonds = []
        self.molecule_info = {}

        section = None
        for line_num, line in enumerate(map(self.sanitizer, lines), 1):
            if not line:  # Skip empty lines
                continue

            if line.startswith("@<TRIPOS>"):
                section = line[9:].strip()
                continue

            try:
                if section == "MOLECULE":
                    self._parse_molecule_section(line, line_num)

                elif section == "ATOM":
                    self._parse_atom_section(line, line_num)

                elif section == "BOND":
                    self._parse_bond_section(line, line_num)

                elif section == "SUBSTRUCTURE":
                    # We can extend this to parse substructure info if needed
                    pass

            except (ValueError, IndexError) as e:
                print(
                    f"Warning: Error parsing line {line_num} in section {section}: {e}"
                )
                continue

        # Assign atomic numbers after parsing all atoms
        self._assign_atomic_numbers()

        # Build datasets
        if self.atoms:
            # Convert atom list to Frame Block structure
            atoms_dict = {}
            for key in self.atoms[0]:
                values = [atom[key] for atom in self.atoms]
                if key == "xyz":
                    # Convert tuples to separate x, y, z arrays
                    xyz_array = np.array(values)
                    atoms_dict["x"] = xyz_array[:, 0]
                    atoms_dict["y"] = xyz_array[:, 1]
                    atoms_dict["z"] = xyz_array[:, 2]
                else:
                    atoms_dict[key] = np.array(values)

            frame["atoms"] = atoms_dict

        if self.bonds:
            # Convert bond list to Frame Block structure
            bonds_dict = {}
            for key in self.bonds[0]:
                bonds_dict[key] = np.array([bond[key] for bond in self.bonds])
            frame["bonds"] = bonds_dict

        return frame

    def _parse_molecule_section(self, line: str, line_num: int) -> None:
        """Parse MOLECULE section lines."""
        if not hasattr(self, "_molecule_line_count"):
            self._molecule_line_count = 0

        self._molecule_line_count += 1

        if self._molecule_line_count == 1:
            # First line: molecule name
            self.molecule_info["name"] = line
        elif self._molecule_line_count == 2:
            # Second line: counts (atoms, bonds, subst, feat, sets)
            parts = line.split()
            if len(parts) >= 2:
                self.molecule_info["num_atoms"] = int(parts[0])
                self.molecule_info["num_bonds"] = int(parts[1])
        elif self._molecule_line_count == 3:
            # Third line: molecule type
            self.molecule_info["mol_type"] = line
        elif self._molecule_line_count == 4:
            # Fourth line: charge type
            self.molecule_info["charge_type"] = line

    def _parse_atom_section(self, line: str, line_num: int) -> None:
        """Parse an ATOM entry in the MOL2 file.

        MOL2 ATOM format:
        atom_id atom_name x y z atom_type [subst_id] [subst_name] [charge] [status_bit]
        """
        data = line.split()
        if len(data) < 6:
            raise ValueError(
                f"Insufficient atom data fields: expected at least 6, got {len(data)}"
            )

        try:
            index = int(data[0])
            name = data[1]

            # Coordinates - must be present
            x, y, z = float(data[2]), float(data[3]), float(data[4])
            xyz = (x, y, z)

            atom_type = data[5]

            # Optional fields with defaults
            subst_id = int(data[6]) if len(data) > 6 and data[6].strip() else 1
            subst_name = data[7] if len(data) > 7 and data[7].strip() else ""
            charge = float(data[8]) if len(data) > 8 and data[8].strip() else 0.0

            # Status bit (rarely used)
            status_bit = data[9] if len(data) > 9 else ""

            self.atoms.append(
                {
                    "id": index,
                    "name": name,
                    "xyz": xyz,
                    "type": atom_type,
                    "subst_id": subst_id,
                    "subst_name": subst_name,
                    "charge": charge,
                    "status_bit": status_bit,
                }
            )

        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing atom data: {e}")

    def _parse_bond_section(self, line: str, line_num: int) -> None:
        """Parse a BOND entry in the MOL2 file.

        MOL2 BOND format:
        bond_id origin_atom_id target_atom_id bond_type [status_bits]
        """
        data = line.split()
        if len(data) < 4:
            raise ValueError(
                f"Insufficient bond data fields: expected at least 4, got {len(data)}"
            )

        try:
            index = int(data[0])
            atom1 = int(data[1])
            atom2 = int(data[2])
            bond_type = data[3]

            # Optional status bits
            status_bits = " ".join(data[4:]) if len(data) > 4 else ""

            self.bonds.append(
                {
                    "id": index,
                    "i": atom1 - 1,
                    "j": atom2 - 1,  # Convert to zero-based index
                    "type": bond_type,
                    "status_bits": status_bits,
                }
            )

        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing bond data: {e}")

    def _assign_atomic_numbers(self) -> None:
        """Assign atomic numbers to all atoms based on name or type."""
        for atom in self.atoms:
            element_data = self._guess_atomic_number(atom["name"])
            atomic_number = element_data.number
            if atomic_number == 0:
                element_data = self._guess_atomic_number(atom["type"])
                atomic_number = element_data.number
            atom["number"] = atomic_number

    def _guess_atomic_number(self, name_or_type: str):
        """Guess the atomic number from atom name or type.

        Args:
            name_or_type: Atom name (e.g., 'C1', 'H') or type (e.g., 'c3', 'hc')

        Returns:
            ElementData object
        """
        if not name_or_type:
            return Element(0)  # Unknown element

        # Clean the name: remove numbers and special characters, keep only letters
        clean_name = "".join(c for c in name_or_type if c.isalpha())

        if not clean_name:
            return Element(0)

        # Try common element patterns
        clean_upper = clean_name.upper()

        # Handle common two-letter elements first
        if len(clean_upper) >= 2:
            two_letter = clean_upper[:2]
            if two_letter in [
                "BR",
                "CL",
                "FE",
                "MG",
                "CA",
                "ZN",
                "NI",
                "CU",
                "AL",
                "SI",
                "AR",
                "KR",
                "XE",
            ]:
                try:
                    return Element(two_letter)
                except KeyError:
                    pass

        # Try first letter only
        try:
            return Element(clean_upper[0])
        except KeyError:
            pass

        # If all else fails, try the original string
        try:
            return Element(clean_upper)
        except KeyError:
            return Element(0)  # Unknown element
