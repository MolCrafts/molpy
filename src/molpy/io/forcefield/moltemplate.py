"""
MolTemplate (.lt) file reader for MolPy.

This module provides functionality to read MolTemplate force field files
and convert them to MolPy ForceField objects, as well as molecular
structure definitions to Atomistic objects.

Supports both regular file paths and @-prefixed builtin paths:
- Regular: "path/to/file.lt"
- Builtin: "@forcefield/tip3p.lt"
"""

import re
from pathlib import Path
from typing import Any, cast

from molpy import (
    Atomistic,
    AtomType,
    ForceField,
)
from molpy.io.builtin_resolver import resolve_builtin_path


class MolTemplateReader:
    """
    Reader for MolTemplate (.lt) force field files.

    MolTemplate files define force field parameters in a hierarchical structure
    with sections for different interaction types, as well as molecular structures.
    """

    def __init__(self):
        self.current_section = None
        self.current_style = None
        self.molecules: list[dict[str, Any]] = []

    def read(self, file_path: str | Path) -> ForceField:
        """
        Read a MolTemplate .lt file and return a ForceField object.

        Args:
            file_path: Path to the .lt file (supports @-prefixed builtin paths)

        Returns:
            ForceField object with parsed parameters
        """
        # Resolve builtin paths if needed
        resolved_path = resolve_builtin_path(file_path)

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"MolTemplate file not found: {file_path} -> {resolved_path}"
            )

        # Initialize force field
        forcefield = ForceField(name=resolved_path.stem, units="real")

        with open(resolved_path) as f:
            content = f.read()

        # Parse the content
        self._parse_content(content, forcefield)

        return forcefield

    def read_molecule(
        self, file_path: str | Path, molecule_name: str | None = None
    ) -> Atomistic:
        """
        Read a MolTemplate .lt file and return a specific molecule as Atomistic object.

        Args:
            file_path: Path to the .lt file (supports @-prefixed builtin paths)
            molecule_name: Name of the molecule to extract (if None, returns first molecule)

        Returns:
            Atomistic object representing the molecule
        """
        # Resolve builtin paths if needed
        resolved_path = resolve_builtin_path(file_path)

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"MolTemplate file not found: {file_path} -> {resolved_path}"
            )

        # First read the force field to get atom types
        forcefield = self.read(file_path)

        # Find the specified molecule or first available
        if molecule_name is None and self.molecules:
            molecule_data = self.molecules[0]
        else:
            molecule_data = next(
                (m for m in self.molecules if m["name"] == molecule_name), None
            )
            if molecule_data is None:
                name_str = molecule_name if molecule_name is not None else "None"
                raise ValueError(f"Molecule '{name_str}' not found in file")

        # Convert to Atomistic object
        return self._create_atomistic_from_molecule(molecule_data, forcefield)

    def read_all_molecules(self, file_path: str | Path) -> list[Atomistic]:
        """
        Read all molecules from a MolTemplate .lt file.

        Args:
            file_path: Path to the .lt file (supports @-prefixed builtin paths)

        Returns:
            List of Atomistic objects representing all molecules
        """
        # Resolve builtin paths if needed
        resolved_path = resolve_builtin_path(file_path)

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"MolTemplate file not found: {file_path} -> {resolved_path}"
            )

        # First read the force field to get atom types
        forcefield = self.read(file_path)

        # Convert all molecules to Atomistic objects
        atomistic_molecules = []
        for molecule_data in self.molecules:
            atomistic_mol = self._create_atomistic_from_molecule(
                molecule_data, forcefield
            )
            atomistic_molecules.append(atomistic_mol)

        return atomistic_molecules

    def _parse_content(self, content: str, forcefield: ForceField):
        """Parse the content of a .lt file."""
        lines = content.split("\n")

        for line in lines:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse section headers
            if self._is_section_header(line):
                self._parse_section_header(line, forcefield)
            else:
                # Parse content within sections
                self._parse_line(line, forcefield)

    def _is_section_header(self, line: str) -> bool:
        """Check if a line is a section header."""
        # Section headers typically start with keywords like "write_once", "write", etc.
        section_keywords = [
            "write_once",
            "write",
            "import",
            "create_var",
            "delete_var",
            "set",
            "if",
            "while",
            "for",
            "foreach",
            "include",
        ]

        return any(line.startswith(keyword) for keyword in section_keywords)

    def _parse_section_header(self, line: str, forcefield: ForceField):
        """Parse a section header and set the current section."""
        # Extract section type and name
        if line.startswith("write_once"):
            # Force field parameters section
            self.current_section = "force_field"
            self._parse_write_once_section(line, forcefield)
        elif line.startswith("write"):
            # Output section - may contain molecular structure
            self.current_section = "output"
            self._parse_write_section(line, forcefield)
        elif line.startswith("import"):
            # Import section
            self.current_section = "import"
        else:
            self.current_section = "unknown"

    def _parse_write_once_section(self, line: str, forcefield: ForceField):
        """Parse a write_once section which typically contains force field parameters."""
        # Extract the section name/content
        match = re.match(r'write_once\s*\(\s*["\']([^"\']+)["\']\s*\)\s*{', line)
        if match:
            section_name = match.group(1)
            self.current_style = section_name
            self.current_section = "force_field"  # Set the section type

            # Create appropriate style based on section name
            if "atom" in section_name.lower():
                forcefield.def_atomstyle(section_name)
            elif "bond" in section_name.lower():
                forcefield.def_bondstyle(section_name)
            elif "angle" in section_name.lower():
                forcefield.def_anglestyle(section_name)
            elif "dihedral" in section_name.lower():
                forcefield.def_dihedralstyle(section_name)
            elif "improper" in section_name.lower():
                forcefield.def_improperstyle(section_name)
            elif "pair" in section_name.lower():
                forcefield.def_pairstyle(section_name)

    def _parse_write_section(self, line: str, forcefield: ForceField):
        """Parse a write section which may contain molecular structure."""
        # Extract the section name/content
        match = re.match(r'write\s*\(\s*["\']([^"\']+)["\']\s*\)\s*{', line)
        if match:
            section_name = match.group(1)
            self.current_style = section_name

            # Check if this is a molecular structure section
            if any(
                keyword in section_name.lower()
                for keyword in ["data", "molecule", "atoms", "bonds"]
            ):
                self.current_section = "molecular_structure"
            else:
                # Keep as output section if not molecular structure
                self.current_section = "output"

    def _parse_line(self, line: str, forcefield: ForceField):
        """Parse a line within a section."""
        if self.current_section == "force_field" and self.current_style:
            self._parse_force_field_line(line, forcefield)
        elif self.current_section == "molecular_structure":
            self._parse_molecular_structure_line(line, forcefield)

    def _parse_force_field_line(self, line: str, forcefield: ForceField):
        """Parse a line containing force field parameters."""
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            return

        # Try to parse different parameter types
        if (
            self._parse_atom_type_line(line, forcefield)
            or self._parse_bond_type_line(line, forcefield)
            or self._parse_angle_type_line(line, forcefield)
            or self._parse_dihedral_type_line(line, forcefield)
            or self._parse_pair_type_line(line, forcefield)
        ):
            return

    def _parse_molecular_structure_line(self, line: str, forcefield: ForceField):
        """Parse a line containing molecular structure information."""
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            return

        # Try to parse different structure elements
        if (
            self._parse_atom_line(line, forcefield)
            or self._parse_bond_line(line, forcefield)
            or self._parse_angle_line(line, forcefield)
            or self._parse_dihedral_line(line, forcefield)
        ):
            return

    def _parse_atom_line(self, line: str, forcefield: ForceField) -> bool:
        """Parse a line defining an atom in molecular structure."""
        # Pattern: atom_id atom_type x y z [optional_params]
        # Example: 1 @atom:OW 0.0 0.0 0.0
        pattern = r"(\d+)\s+@atom:(\w+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"
        match = re.search(pattern, line)

        if match:
            atom_id, atom_type, x, y, z = match.groups()

            # Store atom information for later molecule creation
            if not hasattr(self, "current_molecule"):
                self.current_molecule = {
                    "name": "molecule",
                    "atoms": [],
                    "bonds": [],
                    "angles": [],
                    "dihedrals": [],
                }
                self.molecules.append(self.current_molecule)

            self.current_molecule["atoms"].append(
                {
                    "id": int(atom_id),
                    "type": atom_type,
                    "xyz": [float(x), float(y), float(z)],
                }
            )
            return True

        return False

    def _parse_bond_line(self, line: str, forcefield: ForceField) -> bool:
        """Parse a line defining a bond in molecular structure."""
        # Pattern: bond_id bond_type atom1_id atom2_id
        # Example: 1 @bond:OW-HW 1 2
        pattern = r"^(\d+)\s+@bond:([\w-]+)\s+(\d+)\s+(\d+)$"
        match = re.search(pattern, line)

        if match:
            bond_id, bond_type, atom1_id, atom2_id = match.groups()

            if hasattr(self, "current_molecule"):
                self.current_molecule["bonds"].append(
                    {
                        "id": int(bond_id),
                        "type": bond_type,
                        "atoms": [int(atom1_id), int(atom2_id)],
                    }
                )
            return True

        return False

    def _parse_angle_line(self, line: str, forcefield: ForceField) -> bool:
        """Parse a line defining an angle in molecular structure."""
        # Pattern: angle_id angle_type atom1_id atom2_id atom3_id
        # Example: 1 @angle:HW-OW-HW 2 1 3
        pattern = r"^(\d+)\s+@angle:([\w-]+)\s+(\d+)\s+(\d+)\s+(\d+)$"
        match = re.search(pattern, line)

        if match:
            angle_id, angle_type, atom1_id, atom2_id, atom3_id = match.groups()

            if hasattr(self, "current_molecule"):
                self.current_molecule["angles"].append(
                    {
                        "id": int(angle_id),
                        "type": angle_type,
                        "atoms": [int(atom1_id), int(atom2_id), int(atom3_id)],
                    }
                )
            return True

        return False

    def _parse_dihedral_line(self, line: str, forcefield: ForceField) -> bool:
        """Parse a line defining a dihedral in molecular structure."""
        # Pattern: dihedral_id dihedral_type atom1_id atom2_id atom3_id atom4_id
        # Example: 1 @dihedral:HW-OW-HW-HW 2 1 3 4
        pattern = r"(\d+)\s+@dihedral:(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
        match = re.search(pattern, line)

        if match:
            dihedral_id, dihedral_type, atom1_id, atom2_id, atom3_id, atom4_id = (
                match.groups()
            )

            if hasattr(self, "current_molecule"):
                self.current_molecule["dihedrals"].append(
                    {
                        "id": int(dihedral_id),
                        "type": dihedral_type,
                        "atoms": [
                            int(atom1_id),
                            int(atom2_id),
                            int(atom3_id),
                            int(atom4_id),
                        ],
                    }
                )
            return True

        return False

    def _create_atomistic_from_molecule(
        self, molecule_data: dict[str, Any], forcefield: ForceField
    ) -> Atomistic:
        """Create an Atomistic object from parsed molecule data."""
        molecule_name = molecule_data.get("name", "molecule")
        molecule = Atomistic(name=molecule_name)

        # Create atoms
        atom_map = {}  # id -> atom object mapping
        for atom_info in molecule_data["atoms"]:
            atom = molecule.def_atom(
                name=f"atom_{atom_info['id']}",
                element=self._get_element_from_type(atom_info["type"]),
                xyz=atom_info["xyz"],
            )
            atom["type"] = atom_info["type"]
            atom_map[atom_info["id"]] = atom

        # Create bonds
        for bond_info in molecule_data["bonds"]:
            atom1 = atom_map[bond_info["atoms"][0]]
            atom2 = atom_map[bond_info["atoms"][1]]
            bond = molecule.def_bond(i=atom1, j=atom2)
            bond["type"] = bond_info["type"]

        # Create angles
        for angle_info in molecule_data["angles"]:
            atom1 = atom_map[angle_info["atoms"][0]]
            atom2 = atom_map[angle_info["atoms"][1]]
            atom3 = atom_map[angle_info["atoms"][2]]
            angle = molecule.def_angle(i=atom1, j=atom2, k=atom3)
            angle["type"] = angle_info["type"]

        # Create dihedrals
        for dihedral_info in molecule_data["dihedrals"]:
            atom1 = atom_map[dihedral_info["atoms"][0]]
            atom2 = atom_map[dihedral_info["atoms"][1]]
            atom3 = atom_map[dihedral_info["atoms"][2]]
            atom4 = atom_map[dihedral_info["atoms"][3]]
            dihedral = molecule.def_dihedral(i=atom1, j=atom2, k=atom3, l=atom4)
            dihedral["type"] = dihedral_info["type"]

        return molecule

    def _get_element_from_type(self, atom_type: str) -> str:
        """Get element symbol from atom type name."""
        # Simple mapping - could be enhanced with more sophisticated logic
        element_map = {
            "OW": "O",  # Oxygen in water
            "HW": "H",  # Hydrogen in water
            "CW": "C",  # Carbon in water
            "NW": "N",  # Nitrogen in water
            "SW": "S",  # Sulfur in water
            "PW": "P",  # Phosphorus in water
        }

        # Try to extract element from type name
        for type_prefix, element in element_map.items():
            if atom_type.startswith(type_prefix):
                return element

        # Default: try to extract first letter as element
        if atom_type and atom_type[0].isupper():
            return atom_type[0]

        return "X"  # Unknown element

    def _parse_atom_type_line(self, line: str, forcefield: ForceField) -> bool:
        """Parse a line defining an atom type."""
        # Pattern: @atom:atom_type_name (standalone line)
        pattern = r"^@atom:(\w+)$"
        match = re.match(pattern, line.strip())

        if match:
            name = match.group(1)

            # Get or create atom style
            atomstyle = forcefield.get_atomstyle("full")
            if not atomstyle:
                atomstyle = forcefield.def_atomstyle("full")

            # Create atom type with default values
            atomstyle.def_type(
                name=name,
                mass=1.0,
                charge=0.0,  # Default mass  # Default charge
            )
            return True

        return False

    def _get_or_create_atom_type(
        self, forcefield: ForceField, atom_type_name: str
    ) -> AtomType:
        """Get an existing atom type or create a new one."""
        atomstyle = forcefield.get_atomstyle("full")
        if not atomstyle:
            atomstyle = forcefield.def_atomstyle("full")

        atom_type = atomstyle.get(atom_type_name)
        if not atom_type:
            atom_type = atomstyle.def_type(atom_type_name)

        return cast(AtomType, atom_type)

    def _parse_bond_type_line(self, line: str, forcefield: ForceField) -> bool:
        """Parse a line defining a bond type."""
        # Pattern: @bond:bond_type_name @atom:atom1_type @atom:atom2_type k r0
        # Note: bond_type_name can contain hyphens (e.g., OW-HW)
        pattern = r"^@bond:([\w-]+)\s+@atom:(\w+)\s+@atom:(\w+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)$"
        match = re.match(pattern, line.strip())

        if match:
            name, atom1_type, atom2_type, k, r0 = match.groups()

            # Get or create bond style
            bondstyle = forcefield.get_bondstyle("harmonic")
            if not bondstyle:
                bondstyle = forcefield.def_bondstyle("harmonic")

            # Get atom types
            atom1 = self._get_or_create_atom_type(forcefield, atom1_type)
            atom2 = self._get_or_create_atom_type(forcefield, atom2_type)

            # Create bond type
            bondstyle.def_type(
                itype=atom1,
                jtype=atom2,
                name=name,
                force_constant=float(k),
                equilibrium_length=float(r0),
            )
            return True

        return False

    def _parse_angle_type_line(self, line: str, forcefield: ForceField) -> bool:
        """Parse a line defining an angle type."""
        # Pattern: @angle:angle_type_name @atom:atom1_type @atom:atom2_type @atom:atom3_type k theta0
        # Note: angle_type_name can contain hyphens (e.g., HW-OW-HW)
        pattern = r"^@angle:([\w-]+)\s+@atom:(\w+)\s+@atom:(\w+)\s+@atom:(\w+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)$"
        match = re.match(pattern, line.strip())

        if match:
            name, atom1_type, atom2_type, atom3_type, k, theta0 = match.groups()

            # Get or create angle style
            anglestyle = forcefield.get_anglestyle("harmonic")
            if not anglestyle:
                anglestyle = forcefield.def_anglestyle("harmonic")

            # Get atom types
            atom1 = self._get_or_create_atom_type(forcefield, atom1_type)
            atom2 = self._get_or_create_atom_type(forcefield, atom2_type)
            atom3 = self._get_or_create_atom_type(forcefield, atom3_type)

            # Create angle type
            anglestyle.def_type(
                itype=atom1,
                jtype=atom2,
                ktype=atom3,
                name=name,
                force_constant=float(k),
                equilibrium_angle=float(theta0),
            )
            return True

        return False

    def _parse_dihedral_type_line(self, line: str, forcefield: ForceField) -> bool:
        """Parse a line defining a dihedral type."""
        # Pattern: @dihedral:dihedral_type_name @atom:atom1_type @atom:atom2_type @atom:atom3_type @atom:atom4_type k n delta
        pattern = r"@dihedral:(\w+)\s+@atom:(\w+)\s+@atom:(\w+)\s+@atom:(\w+)\s+@atom:(\w+)\s+(-?\d+\.?\d*)\s+(\d+)\s+(-?\d+\.?\d*)"
        match = re.search(pattern, line)

        if match:
            name, atom1_type, atom2_type, atom3_type, atom4_type, k, n, delta = (
                match.groups()
            )

            # Get or create dihedral style
            dihedralstyle = forcefield.get_dihedralstyle("harmonic")
            if not dihedralstyle:
                dihedralstyle = forcefield.def_dihedralstyle("harmonic")

            # Get atom types
            atom1 = self._get_or_create_atom_type(forcefield, atom1_type)
            atom2 = self._get_or_create_atom_type(forcefield, atom2_type)
            atom3 = self._get_or_create_atom_type(forcefield, atom3_type)
            atom4 = self._get_or_create_atom_type(forcefield, atom4_type)

            # Create dihedral type
            dihedralstyle.def_type(
                itype=atom1,
                jtype=atom2,
                ktype=atom3,
                ltype=atom4,
                name=name,
                force_constant=float(k),
                multiplicity=int(n),
                phase=float(delta),
            )
            return True

        return False

    def _parse_pair_type_line(self, line: str, forcefield: ForceField) -> bool:
        """Parse a line defining a pair type."""
        # Pattern: @pair:pair_type_name @atom:atom_type epsilon sigma
        pattern = r"@pair:(\w+)\s+@atom:(\w+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"
        match = re.search(pattern, line)

        if match:
            name, atom_type, epsilon, sigma = match.groups()

            # Get or create pair style
            pairstyle = forcefield.get_pairstyle("lj")
            if not pairstyle:
                pairstyle = forcefield.def_pairstyle("lj")

            # Get atom type
            atom = self._get_or_create_atom_type(forcefield, atom_type)

            # Create pair type
            pairstyle.def_type(
                itype=atom,
                jtype=atom,
                name=name,
                epsilon=float(epsilon),
                sigma=float(sigma),
            )
            return True

        return False


def read_moltemplate(file_path: str | Path) -> ForceField:
    """
    Convenience function to read a MolTemplate .lt file.

    Args:
        file_path: Path to the .lt file (supports @-prefixed builtin paths)

    Returns:
        ForceField object with parsed parameters
    """
    reader = MolTemplateReader()
    return reader.read(file_path)


def read_moltemplate_molecule(
    file_path: str | Path, molecule_name: str | None = None
) -> Atomistic:
    """
    Convenience function to read a molecule from a MolTemplate .lt file.

    Args:
        file_path: Path to the .lt file (supports @-prefixed builtin paths)
        molecule_name: Name of the molecule to extract (if None, returns first molecule)

    Returns:
        Atomistic object representing the molecule
    """
    reader = MolTemplateReader()
    return reader.read_molecule(file_path, molecule_name)


def read_moltemplate_molecules(file_path: str | Path) -> list[Atomistic]:
    """
    Convenience function to read all molecules from a MolTemplate .lt file.

    Args:
        file_path: Path to the .lt file (supports @-prefixed builtin paths)

    Returns:
        List of Atomistic objects representing all molecules
    """
    reader = MolTemplateReader()
    return reader.read_all_molecules(file_path)
