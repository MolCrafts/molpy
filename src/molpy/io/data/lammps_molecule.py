"""
LAMMPS molecule file I/O.

This module provides readers and writers for LAMMPS molecule template files,
supporting both native format and JSON format as described in the LAMMPS documentation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import molpy as mp

from .base import DataReader, DataWriter


class LammpsMoleculeReader(DataReader):
    """LAMMPS molecule file reader supporting both native and JSON formats."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(Path(path))
        self.is_json = self._path.suffix.lower() == ".json"

    def read(self, frame: mp.Frame | None = None) -> mp.Frame:
        """Read LAMMPS molecule file into a Frame."""
        frame = frame or mp.Frame()

        if self.is_json:
            return self._read_json_format(frame)
        else:
            return self._read_native_format(frame)

    def _read_json_format(self, frame: mp.Frame) -> mp.Frame:
        """Read JSON format molecule file."""
        with open(self._path) as f:
            data = json.load(f)

        # Validate required fields
        if data.get("format") != "molecule":
            raise ValueError("JSON file must have format='molecule'")

        # Store metadata
        frame.metadata.update(
            {
                "format": "lammps_molecule",
                "source_format": "json",
                "source_file": str(self._path),
                "title": data.get("title", ""),
                "units": data.get("units", "lj"),
                "revision": data.get("revision", 1),
            }
        )

        # Parse atoms section (required)
        if "types" not in data:
            raise ValueError("JSON molecule file must contain 'types' section")

        atoms_data = {}

        # Parse types (required)
        types_data = data["types"]["data"]
        n_atoms = len(types_data)
        atom_ids = []
        atom_types = []

        for atom_entry in types_data:
            atom_ids.append(int(atom_entry[0]))
            atom_types.append(atom_entry[1])

        atoms_data["id"] = np.array(atom_ids)
        atoms_data["type"] = np.array(atom_types)

        # Parse coordinates if present
        if "coords" in data:
            coords_data = data["coords"]["data"]
            positions = np.zeros((n_atoms, 3))
            coord_atom_ids = []

            for coord_entry in coords_data:
                atom_id = int(coord_entry[0])
                coord_atom_ids.append(atom_id)
                idx = atom_ids.index(atom_id)
                positions[idx] = [
                    float(coord_entry[1]),
                    float(coord_entry[2]),
                    float(coord_entry[3]),
                ]

            atoms_data["x"] = positions[:, 0]
            atoms_data["y"] = positions[:, 1]
            atoms_data["z"] = positions[:, 2]

        # Parse charges if present
        if "charges" in data:
            charges = np.zeros(n_atoms)
            for charge_entry in data["charges"]["data"]:
                atom_id = int(charge_entry[0])
                idx = atom_ids.index(atom_id)
                charges[idx] = float(charge_entry[1])
            atoms_data["q"] = charges

        # Parse masses if present
        if "masses" in data:
            masses = np.ones(n_atoms)  # default mass 1.0
            for mass_entry in data["masses"]["data"]:
                atom_id = int(mass_entry[0])
                idx = atom_ids.index(atom_id)
                masses[idx] = float(mass_entry[1])
            atoms_data["mass"] = masses

        # Parse molecule IDs if present
        if "molecule" in data:
            mol_ids = np.ones(n_atoms, dtype=int)  # default mol_id 1
            for mol_entry in data["molecule"]["data"]:
                atom_id = int(mol_entry[0])
                idx = atom_ids.index(atom_id)
                mol_ids[idx] = int(mol_entry[1])
            atoms_data["mol"] = mol_ids

        # Parse diameters if present
        if "diameters" in data:
            diameters = np.ones(n_atoms)  # default diameter 1.0
            for diam_entry in data["diameters"]["data"]:
                atom_id = int(diam_entry[0])
                idx = atom_ids.index(atom_id)
                diameters[idx] = float(diam_entry[1])
            atoms_data["diameter"] = diameters

        frame["atoms"] = mp.Block(atoms_data)

        # Parse connectivity sections
        self._parse_json_connectivity(data, frame, "bonds")
        self._parse_json_connectivity(data, frame, "angles")
        self._parse_json_connectivity(data, frame, "dihedrals")
        self._parse_json_connectivity(data, frame, "impropers")

        # Store molecule properties in metadata
        if "com" in data:
            frame.metadata["center_of_mass"] = np.array(data["com"])
        if "masstotal" in data:
            frame.metadata["total_mass"] = float(data["masstotal"])
        if "inertia" in data:
            frame.metadata["inertia"] = np.array(data["inertia"])

        return frame

    def _parse_json_connectivity(
        self, data: dict, frame: mp.Frame, section: str
    ) -> None:
        """Parse connectivity sections from JSON data."""
        if section not in data:
            return

        connectivity_data = data[section]["data"]
        if not connectivity_data:
            return

        if section == "bonds":
            bond_types = []
            atom1_list = []
            atom2_list = []

            for _i, bond_entry in enumerate(connectivity_data):
                bond_types.append(bond_entry[0])
                atom1_list.append(int(bond_entry[1]))
                atom2_list.append(int(bond_entry[2]))

            frame[section] = mp.Block(
                {
                    "id": np.arange(1, len(bond_types) + 1),
                    "type": np.array(bond_types),
                    "atom1": np.array(atom1_list),
                    "atom2": np.array(atom2_list),
                }
            )

        elif section == "angles":
            angle_types = []
            atom1_list = []
            atom2_list = []
            atom3_list = []

            for _i, angle_entry in enumerate(connectivity_data):
                angle_types.append(angle_entry[0])
                atom1_list.append(int(angle_entry[1]))
                atom2_list.append(int(angle_entry[2]))
                atom3_list.append(int(angle_entry[3]))

            frame[section] = mp.Block(
                {
                    "id": np.arange(1, len(angle_types) + 1),
                    "type": np.array(angle_types),
                    "atom1": np.array(atom1_list),
                    "atom2": np.array(atom2_list),
                    "atom3": np.array(atom3_list),
                }
            )

        elif section in ["dihedrals", "impropers"]:
            types_list = []
            atom1_list = []
            atom2_list = []
            atom3_list = []
            atom4_list = []

            for _i, entry in enumerate(connectivity_data):
                types_list.append(entry[0])
                atom1_list.append(int(entry[1]))
                atom2_list.append(int(entry[2]))
                atom3_list.append(int(entry[3]))
                atom4_list.append(int(entry[4]))

            frame[section] = mp.Block(
                {
                    "id": np.arange(1, len(types_list) + 1),
                    "type": np.array(types_list),
                    "atom1": np.array(atom1_list),
                    "atom2": np.array(atom2_list),
                    "atom3": np.array(atom3_list),
                    "atom4": np.array(atom4_list),
                }
            )

    def _read_native_format(self, frame: mp.Frame) -> mp.Frame:
        """Read native format molecule file."""
        lines = self._read_lines()
        if not lines:
            raise ValueError("Empty molecule file")

        # Skip first line (comment)
        lines = lines[1:]

        # Parse header and sections
        header_info, sections = self._parse_native_sections(lines)

        # Store metadata
        frame.metadata.update(
            {
                "format": "lammps_molecule",
                "source_format": "native",
                "source_file": str(self._path),
                "title": header_info.get("title", ""),
                **header_info,
            }
        )

        # Parse atoms data
        atoms_data = self._parse_native_atoms(sections, header_info)
        if atoms_data:
            frame["atoms"] = mp.Block(atoms_data)

        # Parse connectivity sections
        for section in ["bonds", "angles", "dihedrals", "impropers"]:
            connectivity_data = self._parse_native_connectivity(sections, section)
            if connectivity_data:
                frame[section] = mp.Block(connectivity_data)

        return frame

    def _read_lines(self) -> list[str]:
        """Read file and return all lines."""
        with open(self._path) as f:
            return [line.rstrip("\n\r") for line in f]

    def _parse_native_sections(
        self, lines: list[str]
    ) -> tuple[dict[str, Any], dict[str, list[str]]]:
        """Parse header and body sections from native format."""
        header_info = {}
        sections = {}
        current_section = None

        i = 0
        # Parse header
        while i < len(lines):
            line = lines[i].strip()

            # Skip blank lines and comments
            if not line or line.startswith("#"):
                i += 1
                continue

            # Check if this is a header line
            header_parsed = self._parse_header_line(line, header_info)
            if header_parsed:
                i += 1
                continue

            # Check if this is a section keyword
            section_name = self._get_section_name(line)
            if section_name:
                current_section = section_name
                sections[current_section] = []
                i += 1
                # Skip next line if it's blank (as per LAMMPS format)
                if i < len(lines) and not lines[i].strip():
                    i += 1
                continue

            # If we have a current section, add this line to it
            if current_section:
                if line.strip():  # Skip blank lines within sections
                    sections[current_section].append(line)

            i += 1

        return header_info, sections

    def _parse_header_line(self, line: str, header_info: dict[str, Any]) -> bool:
        """Parse a header line and update header_info. Returns True if parsed."""
        parts = line.split()
        if len(parts) < 2:
            return False

        try:
            # Check for count keywords
            if "atoms" in line and not line.startswith("atoms"):
                header_info["n_atoms"] = int(parts[0])
                return True
            elif "bonds" in line and not line.startswith("bonds"):
                header_info["n_bonds"] = int(parts[0])
                return True
            elif "angles" in line and not line.startswith("angles"):
                header_info["n_angles"] = int(parts[0])
                return True
            elif "dihedrals" in line and not line.startswith("dihedrals"):
                header_info["n_dihedrals"] = int(parts[0])
                return True
            elif "impropers" in line and not line.startswith("impropers"):
                header_info["n_impropers"] = int(parts[0])
                return True
            elif "fragments" in line and not line.startswith("fragments"):
                header_info["n_fragments"] = int(parts[0])
                return True
            elif "body" in line and not line.startswith("body"):
                header_info["n_body_integers"] = int(parts[0])
                header_info["n_body_doubles"] = int(parts[1])
                return True
            elif "mass" in line and not line.startswith("mass"):
                header_info["total_mass"] = float(parts[0])
                return True
            elif "com" in line and not line.startswith("com"):
                header_info["center_of_mass"] = [
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                ]
                return True
            elif "inertia" in line and not line.startswith("inertia"):
                header_info["inertia"] = [float(p) for p in parts[:6]]
                return True
        except (ValueError, IndexError):
            pass

        return False

    def _get_section_name(self, line: str) -> str | None:
        """Get section name if line is a section header."""
        line_lower = line.lower().strip()

        section_mapping = {
            "coords": "Coords",
            "types": "Types",
            "molecules": "Molecules",
            "fragments": "Fragments",
            "charges": "Charges",
            "diameters": "Diameters",
            "dipoles": "Dipoles",
            "masses": "Masses",
            "bonds": "Bonds",
            "angles": "Angles",
            "dihedrals": "Dihedrals",
            "impropers": "Impropers",
            "special bond counts": "Special Bond Counts",
            "special bonds": "Special Bonds",
            "shake flags": "Shake Flags",
            "shake atoms": "Shake Atoms",
            "shake bond types": "Shake Bond Types",
            "body integers": "Body Integers",
            "body doubles": "Body Doubles",
        }

        for key, value in section_mapping.items():
            if line_lower == key:
                return value

        return None

    def _parse_native_atoms(
        self, sections: dict[str, list[str]], header_info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """Parse atoms data from native format sections."""
        atoms_data = {}

        # Get number of atoms
        n_atoms = header_info.get("n_atoms", 0)
        if n_atoms == 0:
            return atoms_data

        # Initialize arrays
        atom_ids = []

        # Parse Types section (required)
        if "Types" not in sections:
            raise ValueError("Native molecule file must contain Types section")

        types_lines = sections["Types"]
        atom_types = []

        for line in types_lines:
            parts = line.split("#")[0].split()  # Remove comments
            if len(parts) >= 2:
                atom_id = int(parts[0])
                atom_type = parts[1]
                atom_ids.append(atom_id)
                atom_types.append(atom_type)

        atoms_data["id"] = np.array(atom_ids)
        atoms_data["type"] = np.array(atom_types)

        # Parse Coords section if present
        if "Coords" in sections:
            coords_lines = sections["Coords"]
            positions = np.zeros((len(atom_ids), 3))

            for line in coords_lines:
                parts = line.split("#")[0].split()
                if len(parts) >= 4:
                    atom_id = int(parts[0])
                    if atom_id in atom_ids:
                        idx = atom_ids.index(atom_id)
                        positions[idx] = [
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                        ]

            atoms_data["x"] = positions[:, 0]
            atoms_data["y"] = positions[:, 1]
            atoms_data["z"] = positions[:, 2]

        # Parse other optional atom properties
        optional_sections = {
            "Charges": "q",
            "Masses": "mass",
            "Molecules": "mol",
            "Diameters": "diameter",
        }

        for section_name, attr_name in optional_sections.items():
            if section_name in sections:
                values = np.zeros(len(atom_ids))

                for line in sections[section_name]:
                    parts = line.split("#")[0].split()
                    if len(parts) >= 2:
                        atom_id = int(parts[0])
                        if atom_id in atom_ids:
                            idx = atom_ids.index(atom_id)
                            values[idx] = float(parts[1])

                atoms_data[attr_name] = values

        return atoms_data

    def _parse_native_connectivity(
        self, sections: dict[str, list[str]], section_type: str
    ) -> dict[str, np.ndarray] | None:
        """Parse connectivity sections from native format."""
        section_name = section_type.capitalize()

        if section_name not in sections:
            return None

        lines = sections[section_name]
        if not lines:
            return None

        connectivity_data = {}
        ids = []
        types = []
        atoms = []

        for line in lines:
            parts = line.split("#")[0].split()  # Remove comments

            if section_type == "bonds" and len(parts) >= 4:
                ids.append(int(parts[0]))
                types.append(parts[1])
                atoms.append([int(parts[2]), int(parts[3])])
            elif section_type == "angles" and len(parts) >= 5:
                ids.append(int(parts[0]))
                types.append(parts[1])
                atoms.append([int(parts[2]), int(parts[3]), int(parts[4])])
            elif section_type in ["dihedrals", "impropers"] and len(parts) >= 6:
                ids.append(int(parts[0]))
                types.append(parts[1])
                atoms.append(
                    [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]
                )

        if not ids:
            return None

        connectivity_data["id"] = np.array(ids)
        connectivity_data["type"] = np.array(types)

        atoms_array = np.array(atoms)
        if section_type == "bonds":
            connectivity_data["atom1"] = atoms_array[:, 0]
            connectivity_data["atom2"] = atoms_array[:, 1]
        elif section_type == "angles":
            connectivity_data["atom1"] = atoms_array[:, 0]
            connectivity_data["atom2"] = atoms_array[:, 1]
            connectivity_data["atom3"] = atoms_array[:, 2]
        elif section_type in ["dihedrals", "impropers"]:
            connectivity_data["atom1"] = atoms_array[:, 0]
            connectivity_data["atom2"] = atoms_array[:, 1]
            connectivity_data["atom3"] = atoms_array[:, 2]
            connectivity_data["atom4"] = atoms_array[:, 3]

        return connectivity_data


class LammpsMoleculeWriter(DataWriter):
    """LAMMPS molecule file writer supporting both native and JSON formats."""

    def __init__(self, path: str | Path, format_type: str = "native") -> None:
        super().__init__(Path(path))
        self.format_type = format_type.lower()
        if self.format_type not in ["native", "json"]:
            raise ValueError("format_type must be 'native' or 'json'")

        # Set appropriate file extension if not provided
        if self.format_type == "json" and self._path.suffix.lower() != ".json":
            self._path = self._path.with_suffix(".json")

    def write(self, frame: mp.Frame) -> None:
        """Write Frame to LAMMPS molecule file."""
        if self.format_type == "json":
            self._write_json_format(frame)
        else:
            self._write_native_format(frame)

    def _write_json_format(self, frame: mp.Frame) -> None:
        """Write molecule in JSON format."""
        # Check if frame has atoms
        if "atoms" not in frame:
            raise ValueError("Frame must contain atoms data")

        data = {
            "application": "LAMMPS",
            "format": "molecule",
            "revision": 1,
            "title": frame.metadata.get(
                "title",
                f"Molecule template written by molpy on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ),
            "schema": "https://download.lammps.org/json/molecule-schema.json",
            "units": frame.metadata.get("units", "lj"),
        }

        # Add atoms data (required)
        if "atoms" not in frame:
            raise ValueError("Frame must contain atoms data")

        atoms = frame["atoms"]

        # Types section (required)
        types_data = []
        for i in range(atoms.nrows):
            atom_id = int(atoms["id"][i]) if "id" in atoms else i + 1
            atom_type = atoms["type"][i]
            types_data.append([atom_id, atom_type])

        data["types"] = {"format": ["atom-id", "type"], "data": types_data}

        # Coordinates section
        if "x" in atoms and "y" in atoms and "z" in atoms:
            coords_data = []
            for i in range(atoms.nrows):
                atom_id = int(atoms["id"][i]) if "id" in atoms else i + 1
                x = float(atoms["x"][i])
                y = float(atoms["y"][i])
                z = float(atoms["z"][i])
                coords_data.append([atom_id, x, y, z])

            data["coords"] = {"format": ["atom-id", "x", "y", "z"], "data": coords_data}

        # Charges section
        if "q" in atoms:
            charges_data = []
            for i in range(atoms.nrows):
                atom_id = int(atoms["id"][i]) if "id" in atoms else i + 1
                charge = float(atoms["q"][i])
                charges_data.append([atom_id, charge])

            data["charges"] = {"format": ["atom-id", "charge"], "data": charges_data}

        # Masses section
        if "mass" in atoms:
            masses_data = []
            for i in range(atoms.nrows):
                atom_id = int(atoms["id"][i]) if "id" in atoms else i + 1
                mass = float(atoms["mass"][i])
                masses_data.append([atom_id, mass])

            data["masses"] = {"format": ["atom-id", "mass"], "data": masses_data}

        # Molecule IDs section
        if "mol" in atoms:
            molecule_data = []
            for i in range(atoms.nrows):
                atom_id = int(atoms["id"][i]) if "id" in atoms else i + 1
                mol_id = int(atoms["mol"][i])
                molecule_data.append([atom_id, mol_id])

            data["molecule"] = {
                "format": ["atom-id", "molecule-id"],
                "data": molecule_data,
            }

        # Diameters section
        if "diameter" in atoms:
            diameters_data = []
            for i in range(atoms.nrows):
                atom_id = int(atoms["id"][i]) if "id" in atoms else i + 1
                diameter = float(atoms["diameter"][i])
                diameters_data.append([atom_id, diameter])

            data["diameters"] = {
                "format": ["atom-id", "diameter"],
                "data": diameters_data,
            }

        # Connectivity sections
        self._add_json_connectivity(data, frame, "bonds")
        self._add_json_connectivity(data, frame, "angles")
        self._add_json_connectivity(data, frame, "dihedrals")
        self._add_json_connectivity(data, frame, "impropers")

        # Add molecule properties
        if "center_of_mass" in frame.metadata:
            data["com"] = frame.metadata["center_of_mass"].tolist()
        if "total_mass" in frame.metadata:
            data["masstotal"] = float(frame.metadata["total_mass"])
        if "inertia" in frame.metadata:
            data["inertia"] = frame.metadata["inertia"].tolist()

        # Write to file
        with open(self._path, "w") as f:
            json.dump(data, f, indent=4)

    def _add_json_connectivity(self, data: dict, frame: mp.Frame, section: str) -> None:
        """Add connectivity section to JSON data."""
        if section not in frame:
            return

        block = frame[section]
        if block.nrows == 0:
            return

        connectivity_data = []
        format_list = []  # Initialize format_list

        if section == "bonds":
            format_list = ["bond-type", "atom1", "atom2"]
            for i in range(block.nrows):
                bond_type = block["type"][i]
                atom1 = int(block["atom1"][i])
                atom2 = int(block["atom2"][i])
                connectivity_data.append([bond_type, atom1, atom2])

        elif section == "angles":
            format_list = ["angle-type", "atom1", "atom2", "atom3"]
            for i in range(block.nrows):
                angle_type = block["type"][i]
                atom1 = int(block["atom1"][i])
                atom2 = int(block["atom2"][i])
                atom3 = int(block["atom3"][i])
                connectivity_data.append([angle_type, atom1, atom2, atom3])

        elif section in ["dihedrals", "impropers"]:
            type_name = section[:-1] + "-type"  # 'dihedral-type' or 'improper-type'
            format_list = [type_name, "atom1", "atom2", "atom3", "atom4"]
            for i in range(block.nrows):
                item_type = block["type"][i]
                atom1 = int(block["atom1"][i])
                atom2 = int(block["atom2"][i])
                atom3 = int(block["atom3"][i])
                atom4 = int(block["atom4"][i])
                connectivity_data.append([item_type, atom1, atom2, atom3, atom4])

        if connectivity_data and format_list:
            data[section] = {"format": format_list, "data": connectivity_data}

    def _write_native_format(self, frame: mp.Frame) -> None:
        """Write molecule in native format."""
        # Check if frame has atoms
        if "atoms" not in frame:
            raise ValueError("Frame must contain atoms data")

        lines = []

        # Header comment
        title = frame.metadata.get(
            "title",
            f"Molecule template written by molpy on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )
        lines.append(f"# {title}")
        lines.append("")

        # Header section
        if "atoms" in frame:
            lines.append(f"{frame['atoms'].nrows} atoms")

        for section in ["bonds", "angles", "dihedrals", "impropers"]:
            if section in frame and frame[section].nrows > 0:
                lines.append(f"{frame[section].nrows} {section}")

        # Add molecule properties to header
        if "total_mass" in frame.metadata:
            lines.append(f"{frame.metadata['total_mass']:.6f} mass")

        if "center_of_mass" in frame.metadata:
            com = frame.metadata["center_of_mass"]
            lines.append(f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f} com")

        if "inertia" in frame.metadata:
            inertia = frame.metadata["inertia"]
            inertia_str = " ".join([f"{val:.6f}" for val in inertia])
            lines.append(f"{inertia_str} inertia")

        lines.append("")

        # Body sections
        if "atoms" in frame:
            self._write_native_atoms_sections(lines, frame["atoms"])

        # Connectivity sections
        for section in ["bonds", "angles", "dihedrals", "impropers"]:
            if section in frame and frame[section].nrows > 0:
                self._write_native_connectivity_section(lines, frame[section], section)

        # Write to file
        with open(self._path, "w") as f:
            f.write("\n".join(lines))

    def _write_native_atoms_sections(self, lines: list[str], atoms: mp.Block) -> None:
        """Write atoms-related sections in native format."""
        # Coords section
        if "x" in atoms and "y" in atoms and "z" in atoms:
            lines.append("Coords")
            lines.append("")

            for i in range(atoms.nrows):
                atom_id = int(atoms["id"][i]) if "id" in atoms else i + 1
                x = float(atoms["x"][i])
                y = float(atoms["y"][i])
                z = float(atoms["z"][i])
                lines.append(f"{atom_id} {x:.6f} {y:.6f} {z:.6f}")

            lines.append("")

        # Types section (required)
        lines.append("Types")
        lines.append("")

        for i in range(atoms.nrows):
            atom_id = int(atoms["id"][i]) if "id" in atoms else i + 1
            atom_type = atoms["type"][i]
            lines.append(f"{atom_id} {atom_type}")

        lines.append("")

        # Optional sections
        optional_sections = [
            ("mol", "Molecules"),
            ("q", "Charges"),
            ("mass", "Masses"),
            ("diameter", "Diameters"),
        ]

        for attr_name, section_name in optional_sections:
            if attr_name in atoms:
                lines.append(section_name)
                lines.append("")

                for i in range(atoms.nrows):
                    atom_id = int(atoms["id"][i]) if "id" in atoms else i + 1
                    value = atoms[attr_name][i]
                    if value is None:
                        value = 0.0
                    value = float(value)
                    lines.append(f"{atom_id} {value:.6f}")

                lines.append("")

    def _write_native_connectivity_section(
        self, lines: list[str], block: mp.Block, section_type: str
    ) -> None:
        """Write connectivity section in native format."""
        lines.append(section_type.capitalize())
        lines.append("")

        for i in range(block.nrows):
            item_id = int(block["id"][i]) if "id" in block else i + 1
            item_type = block["type"][i]

            if section_type == "bonds":
                atom1 = int(block["atom1"][i])
                atom2 = int(block["atom2"][i])
                lines.append(f"{item_id} {item_type} {atom1} {atom2}")
            elif section_type == "angles":
                atom1 = int(block["atom1"][i])
                atom2 = int(block["atom2"][i])
                atom3 = int(block["atom3"][i])
                lines.append(f"{item_id} {item_type} {atom1} {atom2} {atom3}")
            elif section_type in ["dihedrals", "impropers"]:
                atom1 = int(block["atom1"][i])
                atom2 = int(block["atom2"][i])
                atom3 = int(block["atom3"][i])
                atom4 = int(block["atom4"][i])
                lines.append(f"{item_id} {item_type} {atom1} {atom2} {atom3} {atom4}")

        lines.append("")
