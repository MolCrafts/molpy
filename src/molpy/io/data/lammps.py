"""
Modern LAMMPS data file I/O using Block.from_csv.

This module provides a clean, imperative approach to reading and writing
LAMMPS data files using the Block.from_csv functionality.
"""

from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from molpy.core.frame import Block
from molpy.core.box import Box
from molpy.core.forcefield import ForceField
from molpy.core.frame import Frame

from .base import DataReader, DataWriter


class LammpsDataReader(DataReader):
    """Modern LAMMPS data file reader using Block.from_csv."""

    def __init__(self, path: str | Path, atom_style: str = "full") -> None:
        super().__init__(Path(path))
        self.atom_style = atom_style

    def read(self, frame: Frame | None = None) -> Frame:
        """Read LAMMPS data file into a Frame."""
        frame = frame or Frame()

        # Read and parse the file
        lines = self._read_lines()
        sections = self._extract_sections(lines)

        # Parse header and set up box
        header_info = self._parse_header(sections.get("header", []))
        frame.metadata["box"] = self._create_box(header_info["box_bounds"])

        # Parse masses if present
        masses = self._parse_masses(sections.get("Masses", []))

        # Parse type labels
        type_labels = self._parse_type_labels(sections)

        # Parse force field parameters
        forcefield = self._parse_force_field(sections)

        # Parse atoms section
        if "Atoms" in sections:
            frame["atoms"] = self._parse_atoms_section(
                sections["Atoms"], masses, type_labels.get("atom", {})
            )

        # Parse connectivity sections
        if "Bonds" in sections and header_info["counts"].get("bonds", 0) > 0:
            frame["bonds"] = self._parse_connectivity_section(
                sections["Bonds"], "bond", type_labels.get("bond", {})
            )

        if "Angles" in sections and header_info["counts"].get("angles", 0) > 0:
            frame["angles"] = self._parse_connectivity_section(
                sections["Angles"], "angle", type_labels.get("angle", {})
            )

        if "Dihedrals" in sections and header_info["counts"].get("dihedrals", 0) > 0:
            frame["dihedrals"] = self._parse_connectivity_section(
                sections["Dihedrals"], "dihedral", type_labels.get("dihedral", {})
            )

        if "Impropers" in sections and header_info["counts"].get("impropers", 0) > 0:
            frame["impropers"] = self._parse_connectivity_section(
                sections["Impropers"], "improper", type_labels.get("improper", {})
            )

        # Store metadata
        frame.metadata.update(
            {
                "format": "lammps_data",
                "atom_style": self.atom_style,
                "counts": header_info["counts"],
                "source_file": str(self._path),
                "forcefield": forcefield,
            }
        )

        return frame

    def _read_lines(self) -> list[str]:
        """Read file and return non-empty, non-comment lines."""
        with open(self._path) as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

    def _extract_sections(self, lines: list[str]) -> dict[str, list[str]]:
        """Extract sections from LAMMPS data file."""
        sections = {"header": []}
        current_section = "header"

        section_keywords = [
            "atoms",
            "masses",
            "bonds",
            "angles",
            "dihedrals",
            "impropers",
            "atom type labels",
            "bond type labels",
            "angle type labels",
            "dihedral type labels",
            "improper type labels",
            "pair coeffs",
            "bond coeffs",
            "angle coeffs",
            "dihedral coeffs",
            "improper coeffs",
        ]

        for line in lines:
            line_lower = line.lower()

            # Check if this line starts a new section
            if any(line_lower.startswith(keyword) for keyword in section_keywords):
                if "type labels" in line_lower:
                    # Handle type labels sections
                    section_name = line.replace("Type Labels", "TypeLabels").replace(
                        " ", ""
                    )
                elif "coeffs" in line_lower:
                    # Handle force field coefficients sections
                    section_name = line.replace(" ", "")
                else:
                    # Handle regular sections
                    section_name = line.split()[0].capitalize()
                current_section = section_name
                sections[current_section] = []
            else:
                # Add line to current section
                if current_section not in sections:
                    sections[current_section] = []
                sections[current_section].append(line)

        return sections

    def _parse_header(self, header_lines: list[str]) -> dict[str, Any]:
        """Parse header information."""
        counts = {}
        box_bounds = {}

        for line in header_lines:
            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                count = int(parts[0])
                if "atoms" in line.lower() and not line.lower().startswith("atoms"):
                    counts["atoms"] = count
                elif "bonds" in line.lower() and not line.lower().startswith("bonds"):
                    counts["bonds"] = count
                elif "angles" in line.lower() and not line.lower().startswith("angles"):
                    counts["angles"] = count
                elif "dihedrals" in line.lower() and not line.lower().startswith(
                    "dihedrals"
                ):
                    counts["dihedrals"] = count
                elif "impropers" in line.lower() and not line.lower().startswith(
                    "impropers"
                ):
                    counts["impropers"] = count
                elif "atom types" in line.lower():
                    counts["atom_types"] = count
                elif "bond types" in line.lower():
                    counts["bond_types"] = count
                elif "angle types" in line.lower():
                    counts["angle_types"] = count
                elif "dihedral types" in line.lower():
                    counts["dihedral_types"] = count
                elif "improper types" in line.lower():
                    counts["improper_types"] = count
                elif "xlo xhi" in line.lower():
                    box_bounds["x"] = (float(parts[0]), float(parts[1]))
                elif "ylo" in line.lower() and "yhi" in line.lower():
                    box_bounds["y"] = (float(parts[0]), float(parts[1]))
                elif "zlo zhi" in line.lower():
                    box_bounds["z"] = (float(parts[0]), float(parts[1]))
            except (ValueError, IndexError):
                continue

        return {"counts": counts, "box_bounds": box_bounds if box_bounds else None}

    def _create_box(self, box_bounds: dict[str, tuple[float, float]] | None) -> Box:
        """Create Box from bounds."""
        if not box_bounds:
            # Default box
            return Box(np.array([10.0, 10.0, 10.0]))

        lengths = np.array(
            [
                box_bounds.get("x", (0.0, 10.0))[1]
                - box_bounds.get("x", (0.0, 10.0))[0],
                box_bounds.get("y", (0.0, 10.0))[1]
                - box_bounds.get("y", (0.0, 10.0))[0],
                box_bounds.get("z", (0.0, 10.0))[1]
                - box_bounds.get("z", (0.0, 10.0))[0],
            ]
        )
        origin = np.array(
            [
                box_bounds.get("x", (0.0, 10.0))[0],
                box_bounds.get("y", (0.0, 10.0))[0],
                box_bounds.get("z", (0.0, 10.0))[0],
            ]
        )
        return Box(lengths, origin=origin)

    def _parse_masses(self, mass_lines: list[str]) -> dict[str, float]:
        """Parse mass section."""
        masses = {}
        for line in mass_lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    type_str = parts[0]
                    mass = float(parts[1])
                    masses[type_str] = mass
                except ValueError:
                    continue
        return masses

    def _parse_type_labels(
        self, sections: dict[str, list[str]]
    ) -> dict[str, dict[int, str]]:
        """Parse all type labels sections."""
        type_labels = {}

        label_sections = {
            "atom": "AtomTypeLabels",
            "bond": "BondTypeLabels",
            "angle": "AngleTypeLabels",
            "dihedral": "DihedralTypeLabels",
            "improper": "ImproperTypeLabels",
        }

        for label_type, section_name in label_sections.items():
            if section_name in sections:
                id_to_label = {}
                for line in sections[section_name]:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            type_id = int(parts[0])
                            label = parts[1]
                            id_to_label[type_id] = label
                        except ValueError:
                            continue
                type_labels[label_type] = id_to_label

        return type_labels

    def _parse_force_field(self, sections: dict[str, list[str]]) -> ForceField:
        """Parse force field parameters into ForceField."""
        forcefield = ForceField()

        # Parse pair coefficients
        if "PairCoeffs" in sections:
            forcefield.def_pairstyle("lj/cut")
            for line in sections["PairCoeffs"]:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        int(parts[0])
                        float(parts[1])
                        float(parts[2])
                        # TODO: Fix force field implementation
                        # pair_type = pair_style.def_type(type_name)
                        # pair_type["epsilon"] = epsilon
                        # pair_type["sigma"] = sigma
                    except (ValueError, IndexError):
                        continue

        # Parse bond coefficients
        if "BondCoeffs" in sections:
            forcefield.def_bondstyle("harmonic")
            for line in sections["BondCoeffs"]:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        int(parts[0])
                        float(parts[1])
                        float(parts[2])
                        # TODO: Fix force field implementation
                        # bond_type = bond_style.def_type(type_name)
                        # bond_type["k"] = k
                        # bond_type["r0"] = r0
                    except (ValueError, IndexError):
                        continue

        # Parse angle coefficients
        if "AngleCoeffs" in sections:
            forcefield.def_anglestyle("harmonic")
            for line in sections["AngleCoeffs"]:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        int(parts[0])
                        float(parts[1])
                        float(parts[2])
                        # TODO: Fix force field implementation
                        # angle_type = angle_style.def_type(type_name)
                        # angle_type["k"] = k
                        # angle_type["theta0"] = theta0
                    except (ValueError, IndexError):
                        continue

        # Parse dihedral coefficients
        if "DihedralCoeffs" in sections:
            forcefield.def_dihedralstyle("harmonic")
            for line in sections["DihedralCoeffs"]:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        int(parts[0])
                        float(parts[1])
                        int(parts[2])
                        int(parts[3])
                        # TODO: Fix force field implementation
                        # dihedral_type = dihedral_style.def_type(type_name)
                        # dihedral_type["k"] = k
                        # dihedral_type["d"] = d
                        # dihedral_type["n"] = n
                    except (ValueError, IndexError):
                        continue

        # Parse improper coefficients
        if "ImproperCoeffs" in sections:
            forcefield.def_improperstyle("harmonic")
            for line in sections["ImproperCoeffs"]:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        int(parts[0])
                        float(parts[1])
                        int(parts[2])
                        int(parts[3])
                        # TODO: Fix force field implementation
                        # improper_type = improper_style.def_type(type_name)
                        # improper_type["k"] = k
                        # improper_type["d"] = d
                        # improper_type["n"] = n
                    except (ValueError, IndexError):
                        continue

        return forcefield

    def _parse_atoms_section(
        self,
        atom_lines: list[str],
        masses: dict[str, float],
        type_labels: dict[int, str],
    ) -> Block:
        """Parse atoms section using Block.from_csv with space delimiter."""
        if not atom_lines:
            return Block()

        # Create space-separated string for Block.from_csv
        csv_lines = []

        # Add header based on atom style
        if self.atom_style == "full":
            header = ["id", "mol", "type", "q", "x", "y", "z"]
        elif self.atom_style == "charge":
            header = ["id", "type", "q", "x", "y", "z"]
        else:  # atomic
            header = ["id", "type", "x", "y", "z"]

        csv_lines.append(" ".join(header))

        # Add data lines directly
        for line in atom_lines:
            parts = line.split()
            if len(parts) >= len(header):
                csv_lines.append(line)

        # Parse using Block.from_csv with space delimiter
        csv_string = "\n".join(csv_lines)
        block = Block.from_csv(
            StringIO(csv_string), delimiter=" ", skipinitialspace=True
        )

        # Add mass information
        if block.nrows > 0:
            mass_values = []
            for type_str in block["type"]:
                mass_values.append(masses.get(str(type_str), 1.0))
            block["mass"] = np.array(mass_values)

            # Convert numeric types back to string types using type labels
            if type_labels:
                converted_types = []
                for type_id in block["type"]:
                    try:
                        type_id_int = int(type_id)
                        converted_type = type_labels.get(type_id_int, str(type_id))
                        converted_types.append(converted_type)
                    except (ValueError, TypeError):
                        converted_types.append(str(type_id))
                block["type"] = np.array(converted_types)

        return block

    def _parse_connectivity_section(
        self, lines: list[str], section_type: str, type_labels: dict[int, str]
    ) -> Block:
        """Parse connectivity sections (bonds, angles, dihedrals, impropers)."""
        if not lines:
            return Block()

        # Define headers for each section type
        # Note: atom IDs are kept as 1-based (no conversion to indices)
        headers = {
            "bond": ["id", "type", "atom1", "atom2"],
            "angle": ["id", "type", "atom1", "atom2", "atom3"],
            "dihedral": ["id", "type", "atom1", "atom2", "atom3", "atom4"],
            "improper": ["id", "type", "atom1", "atom2", "atom3", "atom4"],
        }

        header = headers[section_type]
        csv_lines = [" ".join(header)]

        # Add data lines directly
        for line in lines:
            parts = line.split()
            if len(parts) >= len(header):
                csv_lines.append(line)

        # Parse using Block.from_csv with space delimiter
        csv_string = "\n".join(csv_lines)
        block = Block.from_csv(
            StringIO(csv_string), delimiter=" ", skipinitialspace=True
        )

        # Keep atom IDs as 1-based (no conversion)
        # atom1, atom2, etc. are stored as atom IDs (1-based)

        return block


class LammpsDataWriter(DataWriter):
    """Modern LAMMPS data file writer using Block.to_csv approach.

    **Important Requirements:**
    - Atoms in the frame must have an 'id' field. This field is required
      to map atom indices to atom IDs for LAMMPS output.
    - Connectivity data (bonds, angles, dihedrals) in the frame uses atom
      indices (0-based from to_frame()). The writer automatically converts
      these indices to atom IDs using the index->ID mapping from the atoms
      'id' field.

    **Frame Structure:**
    - Atoms: Must include 'id' field. Other required fields depend on atom_style.
    - Bonds/Angles/Dihedrals: Use atom indices in 'atom_i', 'atom_j', 'atom_k', 'atom_l'
      (from to_frame()). These are 0-based indices that will be converted to 1-based atom IDs.
    """

    def __init__(self, path: str | Path, atom_style: str = "full") -> None:
        super().__init__(Path(path))
        self.atom_style = atom_style

    def write(self, frame: Frame) -> None:
        """Write Frame to LAMMPS data file.

        Args:
            frame: Frame containing atoms and optionally bonds/angles/dihedrals.
                  Atoms must have 'id' field.

        Raises:
            ValueError: If atoms are missing 'id' field.
        """
        lines = []

        # Header
        lines.append(
            f"# LAMMPS data file written by molpy on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append("")

        # Count sections
        counts = self._get_counts(frame)
        self._write_counts(lines, counts)
        lines.append("")

        # Type counts
        self._write_type_counts(lines, frame)
        lines.append("")

        # Box bounds
        self._write_box_bounds(lines, frame)
        lines.append("")

        # Type labels sections (must come before Masses for LAMMPS)
        self._write_type_labels_sections(lines, frame)

        # Masses section
        if "atoms" in frame:
            self._write_masses_section(lines, frame)

        # Force field coefficients sections
        self._write_force_field_coeffs_sections(lines, frame)

        # Data sections
        if "atoms" in frame:
            self._write_atoms_section(lines, frame)

        if "bonds" in frame and counts.get("bonds", 0) > 0:
            self._write_connectivity_section(lines, frame, "bonds")

        if "angles" in frame and counts.get("angles", 0) > 0:
            self._write_connectivity_section(lines, frame, "angles")

        if "dihedrals" in frame and counts.get("dihedrals", 0) > 0:
            self._write_connectivity_section(lines, frame, "dihedrals")

        if "impropers" in frame and counts.get("impropers", 0) > 0:
            self._write_connectivity_section(lines, frame, "impropers")

        # Write to file
        with open(self._path, "w") as f:
            f.write("\n".join(lines))

    def _get_counts(self, frame: Frame) -> dict[str, int]:
        """Get counts from frame."""
        counts = {}
        if "atoms" in frame:
            counts["atoms"] = frame["atoms"].nrows
        if "bonds" in frame:
            counts["bonds"] = frame["bonds"].nrows
        if "angles" in frame:
            counts["angles"] = frame["angles"].nrows
        if "dihedrals" in frame:
            counts["dihedrals"] = frame["dihedrals"].nrows
        if "impropers" in frame:
            counts["impropers"] = frame["impropers"].nrows
        return counts

    def _collect_actual_types(self, frame: Frame) -> dict[str, set[str]]:
        """Collect actual types used in frame blocks."""
        actual_types = {
            "atom_types": set(),
            "bond_types": set(),
            "angle_types": set(),
            "dihedral_types": set(),
            "improper_types": set(),
        }

        mapping = {
            "atoms": "atom_types",
            "bonds": "bond_types",
            "angles": "angle_types",
            "dihedrals": "dihedral_types",
            "impropers": "improper_types",
        }

        for block_name, type_key in mapping.items():
            if block_name in frame and frame[block_name].nrows > 0:
                types = frame[block_name]["type"]
                # Convert to string set, handling both string and numeric types
                unique_types = set(str(t) for t in np.unique(types))
                actual_types[type_key] = unique_types

        return actual_types

    def _merge_type_labels(
        self,
        metadata_types: dict[str, list[str]] | None,
        actual_types: dict[str, set[str]],
    ) -> dict[str, list[str]]:
        """Merge metadata type labels with actual types used in frame.

        Args:
            metadata_types: Type labels from metadata (if any)
            actual_types: Actual types collected from frame blocks

        Returns:
            Merged type labels dict with sorted lists
        """
        merged = {}

        type_keys = [
            "atom_types",
            "bond_types",
            "angle_types",
            "dihedral_types",
            "improper_types",
        ]

        for key in type_keys:
            metadata_list = metadata_types.get(key, []) if metadata_types else []
            actual_set = actual_types.get(key, set())

            # Merge: start with metadata types, then add any actual types not in metadata
            merged_set = set(metadata_list) | actual_set
            merged[key] = sorted(list(merged_set))

        return merged

    def _get_merged_type_labels(self, frame: Frame) -> dict[str, list[str]]:
        """Get merged type labels from metadata and actual frame blocks.

        This method computes the final type labels that should be used
        throughout the file writing process, ensuring consistency.

        Returns:
            Dict mapping type keys to sorted lists of type names
        """
        metadata_type_labels = frame.metadata.get("type_labels")
        actual_types = self._collect_actual_types(frame)

        if metadata_type_labels is not None:
            return self._merge_type_labels(metadata_type_labels, actual_types)
        else:
            # Fallback to old behavior: only use actual types if they are strings
            merged_types = {}
            type_key_mapping = {
                "atoms": "atom_types",
                "bonds": "bond_types",
                "angles": "angle_types",
                "dihedrals": "dihedral_types",
                "impropers": "improper_types",
            }
            for block_name, type_key in type_key_mapping.items():
                if block_name in frame and frame[block_name].nrows > 0:
                    types = frame[block_name]["type"]
                    if self._needs_type_labels(types):
                        merged_types[type_key] = sorted(list(actual_types[type_key]))
            return merged_types

    def _get_type_to_id_mapping(self, type_list: list[str] | None) -> dict[str, int]:
        """Get mapping from type name to type ID (1-based).

        Args:
            type_list: Sorted list of type names, or None if types are numeric

        Returns:
            Dict mapping type name to type ID (1-based)
        """
        if type_list is None or len(type_list) == 0:
            return {}
        return {type_name: type_idx + 1 for type_idx, type_name in enumerate(type_list)}

    def _write_counts(self, lines: list[str], counts: dict[str, int]) -> None:
        """Write count lines."""
        if "atoms" in counts:
            lines.append(f"{counts['atoms']} atoms")
        if "bonds" in counts and counts["bonds"] > 0:
            lines.append(f"{counts['bonds']} bonds")
        if "angles" in counts and counts["angles"] > 0:
            lines.append(f"{counts['angles']} angles")
        if "dihedrals" in counts and counts["dihedrals"] > 0:
            lines.append(f"{counts['dihedrals']} dihedrals")
        if "impropers" in counts and counts["impropers"] > 0:
            lines.append(f"{counts['impropers']} impropers")

    def _write_type_counts(self, lines: list[str], frame: Frame) -> None:
        """Write type count lines.

        Uses merged type labels from metadata (if present) or actual types.
        """
        merged_types = self._get_merged_type_labels(frame)
        metadata_type_labels = frame.metadata.get("type_labels")

        # If metadata has type_labels, use merged types
        if metadata_type_labels is not None:
            # Write counts based on merged types
            if "atom_types" in merged_types and merged_types["atom_types"]:
                lines.append(f"{len(merged_types['atom_types'])} atom types")

            if "bond_types" in merged_types and merged_types["bond_types"]:
                lines.append(f"{len(merged_types['bond_types'])} bond types")

            if "angle_types" in merged_types and merged_types["angle_types"]:
                lines.append(f"{len(merged_types['angle_types'])} angle types")

            if "dihedral_types" in merged_types and merged_types["dihedral_types"]:
                lines.append(f"{len(merged_types['dihedral_types'])} dihedral types")

            if "improper_types" in merged_types and merged_types["improper_types"]:
                lines.append(f"{len(merged_types['improper_types'])} improper types")
        else:
            # Fallback to old behavior: count unique types from blocks
            if "atoms" in frame:
                unique_types = np.unique(frame["atoms"]["type"])
                lines.append(f"{len(unique_types)} atom types")

            if "bonds" in frame and frame["bonds"].nrows > 0:
                unique_types = np.unique(frame["bonds"]["type"])
                lines.append(f"{len(unique_types)} bond types")

            if "angles" in frame and frame["angles"].nrows > 0:
                unique_types = np.unique(frame["angles"]["type"])
                lines.append(f"{len(unique_types)} angle types")

            if "dihedrals" in frame and frame["dihedrals"].nrows > 0:
                unique_types = np.unique(frame["dihedrals"]["type"])
                lines.append(f"{len(unique_types)} dihedral types")

            if "impropers" in frame and frame["impropers"].nrows > 0:
                unique_types = np.unique(frame["impropers"]["type"])
                lines.append(f"{len(unique_types)} improper types")

    def _write_box_bounds(self, lines: list[str], frame: Frame) -> None:
        """Write box bounds."""
        if frame.metadata.get("box") is not None:
            box = frame.metadata["box"]
            lines.append(
                f"{box.origin[0]:.6f} {box.origin[0] + box.lengths[0]:.6f} xlo xhi"
            )
            lines.append(
                f"{box.origin[1]:.6f} {box.origin[1] + box.lengths[1]:.6f} ylo yhi"
            )
            lines.append(
                f"{box.origin[2]:.6f} {box.origin[2] + box.lengths[2]:.6f} zlo zhi"
            )
        else:
            lines.append("0.0 10.0 xlo xhi")
            lines.append("0.0 10.0 ylo yhi")
            lines.append("0.0 10.0 zlo zhi")

    def _write_masses_section(self, lines: list[str], frame: Frame) -> None:
        """Write masses section.

        Uses merged type labels to ensure type_id consistency.
        Writes masses for all types in merged list. For types not in atoms,
        uses default mass of 1.0.
        """
        lines.append("Masses")
        lines.append("")

        merged_types = self._get_merged_type_labels(frame)
        atom_type_list = merged_types.get("atom_types", [])

        # Build type_to_id mapping from merged types
        type_to_id = self._get_type_to_id_mapping(atom_type_list)

        if not atom_type_list:
            # Fallback: use actual types from atoms if no merged types
            atoms_data = frame["atoms"]
            unique_types = np.unique(atoms_data["type"])
            atom_type_list = sorted([str(t) for t in unique_types])
            type_to_id = self._get_type_to_id_mapping(atom_type_list)

        atoms_data = frame["atoms"]
        # Create a dict of type -> mass for actual atoms
        type_to_mass = {}
        for atom_type in np.unique(atoms_data["type"]):
            atom_type_str = str(atom_type)
            mask = atoms_data["type"] == atom_type

            # Get mass - prefer element field, fallback to mass field
            if "element" in atoms_data:
                from molpy.core.element import Element

                element_symbol = atoms_data["element"][mask][0]
                mass = Element(element_symbol).mass
            elif "mass" in atoms_data:
                mass = atoms_data["mass"][mask][0]
            else:
                mass = 1.0  # Default fallback
            type_to_mass[atom_type_str] = mass

        # Write masses for all types in merged list
        # For types not in atoms, use default mass
        for atom_type in atom_type_list:
            type_id = type_to_id[
                atom_type
            ]  # Should always exist since built from same list
            mass = type_to_mass.get(atom_type, 1.0)  # Default to 1.0 if not in atoms
            lines.append(f"{type_id} {mass:.6f}")

        lines.append("")

    def _write_type_labels_sections(self, lines: list[str], frame: Frame) -> None:
        """Write type labels sections if needed.

        If frame.metadata contains 'type_labels' (dict[str, list[str]]),
        use those as the base and merge with actual types used in blocks.
        Otherwise, infer from blocks as before.
        """
        merged_types = self._get_merged_type_labels(frame)

        # Write sections
        section_configs = [
            ("atom_types", "Atom Type Labels"),
            ("bond_types", "Bond Type Labels"),
            ("angle_types", "Angle Type Labels"),
            ("dihedral_types", "Dihedral Type Labels"),
            ("improper_types", "Improper Type Labels"),
        ]

        for type_key, section_name in section_configs:
            if type_key in merged_types and merged_types[type_key]:
                lines.append(section_name)
                lines.append("")

                type_list = merged_types[type_key]
                for type_id, type_label in enumerate(type_list, 1):
                    lines.append(f"{type_id} {type_label}")

                lines.append("")

    def _needs_type_labels(self, types: np.ndarray) -> bool:
        """Check if type labels section is needed."""
        # Only write type labels if types are non-numeric (strings)
        # Numeric types (integers) don't need labels
        if len(types) == 0:
            return False
        # Check if any type is a string (not numeric)
        return types.dtype.kind in ("U", "S", "O")  # Unicode, byte string, or object

    def _write_force_field_coeffs_sections(
        self, lines: list[str], frame: Frame
    ) -> None:
        """Write force field coefficients sections."""
        forcefield = frame.metadata.get("forcefield")
        if not forcefield:
            return

        # Import style classes
        from molpy import (
            AngleStyle,
            BondStyle,
            DihedralStyle,
            ImproperStyle,
            PairStyle,
            Type,
        )

        # Write pair coefficients
        pair_styles = forcefield.get_styles(PairStyle)
        if pair_styles:
            lines.append("Pair Coeffs")
            lines.append("")
            for style in pair_styles:
                for type_obj in style.types.bucket(Type):
                    type_id = int(type_obj.name.split("_")[1])
                    epsilon = type_obj.get("epsilon", 0.0)
                    sigma = type_obj.get("sigma", 1.0)
                    lines.append(f"{type_id} {epsilon:.6f} {sigma:.6f}")
            lines.append("")

        # Write bond coefficients
        bond_styles = forcefield.get_styles(BondStyle)
        if bond_styles:
            lines.append("Bond Coeffs")
            lines.append("")
            for style in bond_styles:
                for type_obj in style.types.bucket(Type):
                    type_id = int(type_obj.name.split("_")[1])
                    k = type_obj.get("k", 0.0)
                    r0 = type_obj.get("r0", 1.0)
                    lines.append(f"{type_id} {k:.6f} {r0:.6f}")
            lines.append("")

        # Write angle coefficients
        angle_styles = forcefield.get_styles(AngleStyle)
        if angle_styles:
            lines.append("Angle Coeffs")
            lines.append("")
            for style in angle_styles:
                for type_obj in style.types.bucket(Type):
                    type_id = int(type_obj.name.split("_")[1])
                    k = type_obj.get("k", 0.0)
                    theta0 = type_obj.get("theta0", 0.0)
                    lines.append(f"{type_id} {k:.6f} {theta0:.6f}")
            lines.append("")

        # Write dihedral coefficients
        dihedral_styles = forcefield.get_styles(DihedralStyle)
        if dihedral_styles:
            lines.append("Dihedral Coeffs")
            lines.append("")
            for style in dihedral_styles:
                for type_obj in style.types.bucket(Type):
                    type_id = int(type_obj.name.split("_")[1])
                    k = type_obj.get("k", 0.0)
                    d = type_obj.get("d", 1)
                    n = type_obj.get("n", 1)
                    lines.append(f"{type_id} {k:.6f} {d} {n}")
            lines.append("")

        # Write improper coefficients
        improper_styles = forcefield.get_styles(ImproperStyle)
        if improper_styles:
            lines.append("Improper Coeffs")
            lines.append("")
            for style in improper_styles:
                for type_obj in style.types.bucket(Type):
                    type_id = int(type_obj.name.split("_")[1])
                    k = type_obj.get("k", 0.0)
                    d = type_obj.get("d", 1)
                    n = type_obj.get("n", 1)
                    lines.append(f"{type_id} {k:.6f} {d} {n}")
            lines.append("")

    def _write_atoms_section(self, lines: list[str], frame: Frame) -> None:
        """Write atoms section.

        Uses merged type labels to ensure type_id consistency.
        Requires that atoms have an 'id' field.

        Args:
            lines: List of lines to append to
            frame: Frame containing atoms data

        Raises:
            ValueError: If atoms are missing 'id' field
        """
        lines.append("Atoms")
        lines.append("")

        atoms_data = frame["atoms"]

        # Require that all atoms have an 'id' field
        if "id" not in atoms_data:
            raise ValueError(
                "Atoms in frame must have 'id' field. "
                "This field is required for LAMMPS output to map indices to atom IDs."
            )

        merged_types = self._get_merged_type_labels(frame)
        atom_type_list = merged_types.get("atom_types", [])

        # Build type_to_id mapping from merged types
        type_to_id = self._get_type_to_id_mapping(atom_type_list)

        if not atom_type_list:
            # Fallback: use actual types from atoms if no merged types
            unique_types = np.unique(atoms_data["type"])
            atom_type_list = sorted([str(t) for t in unique_types])
            type_to_id = self._get_type_to_id_mapping(atom_type_list)

        for idx in range(len(atoms_data["type"])):
            # Use atom ID from the 'id' field
            atom_id = int(atoms_data["id"][idx])
            atom_type_str = str(atoms_data["type"][idx])
            # Use merged type_id mapping, fallback to old behavior
            if atom_type_str in type_to_id:
                atom_type = type_to_id[atom_type_str]
            else:
                # Fallback: compute on the fly if not in merged list
                unique_types = np.unique(atoms_data["type"])
                fallback_mapping = {
                    str(t): type_idx + 1
                    for type_idx, t in enumerate(sorted(unique_types))
                }
                atom_type = fallback_mapping.get(atom_type_str, 1)

            # Get coordinates - must use separate x, y, z fields
            x = float(atoms_data["x"][idx])
            y = float(atoms_data["y"][idx])
            z = float(atoms_data["z"][idx])

            if self.atom_style == "full":
                mol_id = int(atoms_data["mol"][idx])
                charge = float(atoms_data["q"][idx])
                lines.append(
                    f"{atom_id} {mol_id} {atom_type} {charge:.6f} {x:.6f} {y:.6f} {z:.6f}"
                )
            elif self.atom_style == "charge":
                charge = float(atoms_data["q"][idx])
                lines.append(
                    f"{atom_id} {atom_type} {charge:.6f} {x:.6f} {y:.6f} {z:.6f}"
                )
            else:  # atomic
                lines.append(f"{atom_id} {atom_type} {x:.6f} {y:.6f} {z:.6f}")

        lines.append("")

    def _write_connectivity_section(
        self, lines: list[str], frame: Frame, section_name: str
    ) -> None:
        """Write connectivity section (bonds, angles, dihedrals, impropers).

        Uses merged type labels to ensure type_id consistency.
        Converts atom indices to atom IDs using the index->ID mapping from atoms.

        Args:
            lines: List of lines to append to
            frame: Frame containing connectivity data
            section_name: Name of the connectivity section (bonds, angles, etc.)

        Raises:
            ValueError: If atoms are missing 'id' field
        """
        lines.append(section_name.capitalize())
        lines.append("")

        atoms_data = frame["atoms"]

        # Require that all atoms have an 'id' field
        if "id" not in atoms_data:
            raise ValueError(
                "Atoms in frame must have 'id' field. "
                "This field is required for LAMMPS output to map indices to atom IDs."
            )

        # Build index to ID mapping
        # Frame uses index (0-based), we need to map to atom ID
        index_to_id = {}
        # Also build ID set for validation
        atom_ids_set = set()
        for idx in range(len(atoms_data["type"])):
            atom_id = int(atoms_data["id"][idx])
            index_to_id[idx] = atom_id
            atom_ids_set.add(atom_id)

        merged_types = self._get_merged_type_labels(frame)

        # Map section name to type key
        type_key_mapping = {
            "bonds": "bond_types",
            "angles": "angle_types",
            "dihedrals": "dihedral_types",
            "impropers": "improper_types",
        }
        type_key = type_key_mapping.get(section_name, "")

        # Get merged type list for this section
        type_list = merged_types.get(type_key, [])

        # Build type_to_id mapping from merged types
        type_to_id = self._get_type_to_id_mapping(type_list)

        if not type_list:
            # Fallback: use actual types if no merged types
            data = frame[section_name]
            unique_types = np.unique(data["type"])
            type_list = sorted([str(t) for t in unique_types])
            type_to_id = self._get_type_to_id_mapping(type_list)

        data = frame[section_name]

        # Validate that 'type' field exists
        if "type" not in data:
            raise ValueError(
                f"{section_name.capitalize()} data must have 'type' field. "
                f"Available fields: {list(data.keys())}"
            )

        # Get number of items from the data block (not from type length)
        n_items = data.nrows

        # Validate that all required atom index fields exist
        if section_name == "bonds":
            if "atom_i" not in data or "atom_j" not in data:
                raise ValueError(
                    f"Bonds must have 'atom_i' and 'atom_j' fields (0-based atom indices)"
                )
        elif section_name == "angles":
            if "atom_i" not in data or "atom_j" not in data or "atom_k" not in data:
                raise ValueError(
                    f"Angles must have 'atom_i', 'atom_j', and 'atom_k' fields (0-based atom indices)"
                )
        elif section_name in ["dihedrals", "impropers"]:
            if (
                "atom_i" not in data
                or "atom_j" not in data
                or "atom_k" not in data
                or "atom_l" not in data
            ):
                raise ValueError(
                    f"{section_name.capitalize()} must have 'atom_i', 'atom_j', 'atom_k', and 'atom_l' "
                    f"fields (0-based atom indices)"
                )

        # Validate that type field has the same length as atom index fields
        if len(data["type"]) != n_items:
            raise ValueError(
                f"{section_name.capitalize()} 'type' field has {len(data['type'])} values, "
                f"but expected {n_items} (based on atom index fields)"
            )

        for idx in range(n_items):
            item_id = idx + 1
            item_type_str = str(data["type"][idx])
            # Use merged type_id mapping, fallback to old behavior
            if item_type_str in type_to_id:
                item_type = type_to_id[item_type_str]
            else:
                # Fallback: compute on the fly if not in merged list
                unique_types = np.unique(data["type"])
                fallback_mapping = {
                    str(t): type_idx + 1
                    for type_idx, t in enumerate(sorted(unique_types))
                }
                item_type = fallback_mapping.get(item_type_str, 1)

            if section_name == "bonds":
                # Convert indices to IDs
                atom1_idx = int(data["atom_i"][idx])
                atom2_idx = int(data["atom_j"][idx])

                # Validate indices before converting - raise error if invalid
                if atom1_idx not in index_to_id:
                    raise ValueError(
                        f"Bond {idx + 1}: atom_i index {atom1_idx} is out of range. "
                        f"Valid indices: 0-{len(atoms_data) - 1}"
                    )
                if atom2_idx not in index_to_id:
                    raise ValueError(
                        f"Bond {idx + 1}: atom_j index {atom2_idx} is out of range. "
                        f"Valid indices: 0-{len(atoms_data) - 1}"
                    )

                atom1_id = index_to_id[atom1_idx]
                atom2_id = index_to_id[atom2_idx]
                lines.append(f"{item_id} {item_type} {atom1_id} {atom2_id}")
            elif section_name == "angles":
                # Convert indices to IDs
                atom1_idx = int(data["atom_i"][idx])
                atom2_idx = int(data["atom_j"][idx])
                atom3_idx = int(data["atom_k"][idx])

                # Validate indices before converting - raise error if invalid
                if atom1_idx not in index_to_id:
                    raise ValueError(
                        f"Angle {idx + 1}: atom_i index {atom1_idx} is out of range. "
                        f"Valid indices: 0-{len(atoms_data) - 1}"
                    )
                if atom2_idx not in index_to_id:
                    raise ValueError(
                        f"Angle {idx + 1}: atom_j index {atom2_idx} is out of range. "
                        f"Valid indices: 0-{len(atoms_data) - 1}"
                    )
                if atom3_idx not in index_to_id:
                    raise ValueError(
                        f"Angle {idx + 1}: atom_k index {atom3_idx} is out of range. "
                        f"Valid indices: 0-{len(atoms_data) - 1}"
                    )

                atom1_id = index_to_id[atom1_idx]
                atom2_id = index_to_id[atom2_idx]
                atom3_id = index_to_id[atom3_idx]
                lines.append(f"{item_id} {item_type} {atom1_id} {atom2_id} {atom3_id}")
            elif section_name in ["dihedrals", "impropers"]:
                # Convert indices to IDs
                atom1_idx = int(data["atom_i"][idx])
                atom2_idx = int(data["atom_j"][idx])
                atom3_idx = int(data["atom_k"][idx])
                atom4_idx = int(data["atom_l"][idx])

                # Validate indices before converting - raise error if invalid
                if atom1_idx not in index_to_id:
                    raise ValueError(
                        f"{section_name.capitalize()} {idx + 1}: atom_i index {atom1_idx} is out of range. "
                        f"Valid indices: 0-{len(atoms_data) - 1}"
                    )
                if atom2_idx not in index_to_id:
                    raise ValueError(
                        f"{section_name.capitalize()} {idx + 1}: atom_j index {atom2_idx} is out of range. "
                        f"Valid indices: 0-{len(atoms_data) - 1}"
                    )
                if atom3_idx not in index_to_id:
                    raise ValueError(
                        f"{section_name.capitalize()} {idx + 1}: atom_k index {atom3_idx} is out of range. "
                        f"Valid indices: 0-{len(atoms_data) - 1}"
                    )
                if atom4_idx not in index_to_id:
                    raise ValueError(
                        f"{section_name.capitalize()} {idx + 1}: atom_l index {atom4_idx} is out of range. "
                        f"Valid indices: 0-{len(atoms_data) - 1}"
                    )

                atom1_id = index_to_id[atom1_idx]
                atom2_id = index_to_id[atom2_idx]
                atom3_id = index_to_id[atom3_idx]
                atom4_id = index_to_id[atom4_idx]
                lines.append(
                    f"{item_id} {item_type} {atom1_id} {atom2_id} {atom3_id} {atom4_id}"
                )
        lines.append("")
