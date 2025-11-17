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

from molpy import Block, Box, ForceField, Frame

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
        headers = {
            "bond": ["id", "type", "atom_i", "atom_j"],
            "angle": ["id", "type", "atom_i", "atom_j", "atom_k"],
            "dihedral": ["id", "type", "atom_i", "atom_j", "atom_k", "atom_l"],
            "improper": ["id", "type", "atom_i", "atom_j", "atom_k", "atom_l"],
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

        # Convert LAMMPS 1-based atom indices to 0-based
        if block.nrows > 0:
            if "atom_i" in block:
                block["atom_i"] = block["atom_i"].astype(int) - 1
            if "atom_j" in block:
                block["atom_j"] = block["atom_j"].astype(int) - 1
            if "k" in block:
                block["k"] = block["k"].astype(int) - 1
            if "l" in block:
                block["l"] = block["l"].astype(int) - 1

        return block


class LammpsDataWriter(DataWriter):
    """Modern LAMMPS data file writer using Block.to_csv approach."""

    def __init__(self, path: str | Path, atom_style: str = "full") -> None:
        super().__init__(Path(path))
        self.atom_style = atom_style

    def write(self, frame: Frame) -> None:
        """Write Frame to LAMMPS data file."""
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
        """Write type count lines."""
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
        """Write masses section."""
        lines.append("Masses")
        lines.append("")

        atoms_data = frame["atoms"]
        unique_types = np.unique(atoms_data["type"])
        type_to_id = {t: i + 1 for i, t in enumerate(sorted(unique_types))}

        for atom_type in sorted(unique_types):
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

            lines.append(f"{type_to_id[atom_type]} {mass:.6f}")

        lines.append("")

    def _write_type_labels_sections(self, lines: list[str], frame: Frame) -> None:
        """Write type labels sections if needed."""
        sections = [
            ("atoms", "Atom Type Labels"),
            ("bonds", "Bond Type Labels"),
            ("angles", "Angle Type Labels"),
            ("dihedrals", "Dihedral Type Labels"),
            ("impropers", "Improper Type Labels"),
        ]

        for block_name, section_name in sections:
            if block_name in frame and frame[block_name].nrows > 0:
                types = frame[block_name]["type"]
                if self._needs_type_labels(types):
                    lines.append(section_name)
                    lines.append("")

                    unique_types = np.unique(types)
                    type_to_id = {t: i + 1 for i, t in enumerate(sorted(unique_types))}

                    for type_label in sorted(unique_types):
                        lines.append(f"{type_to_id[type_label]} {type_label}")

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
        """Write atoms section."""
        lines.append("Atoms")
        lines.append("")

        atoms_data = frame["atoms"]
        unique_types = np.unique(atoms_data["type"])
        type_to_id = {t: i + 1 for i, t in enumerate(sorted(unique_types))}

        for i in range(len(atoms_data["type"])):
            atom_id = i + 1
            atom_type = type_to_id[atoms_data["type"][i]]

            # Get coordinates - prefer xyz field, fallback to x/y/z
            if "xyz" in atoms_data and atoms_data["xyz"][i] is not None:
                xyz = atoms_data["xyz"][i]
                x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            else:
                x = float(atoms_data["x"][i])
                y = float(atoms_data["y"][i])
                z = float(atoms_data["z"][i])

            if self.atom_style == "full":
                mol_id = int(atoms_data["mol"][i])
                charge = float(atoms_data["q"][i])
                lines.append(
                    f"{atom_id} {mol_id} {atom_type} {charge:.6f} {x:.6f} {y:.6f} {z:.6f}"
                )
            elif self.atom_style == "charge":
                charge = float(atoms_data["q"][i]) if "q" in atoms_data else 0.0
                lines.append(
                    f"{atom_id} {atom_type} {charge:.6f} {x:.6f} {y:.6f} {z:.6f}"
                )
            else:  # atomic
                lines.append(f"{atom_id} {atom_type} {x:.6f} {y:.6f} {z:.6f}")

        lines.append("")

    def _write_connectivity_section(
        self, lines: list[str], frame: Frame, section_name: str
    ) -> None:
        """Write connectivity section (bonds, angles, dihedrals, impropers)."""
        lines.append(section_name.capitalize())
        lines.append("")

        data = frame[section_name]
        unique_types = np.unique(data["type"])
        type_to_id = {t: i + 1 for i, t in enumerate(sorted(unique_types))}

        for idx in range(len(data["type"])):
            item_id = idx + 1
            item_type = type_to_id[data["type"][idx]]

            if section_name == "bonds":
                atom_i = int(data["atom_i"][idx]) + 1
                atom_j = int(data["atom_j"][idx]) + 1
                lines.append(f"{item_id} {item_type} {atom_i} {atom_j}")
            elif section_name == "angles":
                atom_i = int(data["atom_i"][idx]) + 1
                atom_j = int(data["atom_j"][idx]) + 1
                atom_k = int(data["atom_k"][idx]) + 1
                lines.append(f"{item_id} {item_type} {atom_i} {atom_j} {atom_k}")
            elif section_name in ["dihedrals", "impropers"]:
                atom_i = int(data["atom_i"][idx]) + 1
                atom_j = int(data["atom_j"][idx]) + 1
                atom_k = int(data["atom_k"][idx]) + 1
                atom_l = int(data["atom_l"][idx]) + 1
                lines.append(
                    f"{item_id} {item_type} {atom_i} {atom_j} {atom_k} {atom_l}"
                )
        lines.append("")
