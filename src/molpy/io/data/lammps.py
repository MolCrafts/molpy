"""
Modern LAMMPS data file I/O using Block.from_csv.

This module provides a clean, imperative approach to reading and writing
LAMMPS data files using the Block.from_csv functionality.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from molrs import Block, MetaValue
from molpy._frame_meta import update_frame_meta
from molpy.core.box import Box
from molpy.core.fields import CHARGE, MOL_ID, FieldFormatter
from molpy.core.forcefield import AtomisticForcefield, ForceField
from molrs import Frame

from .base import DataReader, DataWriter


class LammpsFieldFormatter(FieldFormatter):
    """LAMMPS-specific field name translation.

    Maps LAMMPS atom_style column names to canonical field names::

        "q"   → "charge"
        "mol" → "mol_id"
    """

    _field_formatters = {
        "q": CHARGE,
        "mol": MOL_ID,
    }


@dataclass(frozen=True, slots=True)
class LammpsDataResult:
    """Explicit products of parsing one LAMMPS data file."""

    frame: Frame
    forcefield: ForceField
    counts: dict[str, int]
    type_labels: dict[str, list[str]]


class LammpsDataReader(DataReader[LammpsDataResult]):
    """Modern LAMMPS data file reader using Block.from_csv."""

    def __init__(self, path: str | Path, atom_style: str = "full") -> None:
        super().__init__(Path(path))
        self.atom_style = atom_style

    def read(self, frame: Frame | None = None) -> LammpsDataResult:
        """Read a LAMMPS data file into explicit frame and format products."""
        frame = frame or Frame()

        # Read and parse the file
        lines = self._read_lines()
        sections = self._extract_sections(lines)

        # Parse header and set up box
        header_info = self._parse_header(sections.get("header", []))
        frame.box = self._create_box(
            header_info["box_bounds"], header_info.get("tilts")
        )

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

        # Build id to index mapping for connectivity
        id_to_idx = {}
        if "atoms" in frame:
            for i, atom_id in enumerate(frame["atoms"]["id"]):
                id_to_idx[int(atom_id)] = i

        # Parse connectivity sections
        if "Bonds" in sections and header_info["counts"].get("bonds", 0) > 0:
            frame["bonds"] = self._parse_connectivity_section(
                sections["Bonds"], "bond", type_labels.get("bond", {}), id_to_idx
            )

        if "Angles" in sections and header_info["counts"].get("angles", 0) > 0:
            frame["angles"] = self._parse_connectivity_section(
                sections["Angles"], "angle", type_labels.get("angle", {}), id_to_idx
            )

        if "Dihedrals" in sections and header_info["counts"].get("dihedrals", 0) > 0:
            frame["dihedrals"] = self._parse_connectivity_section(
                sections["Dihedrals"],
                "dihedral",
                type_labels.get("dihedral", {}),
                id_to_idx,
            )

        if "Impropers" in sections and header_info["counts"].get("impropers", 0) > 0:
            frame["impropers"] = self._parse_connectivity_section(
                sections["Impropers"],
                "improper",
                type_labels.get("improper", {}),
                id_to_idx,
            )

        # Store exact-dtype scalar provenance on the Frame.
        update_frame_meta(
            frame,
            {
                "format": MetaValue("string", "lammps_data"),
                "atom_style": MetaValue("string", self.atom_style),
                "source_file": MetaValue("string", str(self._path)),
            },
        )

        # Translate format-specific field names to canonical names
        self._formatter.canonicalize_frame(frame)

        return LammpsDataResult(
            frame=frame,
            forcefield=forcefield,
            counts=dict(header_info["counts"]),
            type_labels={
                f"{key}_types": [labels[index] for index in sorted(labels)]
                for key, labels in type_labels.items()
            },
        )

    _formatter = LammpsFieldFormatter()

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
            "bodies",
            "velocities",
            "ellipsoids",
            "lines",
            "triangles",
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
        box_bounds: dict[str, tuple[float, float]] = {}
        tilts: tuple[float, float, float] | None = None

        for line in header_lines:
            parts = line.split()
            if len(parts) < 2:
                continue

            line_lower = line.lower()
            tokens_lower = [p.lower() for p in parts]

            # Box-bound lines have float values; parse them before the int-count branch.
            if "xlo" in tokens_lower and "xhi" in tokens_lower:
                box_bounds["x"] = self._parse_box_bound(parts, line)
                continue
            if "ylo" in tokens_lower and "yhi" in tokens_lower:
                box_bounds["y"] = self._parse_box_bound(parts, line)
                continue
            if "zlo" in tokens_lower and "zhi" in tokens_lower:
                box_bounds["z"] = self._parse_box_bound(parts, line)
                continue

            # Triclinic tilt factors: "xy xz yz <value-x> <value-y> <value-z>"
            # with values written first per LAMMPS convention.
            if "xy" in tokens_lower and "xz" in tokens_lower and "yz" in tokens_lower:
                try:
                    tilts = (float(parts[0]), float(parts[1]), float(parts[2]))
                except (ValueError, IndexError) as exc:
                    raise ValueError(
                        f"Failed to parse LAMMPS tilt factors line: {line!r}"
                    ) from exc
                continue

            try:
                count = int(parts[0])
            except ValueError:
                continue

            if "atoms" in line_lower and not line_lower.startswith("atoms"):
                counts["atoms"] = count
            elif "bonds" in line_lower and not line_lower.startswith("bonds"):
                counts["bonds"] = count
            elif "angles" in line_lower and not line_lower.startswith("angles"):
                counts["angles"] = count
            elif "dihedrals" in line_lower and not line_lower.startswith("dihedrals"):
                counts["dihedrals"] = count
            elif "impropers" in line_lower and not line_lower.startswith("impropers"):
                counts["impropers"] = count
            elif "atom types" in line_lower:
                counts["atom_types"] = count
            elif "bond types" in line_lower:
                counts["bond_types"] = count
            elif "angle types" in line_lower:
                counts["angle_types"] = count
            elif "dihedral types" in line_lower:
                counts["dihedral_types"] = count
            elif "improper types" in line_lower:
                counts["improper_types"] = count

        return {
            "counts": counts,
            "box_bounds": box_bounds if box_bounds else None,
            "tilts": tilts,
        }

    @staticmethod
    def _parse_box_bound(parts: list[str], line: str) -> tuple[float, float]:
        """Parse a `lo hi <axis>lo <axis>hi` line into a (lo, hi) tuple."""
        try:
            return (float(parts[0]), float(parts[1]))
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Failed to parse LAMMPS box bounds line: {line!r}"
            ) from exc

    def _create_box(
        self,
        box_bounds: dict[str, tuple[float, float]] | None,
        tilts: tuple[float, float, float] | None = None,
    ) -> Box:
        """Create Box from parsed bounds. Raises if any axis is missing."""
        required_axes = ("x", "y", "z")
        missing = (
            list(required_axes)
            if not box_bounds
            else [axis for axis in required_axes if axis not in box_bounds]
        )
        if missing:
            raise ValueError(
                f"LAMMPS data file {self._path} is missing box bounds for axis "
                f"{missing}. Required header lines: 'xlo xhi', 'ylo yhi', 'zlo zhi'."
            )

        lengths = np.array(
            [box_bounds[axis][1] - box_bounds[axis][0] for axis in required_axes]
        )
        origin = np.array([box_bounds[axis][0] for axis in required_axes])
        if tilts is not None and any(t != 0.0 for t in tilts):
            return Box.tric(lengths, tilts, origin=origin)
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
        """Parse LAMMPS ``*Coeffs`` sections into a :class:`ForceField`.

        Each coefficient line is keyed by its 1-indexed numeric type id and the
        per-type parameters are stored on a :class:`Type` named ``"<cat>_<id>"``
        — the exact shape :class:`LammpsDataWriter` reads back, so read→write
        round-trips the parameters instead of silently dropping them.

        Malformed (non-numeric / too-short) lines raise ``ValueError`` rather
        than being swallowed; reading is total.
        """
        forcefield = AtomisticForcefield()

        # (section, category, def_style, canonical param names). LAMMPS coeff
        # arity is style-dependent and not declared in the data file, so the
        # names are applied positionally to however many params each line
        # carries (a 2-param harmonic improper and a 3-param cvff improper both
        # parse). The leading column is always the integer type id.
        # arity = number of atom-type endpoints molrs requires in the dash-form
        # type name for each category. LAMMPS data coeff sections are keyed only
        # by an integer type id (no atom-type names), so a synthetic dash name
        # ``"<id>-<id>..."`` of the right arity is used; :class:`LammpsDataWriter`
        # reads the id back via ``name.split("-")[0]``.
        coeff_specs = [
            (
                "PairCoeffs",
                "pair",
                lambda: forcefield.def_pairstyle("lj/cut"),
                ["epsilon", "sigma"],
                1,
            ),
            (
                "BondCoeffs",
                "bond",
                lambda: forcefield.def_bondstyle("harmonic"),
                ["k", "r0"],
                2,
            ),
            (
                "AngleCoeffs",
                "angle",
                lambda: forcefield.def_anglestyle("harmonic"),
                ["k", "theta0"],
                3,
            ),
            (
                "DihedralCoeffs",
                "dihedral",
                lambda: forcefield.def_dihedralstyle("harmonic"),
                ["k", "d", "n"],
                4,
            ),
            (
                "ImproperCoeffs",
                "improper",
                lambda: forcefield.def_improperstyle("harmonic"),
                ["k", "d", "n"],
                4,
            ),
        ]

        for section, category, make_style, param_names, arity in coeff_specs:
            if section not in sections:
                continue
            style = make_style()
            for line in sections[section]:
                if not line.strip():
                    continue
                parts = line.split()
                try:
                    type_id = int(parts[0])
                    values = [float(x) for x in parts[1:]]
                except (ValueError, IndexError) as e:
                    raise ValueError(
                        f"malformed {section} line {line!r}: expected an integer "
                        f"type id followed by numeric parameters ({e})"
                    ) from e
                params = dict(zip(param_names, values))
                type_name = "-".join([str(type_id)] * arity)
                forcefield.def_type(category, style.name, type_name, params)

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
        elif self.atom_style == "body":
            header = ["id", "type", "bodyflag", "mass", "x", "y", "z"]
        elif self.atom_style == "atomic":
            header = ["id", "type", "x", "y", "z"]
        else:
            raise ValueError(
                f"Unsupported LAMMPS atom_style {self.atom_style!r}. "
                "Supported: 'full', 'charge', 'atomic', 'body'."
            )

        csv_lines.append(" ".join(header))

        # Add data lines, truncating any trailing image flags (ix iy iz) or
        # body-specific extras so columns line up with the header.
        n_cols = len(header)
        for line in atom_lines:
            parts = line.split()
            if len(parts) >= n_cols:
                csv_lines.append(" ".join(parts[:n_cols]))

        # Parse using Block.from_csv with space delimiter
        csv_string = "\n".join(csv_lines)
        block = Block.from_csv(
            StringIO(csv_string), delimiter=" ", skipinitialspace=True
        )

        # Add mass information. atom_style="body" already carries per-atom
        # mass in the atoms section; do not overwrite it from the (absent)
        # Masses section. Other styles look mass up by atom type, falling
        # back to 1.0 when the file has no Masses section.
        if block.nrows > 0 and "mass" not in block:
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
        self,
        lines: list[str],
        section_type: str,
        type_labels: dict[int, str],
        id_to_idx: dict[int, int],
    ) -> Block:
        """Parse connectivity sections (bonds, angles, dihedrals, impropers)."""
        if not lines:
            return Block()

        # Define temporary headers for parsing (using atom1, atom2 etc. as placeholders)
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

        # Parse using Block.from_csv
        csv_string = "\n".join(csv_lines)
        block = Block.from_csv(
            StringIO(csv_string), delimiter=" ", skipinitialspace=True
        )

        # Map atom IDs to 0-based indices and rename columns
        atom_key_map = {
            "atom1": "atomi",
            "atom2": "atomj",
            "atom3": "atomk",
            "atom4": "atoml",
        }

        for old_key, new_key in atom_key_map.items():
            if old_key in block:
                # Convert IDs to indices using the provided mapping
                ids = block[old_key].astype(int)
                indices = np.array([id_to_idx.get(id_val, -1) for id_val in ids])

                # Check for unmapped IDs (on the signed array, -1 sentinel intact).
                if np.any(indices == -1):
                    unmapped = ids[indices == -1]
                    import warnings

                    warnings.warn(
                        f"Found {len(unmapped)} atom IDs in {section_type} section "
                        f"that could not be mapped to atom indices: {unmapped[:5]}..."
                    )

                # Store endpoints as uint32 — the canonical dtype molrs uses for
                # relation endpoints (matching Atomistic.to_frame). A signed-int
                # column is silently ignored by molrs.from_frame, which would drop
                # every bond/angle/dihedral on the Frame->Atomistic round-trip.
                block[new_key] = indices.astype(np.uint32)
                del block[old_key]

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
    - Bonds/Angles/Dihedrals: Use atom indices in 'atomi', 'atomj', 'atomk', 'atoml'
      (from to_frame()). These are 0-based indices that will be converted to 1-based atom IDs.
    """

    def __init__(
        self,
        path: str | Path,
        atom_style: str = "full",
        *,
        type_labels: dict[str, list[str]] | None = None,
        forcefield: ForceField | None = None,
    ) -> None:
        super().__init__(Path(path))
        self.atom_style = atom_style
        self.type_labels = {
            key: list(labels) for key, labels in (type_labels or {}).items()
        }
        self.forcefield = forcefield

    _formatter = LammpsFieldFormatter()

    def write(self, frame: Frame) -> None:
        """Write Frame to LAMMPS data file.

        Args:
            frame: Frame containing atoms and optionally bonds/angles/dihedrals.
                  Atoms must have 'id' field.

        Raises:
            ValueError: If atoms are missing 'id' field.
        """
        # The LAMMPS data file is positional (``atom-ID mol-ID type q x y z``),
        # so the writer reads canonical columns (``charge``, ``mol_id``) straight
        # into the right slots — no canonical→format rename, hence no frame copy.

        lines = []

        # Header
        lines.append(
            f"# LAMMPS data file written by molpy on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        # CL&Pol Drude systems: emit the fix-drude flag string here, where the
        # atom-type → ID ordering is known. Ready to paste into the input script.
        drude_flags = self._drude_flag_string(frame)
        if drude_flags:
            lines.append(f"# CL&Pol Drude — paste into input script:")
            lines.append(f"#   fix DRUDE all drude {drude_flags}")
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

        # Force-field coefficients are an explicit format input, never Frame meta.
        self._write_force_field_coeffs_sections(lines)

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

    #: Connectivity blocks whose entries each carry a force-field type id.
    _CONNECTIVITY = ("bonds", "angles", "dihedrals", "impropers")

    def _get_counts(self, frame: Frame) -> dict[str, int]:
        """Get section counts from frame.

        A connectivity block that has entries but no ``type`` column is untyped
        topology — a relation kind the force field never parameterized (e.g.
        OPLS-AA defines no improper typifier). LAMMPS cannot represent untyped
        connectivity, so such a block is omitted from the data file (and from the
        header count, so the declared count matches the emitted section) with a
        warning. A *present* ``type`` column holding bad values is a real error,
        caught later in :meth:`_collect_actual_types`.
        """
        counts = {}
        if "atoms" in frame:
            counts["atoms"] = frame["atoms"].nrows
        for name in self._CONNECTIVITY:
            if name not in frame:
                continue
            block = frame[name]
            if block.nrows > 0 and "type" not in block:
                warnings.warn(
                    f"{name!r} block has {block.nrows} entries but no 'type' "
                    f"column; these are untyped topology (a relation kind the "
                    f"force field does not parameterize) and are omitted from the "
                    f"LAMMPS data file.",
                    stacklevel=2,
                )
                continue
            counts[name] = block.nrows
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
                # Untyped connectivity (no 'type' column) is omitted from the
                # data file — see _get_counts. Skip here so it contributes no
                # type labels rather than crashing on the missing column.
                if "type" not in frame[block_name]:
                    continue
                types = frame[block_name]["type"]
                # Check for empty strings or None values - these are errors
                for t in types:
                    if t is None:
                        raise ValueError(
                            f"Found None type in {block_name} block. "
                            f"All {block_name} must have valid type values."
                        )
                    if isinstance(t, str) and not t.strip():
                        raise ValueError(
                            f"Found empty string type in {block_name} block. "
                            f"All {block_name} must have non-empty type values."
                        )
                # Convert to string set, handling both string and numeric types
                unique_types = set(str(t) for t in np.unique(types))
                actual_types[type_key] = unique_types

        return actual_types

    def _merge_type_labels(
        self, actual_types: dict[str, set[str]]
    ) -> dict[str, list[str]]:
        """Merge explicit format labels with the types present in the Frame."""
        merged: dict[str, list[str]] = {}
        for key in (
            "atom_types",
            "bond_types",
            "angle_types",
            "dihedral_types",
            "improper_types",
        ):
            explicit = self.type_labels.get(key, [])
            for label in explicit:
                if not label or not str(label).strip():
                    raise ValueError(f"Found empty explicit type label for {key}")
            labels = set(explicit) | actual_types.get(key, set())
            if labels:
                merged[key] = sorted(labels)
        return merged

    def _get_merged_type_labels(self, frame: Frame) -> dict[str, list[str]]:
        """Get type labels from explicit format input and Frame blocks.

        Returns:
            Dict mapping type keys to sorted lists of type names
        """
        actual_types = self._collect_actual_types(frame)
        label_types: dict[str, set[str]] = {}
        type_key_mapping = {
            "atoms": "atom_types",
            "bonds": "bond_types",
            "angles": "angle_types",
            "dihedrals": "dihedral_types",
            "impropers": "improper_types",
        }
        for block_name, type_key in type_key_mapping.items():
            if block_name not in frame or frame[block_name].nrows == 0:
                continue
            if "type" not in frame[block_name]:
                continue
            types = frame[block_name]["type"]
            if self._needs_type_labels(types) or type_key in self.type_labels:
                label_types[type_key] = actual_types[type_key]
        return self._merge_type_labels(label_types)

    def _drude_flag_string(self, frame: Frame) -> str | None:
        """Build the ``fix drude`` C/D/N flag string, or None if not Drude.

        Emits one flag per atom type in the file's (sorted) type-ID order — the
        ordering the LAMMPS DRUDE package's ``fix drude`` consumes: ``D`` for a
        Drude shell type (element ``D``), ``C`` for a polarizable core (an atom
        joined to a shell by a ``drude`` spring bond), ``N`` otherwise.
        """
        if "atoms" not in frame:
            return None
        atoms = frame["atoms"]
        if "element" not in atoms or "type" not in atoms:
            return None
        elements = np.asarray(atoms["element"]).astype(str)
        if not np.any(elements == "D"):
            return None  # no Drude particles → ordinary system

        types = np.asarray(atoms["type"]).astype(str)
        shell_types = set(types[elements == "D"].tolist())

        core_types: set[str] = set()
        bonds = frame["bonds"] if "bonds" in frame else None
        if bonds is not None and bonds.nrows > 0 and "style" in bonds:
            b_style = np.asarray(bonds["style"]).astype(str)
            b_i = np.asarray(bonds["atomi"]).astype(int)
            b_j = np.asarray(bonds["atomj"]).astype(int)
            for k in np.flatnonzero(b_style == "drude"):
                i, j = int(b_i[k]), int(b_j[k])
                core = i if elements[i] != "D" else j
                core_types.add(str(types[core]))

        merged = self._get_merged_type_labels(frame)
        ordered = merged.get("atom_types") or sorted(set(types.tolist()))
        flags = [
            "D" if t in shell_types else "C" if t in core_types else "N"
            for t in ordered
        ]
        return " ".join(flags)

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
        """Write type counts derived directly from Frame blocks."""
        merged = self._get_merged_type_labels(frame)
        labels = {
            "atoms": ("atom", "atom_types"),
            "bonds": ("bond", "bond_types"),
            "angles": ("angle", "angle_types"),
            "dihedrals": ("dihedral", "dihedral_types"),
            "impropers": ("improper", "improper_types"),
        }
        for block_name, (label, type_key) in labels.items():
            if type_key in merged:
                lines.append(f"{len(merged[type_key])} {label} types")
                continue
            if block_name in frame and "type" in frame[block_name]:
                unique_types = np.unique(frame[block_name]["type"])
                if block_name == "atoms" or len(unique_types) > 0:
                    lines.append(f"{len(unique_types)} {label} types")

    def _write_box_bounds(self, lines: list[str], frame: Frame) -> None:
        """Write box bounds."""
        if frame.box is not None:
            box = frame.box
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
                from molrs import Element

                element_symbol = atoms_data["element"][mask][0]
                try:
                    mass = Element(element_symbol).mass
                except KeyError:
                    # Non-physical element (e.g. a Drude shell "D") has no
                    # periodic-table entry; use the stored per-atom mass.
                    mass = atoms_data["mass"][mask][0] if "mass" in atoms_data else 1.0
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
        """Write type-label sections inferred from the Frame blocks."""
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
                # Validate: no empty strings should exist at this point
                for type_label in type_list:
                    if not type_label or not str(type_label).strip():
                        raise ValueError(
                            f"Found empty type label in {type_key}. "
                            f"This should have been caught earlier. "
                            f"All type labels must be non-empty strings."
                        )
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

    def _write_force_field_coeffs_sections(self, lines: list[str]) -> None:
        """Write coefficients from the writer's explicit ForceField input."""
        if self.forcefield is None:
            return

        from molpy import (
            AngleStyle,
            BondStyle,
            DihedralStyle,
            ImproperStyle,
            PairStyle,
        )

        configs = (
            (PairStyle, "Pair Coeffs", ("epsilon", "sigma"), (0.0, 1.0)),
            (BondStyle, "Bond Coeffs", ("k", "r0"), (0.0, 1.0)),
            (AngleStyle, "Angle Coeffs", ("k", "theta0"), (0.0, 0.0)),
            (DihedralStyle, "Dihedral Coeffs", ("k", "d", "n"), (0.0, 1, 1)),
            (ImproperStyle, "Improper Coeffs", ("k", "d", "n"), (0.0, 1, 1)),
        )
        for style_cls, heading, keys, defaults in configs:
            styles = self.forcefield.get_styles(style_cls)
            if not styles:
                continue
            lines.extend((heading, ""))
            for style in styles:
                for type_obj in style.types:
                    type_id = int(type_obj.name.split("-")[0])
                    values = [
                        type_obj.get(key, default)
                        for key, default in zip(keys, defaults, strict=True)
                    ]
                    formatted = [
                        f"{value:.6f}" if isinstance(value, float) else str(value)
                        for value in values
                    ]
                    lines.append(f"{type_id} {' '.join(formatted)}")
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

        # Materialise every column ONCE (each ``atoms_data[col]`` view rebuilds
        # the whole molrs column — indexing it per row is O(N^2), catastrophic
        # for the string ``type`` column on a large packed system).
        ids = np.asarray(atoms_data["id"])
        types = np.asarray(atoms_data["type"])
        xs = np.asarray(atoms_data["x"], dtype=float)
        ys = np.asarray(atoms_data["y"], dtype=float)
        zs = np.asarray(atoms_data["z"], dtype=float)
        mol_ids = charges = None
        if self.atom_style in ("full",):
            mol_ids = np.asarray(atoms_data["mol_id"])
        if self.atom_style in ("full", "charge"):
            charges = np.asarray(atoms_data["charge"], dtype=float)

        # Resolve every type id up front (unique set is tiny vs N atoms).
        fallback_mapping: dict[str, int] = {}
        if any(str(t) not in type_to_id for t in np.unique(types)):
            fallback_mapping = {
                str(t): type_idx + 1
                for type_idx, t in enumerate(sorted(np.unique(types)))
            }

        for idx in range(len(types)):
            atom_id = int(ids[idx])
            atom_type_str = str(types[idx])
            atom_type = type_to_id.get(
                atom_type_str, fallback_mapping.get(atom_type_str, 1)
            )
            x = xs[idx]
            y = ys[idx]
            z = zs[idx]

            if self.atom_style == "full":
                lines.append(
                    f"{atom_id} {int(mol_ids[idx])} {atom_type} "
                    f"{charges[idx]:.6f} {x:.6f} {y:.6f} {z:.6f}"
                )
            elif self.atom_style == "charge":
                lines.append(
                    f"{atom_id} {atom_type} {charges[idx]:.6f} {x:.6f} {y:.6f} {z:.6f}"
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

        # Build index to ID mapping (0-based frame index -> LAMMPS atom ID).
        # Materialise the id column once; per-row ``atoms_data["id"][idx]`` would
        # rebuild the whole molrs column each iteration (O(N^2)).
        atom_id_arr = np.asarray(atoms_data["id"])
        index_to_id = {i: int(v) for i, v in enumerate(atom_id_arr)}

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
            if "atomi" not in data or "atomj" not in data:
                raise ValueError(
                    f"Bonds must have 'atomi' and 'atomj' fields (0-based atom indices)"
                )
        elif section_name == "angles":
            if "atomi" not in data or "atomj" not in data or "atomk" not in data:
                raise ValueError(
                    f"Angles must have 'atomi', 'atomj', and 'atomk' fields (0-based atom indices)"
                )
        elif section_name in ["dihedrals", "impropers"]:
            if (
                "atomi" not in data
                or "atomj" not in data
                or "atomk" not in data
                or "atoml" not in data
            ):
                raise ValueError(
                    f"{section_name.capitalize()} must have 'atomi', 'atomj', 'atomk', and 'atoml' "
                    f"fields (0-based atom indices)"
                )

        # Validate that type field has the same length as atom index fields
        if len(data["type"]) != n_items:
            raise ValueError(
                f"{section_name.capitalize()} 'type' field has {len(data['type'])} values, "
                f"but expected {n_items} (based on atom index fields)"
            )

        # Materialise every column ONCE — per-row ``data[col][idx]`` on a molrs
        # Block rebuilds the whole column each iteration (O(N^2); the string
        # ``type`` column made this the dominant writer cost).
        type_arr = np.asarray(data["type"])
        ai = np.asarray(data["atomi"])
        aj = np.asarray(data["atomj"])
        ak = np.asarray(data["atomk"]) if "atomk" in data else None
        al = np.asarray(data["atoml"]) if "atoml" in data else None
        n_atoms = len(atom_id_arr)

        fallback_mapping: dict[str, int] = {}
        if any(str(t) not in type_to_id for t in np.unique(type_arr)):
            fallback_mapping = {
                str(t): type_idx + 1
                for type_idx, t in enumerate(sorted(np.unique(type_arr)))
            }

        def _id(atom_idx: int, pos: str) -> int:
            if atom_idx not in index_to_id:
                raise ValueError(
                    f"{section_name.capitalize()} {idx + 1}: {pos} index "
                    f"{atom_idx} is out of range. Valid indices: 0-{n_atoms - 1}"
                )
            return index_to_id[atom_idx]

        for idx in range(n_items):
            item_id = idx + 1
            item_type_str = str(type_arr[idx])
            item_type = type_to_id.get(
                item_type_str, fallback_mapping.get(item_type_str, 1)
            )

            if section_name == "bonds":
                a1 = _id(int(ai[idx]), "atom_i")
                a2 = _id(int(aj[idx]), "atom_j")
                lines.append(f"{item_id} {item_type} {a1} {a2}")
            elif section_name == "angles":
                a1 = _id(int(ai[idx]), "atom_i")
                a2 = _id(int(aj[idx]), "atom_j")
                a3 = _id(int(ak[idx]), "atom_k")
                lines.append(f"{item_id} {item_type} {a1} {a2} {a3}")
            elif section_name in ["dihedrals", "impropers"]:
                a1 = _id(int(ai[idx]), "atom_i")
                a2 = _id(int(aj[idx]), "atom_j")
                a3 = _id(int(ak[idx]), "atom_k")
                a4 = _id(int(al[idx]), "atom_l")
                lines.append(f"{item_id} {item_type} {a1} {a2} {a3} {a4}")
        lines.append("")
