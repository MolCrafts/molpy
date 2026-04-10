"""AMBER FRCMOD file format reading and writing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from molpy.io.utils import ensure_parent_dir


class FrcmodSection(NamedTuple):
    """Represents a section in a FRCMOD file."""

    name: str
    content: str


def read_frcmod(file: str | Path) -> dict[str, Any]:
    """Read an AMBER FRCMOD file.

    FRCMOD files contain additional force field parameters. This parser
    reads the file and returns sections as a dictionary.

    Args:
        file: Path to the FRCMOD file.

    Returns:
        Dictionary with sections: 'remark', 'mass', 'bond', 'angle', 'dihe',
        'improper', 'nonbon', and 'raw_text'.

    Example:
        >>> frcmod = read_frcmod("tfsi.frcmod")
        >>> print(frcmod['bond'])  # Raw bond section text
        >>> print(frcmod['remark'])  # Remark/comment line
    """
    file_path = Path(file)
    content = file_path.read_text()

    result = {
        "remark": "",
        "mass": "",
        "bond": "",
        "angle": "",
        "dihe": "",
        "improper": "",
        "nonbon": "",
        "raw_text": content,
    }

    lines = content.split("\n")
    current_section = None
    section_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Check for section headers
        if stripped.upper() in ("MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON"):
            # Save previous section
            if current_section is not None:
                result[current_section] = "\n".join(section_lines).strip()
            # Start new section
            current_section = stripped.lower()
            section_lines = []
        elif current_section is None and stripped:
            # First line is remark (before any section)
            if "remark" in stripped.lower():
                result["remark"] = stripped
            else:
                result["remark"] = stripped
        elif stripped:
            # Add to current section
            if current_section is not None:
                section_lines.append(line)

    # Save last section
    if current_section is not None:
        result[current_section] = "\n".join(section_lines).strip()

    return result


def write_frcmod(
    file: str | Path,
    *,
    remark: str = "",
    mass: str = "",
    bond: str = "",
    angle: str = "",
    dihe: str = "",
    improper: str = "",
    nonbon: str = "",
) -> None:
    """Write an AMBER FRCMOD file.

    Creates a properly formatted FRCMOD file with the provided sections.
    Sections can be empty strings and will still appear in the output.

    Args:
        file: Path to write the FRCMOD file to.
        remark: Optional comment/remark line.
        mass: MASS section content.
        bond: BOND section content.
        angle: ANGLE section content.
        dihe: DIHEDRAL section content.
        improper: IMPROPER section content.
        nonbon: NONBON section content.

    Example:
        >>> write_frcmod(
        ...     "my.frcmod",
        ...     remark="Custom parameters",
        ...     bond="c3-n3   300.0   1.45",
        ... )
    """
    file_path = Path(file)
    ensure_parent_dir(file_path)

    lines: list[str] = []

    # Remark
    if remark:
        # If remark already starts with "Remark", use as-is
        if remark.strip().startswith("Remark"):
            lines.append(remark.strip())
        else:
            lines.append(f"Remark line goes here {remark}".strip())
    else:
        lines.append("Remark line goes here")

    # MASS section
    lines.append("MASS")
    if mass:
        lines.append(mass)
    lines.append("")

    # BOND section
    lines.append("BOND")
    if bond:
        lines.append(bond)
    lines.append("")

    # ANGLE section
    lines.append("ANGLE")
    if angle:
        lines.append(angle)
    lines.append("")

    # DIHEDRAL section
    lines.append("DIHE")
    if dihe:
        lines.append(dihe)
    lines.append("")

    # IMPROPER section
    lines.append("IMPROPER")
    if improper:
        lines.append(improper)
    lines.append("")

    # NONBON section
    lines.append("NONBON")
    if nonbon:
        lines.append(nonbon)
    lines.append("")
    lines.append("")

    file_path.write_text("\n".join(lines))
