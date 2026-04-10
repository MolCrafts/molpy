"""TLeap script generation helpers.

These are small text builders used by Amber workflows. They are colocated
with wrappers because they are tightly coupled to tleap invocation.
"""

from __future__ import annotations

from pathlib import Path


def generate_leap_script(
    force_field: str,
    prep_files: list[Path],
    sequence: list[str],
    output_prefix: str,
    additional_commands: list[str] | None = None,
) -> str:
    """Generate a tleap input script for building a polymer."""
    lines: list[str] = []

    if not force_field.startswith("leaprc."):
        force_field = f"leaprc.{force_field}"
    lines.append(f"source {force_field}")
    lines.append("")

    for prep_file in prep_files:
        lines.append(f"loadamberprep {prep_file}")
    lines.append("")

    if additional_commands:
        lines.extend(additional_commands)
        lines.append("")

    sequence_str = " ".join(sequence)
    lines.append(f"mol = sequence {{ {sequence_str} }}")
    lines.append("")

    lines.append(f"saveamberparm mol {output_prefix}.prmtop {output_prefix}.inpcrd")
    lines.append(f"savepdb mol {output_prefix}.pdb")
    lines.append("")
    lines.append("quit")

    return "\n".join(lines)


def generate_leap_script_with_ions(
    force_field: str,
    prep_files: list[Path],
    sequence: list[str],
    ions: dict[str, int] | None,
    output_prefix: str,
    box_size: tuple[float, float, float] | None = None,
) -> str:
    """Generate a tleap script with optional ion addition."""
    lines: list[str] = []

    if not force_field.startswith("leaprc."):
        force_field = f"leaprc.{force_field}"
    lines.append(f"source {force_field}")
    lines.append("")

    for prep_file in prep_files:
        lines.append(f"loadamberprep {prep_file}")
    lines.append("")

    sequence_str = " ".join(sequence)
    lines.append(f"mol = sequence {{ {sequence_str} }}")
    lines.append("")

    if ions:
        for ion_name, count in ions.items():
            lines.append(f"addIonsRand mol {ion_name} {count}")
        lines.append("")

    if box_size:
        x, y, z = box_size
        lines.append(f"set mol box {{ {x} {y} {z} }}")
        lines.append("")

    lines.append(f"saveamberparm mol {output_prefix}.prmtop {output_prefix}.inpcrd")
    lines.append(f"savepdb mol {output_prefix}.pdb")
    lines.append("")
    lines.append("quit")

    return "\n".join(lines)
