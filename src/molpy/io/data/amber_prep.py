"""Amber prep file I/O helpers.

Prep files (.prepi/.prep) define residue templates consumed by tleap/prepgen.
These helpers are colocated with wrappers to keep Amber-specific workflow code
out of the generic IO namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PrepAtom:
    """Atom entry in a prep file."""

    index: int
    name: str
    atom_type: str
    tree_type: str
    na: int
    nb: int
    nc: int
    r: float
    theta: float
    phi: float
    charge: float
    element: str = ""


@dataclass
class PrepResidue:
    """Residue definition in prep format."""

    name: str
    atoms: list[PrepAtom]
    head_atom: str | None = None
    tail_atom: str | None = None
    impropers: list[tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if self.impropers is None:
            self.impropers = []


def write_prep(residue: PrepResidue, output_file: str | Path) -> None:
    """Write a residue to Amber prep file format."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "    0    0    2",
        f"{residue.name}",
        "",
        "CORRECT     OMIT DU   BEG",
        "",
    ]

    for atom in residue.atoms:
        lines.append(
            f"{atom.index:5d} {atom.name:4s} {atom.atom_type:4s} "
            f"{atom.tree_type:1s} {atom.na:5d} {atom.nb:5d} {atom.nc:5d} "
            f"{atom.r:10.5f} {atom.theta:10.5f} {atom.phi:10.5f} "
            f"{atom.charge:10.6f}"
        )

    lines.append("")
    if residue.impropers:
        lines.append("IMPROPER")
        for improper in residue.impropers:
            lines.append(" ".join(improper))
        lines.append("")

    lines.append("DONE")
    lines.append("")
    output_path.write_text("\n".join(lines))


def read_prep(input_file: str | Path) -> PrepResidue:
    """Read an Amber prep file with the subset needed by MolPy workflow."""
    input_path = Path(input_file)
    lines = input_path.read_text().strip().split("\n")

    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    idx += 1

    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    residue_name = lines[idx].strip()
    idx += 1

    while idx < len(lines) and (
        not lines[idx].strip() or "CORRECT" in lines[idx] or "OMIT" in lines[idx]
    ):
        idx += 1

    atoms: list[PrepAtom] = []
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line.startswith("IMPROPER") or line.startswith("DONE"):
            break

        parts = line.split()
        if len(parts) >= 11:
            atoms.append(
                PrepAtom(
                    index=int(parts[0]),
                    name=parts[1],
                    atom_type=parts[2],
                    tree_type=parts[3],
                    na=int(parts[4]),
                    nb=int(parts[5]),
                    nc=int(parts[6]),
                    r=float(parts[7]),
                    theta=float(parts[8]),
                    phi=float(parts[9]),
                    charge=float(parts[10]),
                )
            )
        idx += 1

    impropers: list[tuple[str, ...]] = []
    if idx < len(lines) and "IMPROPER" in lines[idx]:
        idx += 1
        while idx < len(lines):
            line = lines[idx].strip()
            if not line or line.startswith("DONE"):
                break
            impropers.append(tuple(line.split()))
            idx += 1

    return PrepResidue(name=residue_name, atoms=atoms, impropers=impropers)
