"""Amber prep file I/O helpers (molrs-backed).

Prep files (``.prepi``/``.prep``) define residue templates for tleap/prepgen.
Parse/serialize live in :mod:`molrs.io` (``read_prep`` / ``write_prep``); this
module keeps the historical dataclass surface for AmberTools workflows.
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
    """Write a residue to Amber prep file format (via molrs)."""
    import molrs.io

    payload = {
        "name": residue.name,
        "atoms": [
            {
                "index": a.index,
                "name": a.name,
                "atom_type": a.atom_type,
                "tree_type": a.tree_type,
                "na": a.na,
                "nb": a.nb,
                "nc": a.nc,
                "r": a.r,
                "theta": a.theta,
                "phi": a.phi,
                "charge": a.charge,
                "element": a.element,
            }
            for a in residue.atoms
        ],
        "impropers": [list(imp) for imp in (residue.impropers or [])],
    }
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    molrs.io.write_prep(str(path), payload)


def read_prep(input_file: str | Path) -> PrepResidue:
    """Read an Amber prep file (via molrs)."""
    import molrs.io

    path = Path(input_file)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = molrs.io.read_prep(str(path))
    atoms = [
        PrepAtom(
            index=int(a["index"]),
            name=str(a["name"]),
            atom_type=str(a["atom_type"]),
            tree_type=str(a.get("tree_type", "M")),
            na=int(a.get("na", 0)),
            nb=int(a.get("nb", 0)),
            nc=int(a.get("nc", 0)),
            r=float(a.get("r", 0.0)),
            theta=float(a.get("theta", 0.0)),
            phi=float(a.get("phi", 0.0)),
            charge=float(a.get("charge", 0.0)),
            element=str(a.get("element", "")),
        )
        for a in raw.get("atoms", [])
    ]
    impropers = [tuple(row) for row in raw.get("impropers", [])]
    return PrepResidue(
        name=str(raw["name"]),
        atoms=atoms,
        head_atom=raw.get("head_atom"),
        tail_atom=raw.get("tail_atom"),
        impropers=impropers,
    )
