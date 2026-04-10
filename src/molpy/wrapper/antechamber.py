"""Wrapper for the 'antechamber' CLI.

Higher-level workflow decisions belong in compute nodes.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from molpy.core.atomistic import Atomistic
from molpy.core.frame import Block, Frame
from molpy.io.readers import read_amber_ac, read_mol2
from molpy.io.utils import ensure_parent_dir
from molpy.io.writers import write_pdb

from .base import Wrapper


@dataclass
class AntechamberWrapper(Wrapper):
    """Wrapper for the 'antechamber' CLI."""

    exe: str = "antechamber"

    def run_raw(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Execute antechamber with raw arguments.

        Args:
            args: Command-line arguments (without 'antechamber').
            input_text: Text to send to stdin.

        Returns:
            The completed process result.
        """
        return self.run(args=args, input_text=input_text)

    def atomtype_assign(
        self,
        input_file: str | Path,
        output_file: str | Path,
        *,
        input_format: Literal["pdb", "mol2", "ac"] = "pdb",
        output_format: Literal["mol2", "ac"] = "mol2",
        charge_method: Literal["gas", "bcc", "be3", "cm2", "esp"] = "bcc",
        atom_type: Literal["gaff", "gaff2", "amber", "sybyl"] = "gaff2",
        net_charge: int = 0,
        formal_charges: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Perform atom type assignment and charge calculation.

        This is the primary workflow for preparing ligands with antechamber:
        assigning GAFF atom types and computing partial charges.

        Args:
            input_file: Input structure file.
            output_file: Output structure file (with assigned atom types and charges).
            input_format: Input file format.
            output_format: Output file format.
            charge_method: Method for charge calculation
                (gas: Gasteiger; bcc: Bond charge correction; etc.).
            atom_type: Atom type scheme (gaff, gaff2, amber, sybyl).
            net_charge: Net charge of the molecule.
            formal_charges: If True, use formal charges instead of computing them.

        Returns:
            The completed process result.
        """
        args = [
            "-i",
            str(input_file),
            "-fi",
            input_format,
            "-o",
            str(output_file),
            "-fo",
            output_format,
            "-c",
            charge_method,
            "-at",
            atom_type,
            "-nc",
            str(net_charge),
        ]

        if formal_charges:
            args.extend(["-cf", "y"])

        return self.run_raw(args=args)


def write_antechamber_input_pdb(path: Path, atomistic: Atomistic) -> None:
    """Write a PDB suitable as antechamber input.

    This intentionally avoids using Atomistic.to_frame() which may enforce
    bond typing constraints that are irrelevant for PDB inputs.
    """

    ensure_parent_dir(path)

    atoms = list(atomistic.atoms)
    n_atoms = len(atoms)
    ids = np.arange(n_atoms, dtype=int) + 1

    def _get_str(key: str, default: str) -> list[str]:
        out: list[str] = []
        for a in atoms:
            v = a.get(key)
            out.append(default if v is None else str(v))
        return out

    def _get_int(key: str, default: int) -> np.ndarray:
        out: list[int] = []
        for a in atoms:
            v = a.get(key)
            out.append(default if v is None else int(v))
        return np.array(out, dtype=int)

    xs = np.array([float(a.get("x") or 0.0) for a in atoms], dtype=float)
    ys = np.array([float(a.get("y") or 0.0) for a in atoms], dtype=float)
    zs = np.array([float(a.get("z") or 0.0) for a in atoms], dtype=float)

    frame = Frame()
    frame["atoms"] = Block.from_dict(
        {
            "x": xs,
            "y": ys,
            "z": zs,
            "id": ids,
            "name": np.array(_get_str("name", "X"), dtype=object),
            "resName": np.array(_get_str("resName", "MOL"), dtype=object),
            "resSeq": _get_int("resSeq", 1),
            "chainID": np.array(_get_str("chainID", "A"), dtype=object),
            "element": np.array(_get_str("element", "X"), dtype=object),
        }
    )

    write_pdb(path, frame)


def read_antechamber_output(path: Path) -> Frame:
    """Read antechamber-generated output (.ac or .mol2) into a Frame."""

    suffix = path.suffix.lower()

    if suffix == ".ac":
        return read_amber_ac(path)
    if suffix == ".mol2":
        return read_mol2(path)

    raise ValueError(f"Unsupported antechamber output format: {path}")
