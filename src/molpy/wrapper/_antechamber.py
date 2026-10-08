"""Wrapper for the 'antechamber' CLI.

Higher-level workflow decisions belong in compute nodes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal


from ._wrapper import Wrapper


class AntechamberWrapper(Wrapper):
    """Wrapper for the 'antechamber' CLI."""

    tool = "antechamber"

    def run_raw(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Execute antechamber with raw arguments.

        Args:
            args: Command-line arguments (without 'antechamber').
            input_text: Text to send to stdin.
            check: If True, raise ``CalledProcessError`` on a non-zero exit.

        Returns:
            The completed process result.
        """
        return self.run(args=args, input_text=input_text, check=check)

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
        check: bool = False,
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
            check: If True, raise ``CalledProcessError`` on a non-zero exit.

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

        return self.run_raw(args=args, check=check)
