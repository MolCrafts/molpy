"""Wrapper for the 'antechamber' CLI.

Higher-level workflow decisions belong in the caller (a typifier or a builder).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

from molpy.wrapper._wrapper import Wrapper


class AntechamberWrapper(Wrapper):
    """Wrapper for the 'antechamber' CLI."""

    tool = "antechamber"

    def atomtype_assign(
        self,
        input_file: str | Path,
        output_file: str | Path,
        *,
        parameter_set: Literal["gaff", "gaff2", "amber", "bcc", "abcg2", "sybyl"],
        input_format: Literal["pdb", "mol2", "ac"] = "pdb",
        output_format: Literal["mol2", "ac"] = "mol2",
        charge_method: Literal[
            "gas", "bcc", "abcg2", "mul", "be3", "cm2", "esp"
        ] = "bcc",
        net_charge: int = 0,
        formal_charges: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Perform atom type assignment and charge calculation.

        This is the primary workflow for preparing ligands with antechamber:
        assigning GAFF atom types and computing partial charges.

        Args:
            input_file: Input structure file.
            output_file: Output structure file (with assigned atom types and charges).
            parameter_set: The atom-type table (``-at``), molrs's
                ``parameter_set`` (:class:`molpy.ff.typifier.AtdTypifier`):
                ``gaff``, ``gaff2``, ``amber``, ``bcc``, ``abcg2`` or ``sybyl``.
                Required, as in molrs.
            input_format: Input file format.
            output_format: Output file format.
            charge_method: Method for charge calculation (gas: Gasteiger;
                bcc / abcg2: AM1-BCC / ABCG2; mul: sqm's AM1 Mulliken
                charges, unequivalenced; etc.).
            net_charge: Net charge of the molecule.
            formal_charges: If True, use formal charges instead of computing them.
            check: Raise :class:`~molpy.wrapper.ToolError` on a non-zero exit.

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
            parameter_set,
            "-nc",
            str(net_charge),
        ]

        if formal_charges:
            args.extend(["-cf", "y"])

        return self.run(args, check=check)
