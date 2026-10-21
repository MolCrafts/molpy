"""Wrappers for prepgen and parmchk2 AmberTools utilities.

prepgen generates residue templates (.prepi) with connection points.
parmchk2 checks and generates missing force field parameters.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

from molpy.wrapper._wrapper import Wrapper


class Parmchk2Wrapper(Wrapper):
    """Wrapper for the 'parmchk2' tool."""

    tool = "parmchk2"

    def generate_parameters(
        self,
        input_file: str | Path,
        output_file: str | Path,
        *,
        parameter_set: Literal["gaff", "gaff2"],
        input_format: Literal["mol2", "ac", "mol", "pdb"] = "mol2",
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Check and generate missing force field parameters.

        This is the primary workflow: read atom types and bonds from antechamber
        output, check against force field parameters, and generate missing ones.

        Args:
            input_file: Input structure file (mol2 or ac format, typically from antechamber).
            output_file: Output AMBER parameter file (frcmod format).
            input_format: Input file format (mol2, ac, mol, pdb).
            parameter_set: The GAFF generation whose parameters are checked
                (``-s``), molrs's ``parameter_set``: ``"gaff"`` or
                ``"gaff2"``. Required, as in molrs.
            check: Raise :class:`~molpy.wrapper.ToolError` on a non-zero exit.

        Returns:
            The completed process result.
        """
        args = [
            "-i",
            str(input_file),
            "-f",
            input_format,
            "-o",
            str(output_file),
            "-s",
            parameter_set,
        ]

        return self.run(args, check=check)


class PrepgenWrapper(Wrapper):
    """Wrapper for the 'prepgen' CLI.

    Prepgen generates AMBER residue templates (.prepi files) with connection
    points defined by control files specifying HEAD_NAME, TAIL_NAME, etc.

    Example:
        >>> wrapper = PrepgenWrapper("./work")
        >>> wrapper.generate_residue(
        ...     input_file="mol.ac",
        ...     output_file="mol.prepi",
        ...     control_file="mol.chain",
        ...     residue_name="MOL",
        ... )
    """

    tool = "prepgen"

    def generate_residue(
        self,
        input_file: str | Path,
        output_file: str | Path,
        control_file: str | Path,
        residue_name: str,
        *,
        output_format: Literal["prepi", "prepc"] = "prepi",
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Generate a residue template with connection points.

        Args:
            input_file: Input structure file (.ac format from antechamber).
            output_file: Output residue template file (.prepi).
            control_file: Control file specifying HEAD_NAME, TAIL_NAME, OMIT_NAME.
            residue_name: Name for the residue (max 4 chars recommended).
            output_format: Output format (prepi or prepc).
            check: Raise :class:`~molpy.wrapper.ToolError` on a non-zero exit.

        Returns:
            The completed process result.

        Example control file content:
            HEAD_NAME C1
            TAIL_NAME O5
            OMIT_NAME H1
            OMIT_NAME H2
            PRE_HEAD_TYPE c3
            POST_TAIL_TYPE os
            CHARGE 0
        """
        args = [
            "-i",
            str(input_file),
            "-o",
            str(output_file),
            "-f",
            output_format,
            "-m",
            str(control_file),
            "-rn",
            residue_name,
        ]

        return self.run(args, check=check)
