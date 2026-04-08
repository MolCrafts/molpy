"""Wrappers for prepgen and parmchk2 AmberTools utilities.

prepgen generates residue templates (.prepi) with connection points.
parmchk2 checks and generates missing force field parameters.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .base import Wrapper


@dataclass
class Parmchk2Wrapper(Wrapper):
    """Wrapper for the 'parmchk2' tool."""

    exe: str = "parmchk2"

    def run_raw(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Execute parmchk2 with raw arguments.

        Args:
            args: Command-line arguments (without 'parmchk2').
            input_text: Text to send to stdin.

        Returns:
            The completed process result.
        """
        return self.run(args=args, input_text=input_text)

    def generate_parameters(
        self,
        input_file: str | Path,
        output_file: str | Path,
        *,
        input_format: Literal["mol2", "ac", "mol", "pdb"] = "mol2",
        force_field: Literal["gaff", "gaff2"] = "gaff2",
    ) -> subprocess.CompletedProcess[str]:
        """Check and generate missing force field parameters.

        This is the primary workflow: read atom types and bonds from antechamber
        output, check against force field parameters, and generate missing ones.

        Args:
            input_file: Input structure file (mol2 or ac format, typically from antechamber).
            output_file: Output AMBER parameter file (frcmod format).
            input_format: Input file format (mol2, ac, mol, pdb).
            force_field: Force field parameter set to use (``-s`` flag):
                ``"gaff"`` for GAFF, ``"gaff2"`` for GAFF2 (default).

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
            force_field,
        ]

        return self.run_raw(args=args)


@dataclass
class PrepgenWrapper(Wrapper):
    """Wrapper for the 'prepgen' CLI.

    Prepgen generates AMBER residue templates (.prepi files) with connection
    points defined by control files specifying HEAD_NAME, TAIL_NAME, etc.

    Example:
        >>> wrapper = PrepgenWrapper(name="prepgen", workdir=Path("./work"))
        >>> wrapper.generate_residue(
        ...     input_file="mol.ac",
        ...     output_file="mol.prepi",
        ...     control_file="mol.chain",
        ...     residue_name="MOL",
        ... )
    """

    exe: str = "prepgen"

    def run_raw(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Execute prepgen with raw arguments.

        Args:
            args: Command-line arguments (without 'prepgen').
            input_text: Text to send to stdin.

        Returns:
            The completed process result.
        """
        return self.run(args=args, input_text=input_text)

    def generate_residue(
        self,
        input_file: str | Path,
        output_file: str | Path,
        control_file: str | Path,
        residue_name: str,
        *,
        output_format: Literal["prepi", "prepc"] = "prepi",
    ) -> subprocess.CompletedProcess[str]:
        """Generate a residue template with connection points.

        Args:
            input_file: Input structure file (.ac format from antechamber).
            output_file: Output residue template file (.prepi).
            control_file: Control file specifying HEAD_NAME, TAIL_NAME, OMIT_NAME.
            residue_name: Name for the residue (max 4 chars recommended).
            output_format: Output format (prepi or prepc).

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

        return self.run_raw(args=args)


def write_prepgen_control_file(
    path: Path,
    *,
    variant: Literal["chain", "head", "tail"],
    head_name: str | None = None,
    tail_name: str | None = None,
    head_type: str | None = None,
    tail_type: str | None = None,
    omit_names: list[str] | None = None,
    charge: int = 0,
) -> None:
    """Write a prepgen control file for residue template generation.

    Args:
        path: Output path for the control file.
        variant: Type of residue variant:
            - "chain": Both HEAD and TAIL connection points
            - "head": Only TAIL connection (for chain start)
            - "tail": Only HEAD connection (for chain end)
        head_name: Atom name for HEAD connection (required for chain/tail).
        tail_name: Atom name for TAIL connection (required for chain/head).
        head_type: Atom type for PRE_HEAD_TYPE (e.g., "c3").
        tail_type: Atom type for POST_TAIL_TYPE (e.g., "os").
        omit_names: Atom names to omit (leaving groups).
        charge: Net charge of the residue (default 0).

    Raises:
        ValueError: If required atom names are missing for the variant.

    Example:
        >>> write_prepgen_control_file(
        ...     Path("mol.chain"),
        ...     variant="chain",
        ...     head_name="C1",
        ...     tail_name="O5",
        ...     head_type="c3",
        ...     tail_type="os",
        ...     omit_names=["H1", "H2"],
        ... )
    """
    lines: list[str] = []

    if variant == "chain":
        if head_name is None or tail_name is None:
            raise ValueError("chain variant requires both head_name and tail_name")
        lines.append(f"HEAD_NAME {head_name}")
        lines.append(f"TAIL_NAME {tail_name}")
        if head_type:
            lines.append(f"PRE_HEAD_TYPE {head_type}")
        if tail_type:
            lines.append(f"POST_TAIL_TYPE {tail_type}")

    elif variant == "head":
        # Head of chain: only TAIL connection point (connects to next monomer)
        if tail_name is None:
            raise ValueError("head variant requires tail_name")
        lines.append(f"TAIL_NAME {tail_name}")
        if tail_type:
            lines.append(f"POST_TAIL_TYPE {tail_type}")

    elif variant == "tail":
        # Tail of chain: only HEAD connection point (connects to previous monomer)
        if head_name is None:
            raise ValueError("tail variant requires head_name")
        lines.append(f"HEAD_NAME {head_name}")
        if head_type:
            lines.append(f"PRE_HEAD_TYPE {head_type}")

    # Add OMIT_NAME entries for leaving groups
    if omit_names:
        for name in omit_names:
            lines.append(f"OMIT_NAME {name}")

    lines.append(f"CHARGE {charge}")

    path.write_text("\n".join(lines) + "\n")
