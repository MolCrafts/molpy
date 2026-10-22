"""The AmberTools programs the pipeline runs: antechamber, parmchk2, prepgen, tleap.

Each is a thin shell around one executable: it builds the command line and
nothing else; what to run and in which order is :mod:`molpy.tool.ambertools.steps`'.
A program's executable, environment, environment variables and time limit are
its ``ambertools.<program>`` settings in molpy's configuration
(:mod:`molpy.config`), resolved by the shared :class:`~molpy.tool.tool.Tool`
core; each run is logged to ``molpy.ambertools.<program>`` (:mod:`mollog`).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar, Literal

from molpy.tool.process import ToolError, tail_of
from molpy.tool.tool import Tool


class AmberProgram(Tool):
    """One AmberTools program, run in :attr:`workdir`.

    A subclass names its configuration table in :attr:`tool` (``"tleap"``
    reads ``[ambertools.tleap]``, falling back to ``[ambertools]``). Settings,
    environment, executable, logger and the failure contract come from
    :class:`~molpy.tool.tool.Tool`.

    Args:
        workdir: Where the tool runs; ``None`` is the caller's directory.
        config: molpy's configuration (:func:`molpy.config.load_config`);
            ``None`` loads it (package defaults, user and project files).

    Raises:
        ValueError: The configured environment is incomplete or its manager
            unsupported, or no executable is configured.
    """

    group: ClassVar[str] = "ambertools"

    def run(
        self,
        args: Sequence[str] = (),
        *,
        input_text: str | None = None,
        capture_output: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Execute the program in :attr:`workdir`.

        Args:
            args: Command-line arguments (without the executable).
            input_text: Text to send to stdin.
            capture_output: Whether to capture stdout/stderr.
            check: Raise :class:`~molpy.tool.ToolError` on a non-zero exit.

        Returns:
            The completed process result.

        Raises:
            ToolError: The tool could not start, timed out, or (with *check*)
                exited non-zero.
        """
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)
        return self.run_command(
            self.command(args),
            cwd=self.workdir,
            capture_output=capture_output,
            check=check,
            input_text=input_text,
        )


def run_step(
    tool: Tool,
    output: Path,
    call: Callable[[], subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    """Run one tool step and require the file it must write.

    Args:
        tool: The tool ``call`` runs; checked for its executable first.
        output: The file the step must leave behind.
        call: Runs the step (typically a bound program method in a lambda).

    Returns:
        The completed process.

    Raises:
        ToolError: The executable is missing, the step fails, or it exits 0
            but leaves ``output`` unwritten. The error carries the tool's
            stderr, or its stdout when stderr is empty (tleap and prepgen
            report there).
    """
    tool.check_executable()
    result = call()
    if not output.is_file():
        tool.logger.error("step output missing", output=str(output))
        raise ToolError(
            tool.key,
            f"{tool.executable} left {output.name} unwritten",
            returncode=result.returncode,
            tail=tail_of(result.stderr, result.stdout),
        )
    return result


class Antechamber(AmberProgram):
    """antechamber: atom types and charges (``[ambertools.antechamber]``)."""

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
            check: Raise :class:`~molpy.tool.ToolError` on a non-zero exit.

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


class Parmchk2(AmberProgram):
    """parmchk2: the missing force-field parameters (``[ambertools.parmchk2]``)."""

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
            check: Raise :class:`~molpy.tool.ToolError` on a non-zero exit.

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


class Prepgen(AmberProgram):
    """prepgen: a residue template with connection points (``[ambertools.prepgen]``).

    Prepgen generates AMBER residue templates (.prepi files) with connection
    points defined by control files specifying HEAD_NAME, TAIL_NAME, etc.

    Example:
        >>> prepgen = Prepgen("./work")
        >>> prepgen.generate_residue(
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
            check: Raise :class:`~molpy.tool.ToolError` on a non-zero exit.

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


class Tleap(AmberProgram):
    """tleap, run on a script file it writes (``[ambertools.tleap]``)."""

    tool = "tleap"

    def run_from_script(
        self,
        script_text: str,
        *,
        script_name: str = "tleap.in",
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Execute tleap from a script text.

        Args:
            script_text: The tleap script content.
            script_name: Name of the script file to create (in workdir).
            check: Raise :class:`~molpy.tool.ToolError` on a non-zero exit.

        Returns:
            The completed process result.

        Raises:
            ValueError: If no workdir is set.
            ToolError: tleap failed (with *check*).
        """
        if self.workdir is None:
            raise ValueError("Tleap requires a working directory. Set workdir.")

        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / script_name).write_text(script_text, encoding="utf-8")
        return self.run(["-f", script_name], check=check)
