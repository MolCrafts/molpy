"""Wrapper for the 'antechamber' CLI and related Amber tools.

This module provides a thin wrapper around the antechamber command-line interface.
It does not encode chemistry logic, only how to invoke the binary.

Higher-level choices (file names, flags, workflows) are handled by Compute nodes.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from molpy.external.wrapper import Wrapper


@dataclass
class AntechamberWrapper(Wrapper):
    """Wrapper for the 'antechamber' CLI.

    This class only helps build and run command-line calls.
    Higher-level choices (file names, flags, workflows) are handled by Compute nodes.

    **What this wrapper does:**
    - Knows the executable name (exe = "antechamber")
    - Provides run_raw() convenience method for clarity
    - Manages working directory and environment

    **What this wrapper does NOT do:**
    - Decide which input/output file formats to use
    - Build command-line arguments from domain logic
    - Invoke parmchk2 or other related tools
    - Interpret antechamber output or errors

    All of the above belong in Compute nodes.

    Example:
        >>> from molpy.external import AntechamberWrapper
        >>> from pathlib import Path
        >>>
        >>> # Create wrapper
        >>> ante = AntechamberWrapper(
        ...     name="antechamber",
        ...     exe="antechamber",
        ...     workdir=Path("tmp_ante")
        ... )
        >>>
        >>> # Compute node builds args and calls wrapper
        >>> args = [
        ...     "-i", "lig.mol2",
        ...     "-fi", "mol2",
        ...     "-o", "lig.gaff.mol2",
        ...     "-fo", "mol2",
        ...     "-c", "bcc",
        ...     "-at", "gaff2"
        ... ]
        >>> proc = ante.run_raw(args=args)
        >>> if proc.returncode != 0:
        ...     # Compute node handles error
        ...     raise RuntimeError(f"Antechamber failed: {proc.stderr}")
    """

    exe: str = "antechamber"

    def run_raw(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Convenience alias around Wrapper.run for Antechamber-specific calls.

        This method provides a clearer API for antechamber invocations.
        The actual work is delegated to the base run() method.

        Args:
            args: Command-line arguments to pass to antechamber.
            cwd: Working directory (overrides self.workdir if provided).
            input_text: Text to send to stdin (rarely used for antechamber).

        Returns:
            CompletedProcess object with execution results.

        Example:
            >>> ante = AntechamberWrapper(name="ante", workdir=Path("work"))
            >>> proc = ante.run_raw(["-i", "lig.mol2", "-fi", "mol2", "-o", "out.mol2", "-fo", "mol2"])
        """
        return self.run(args=args, cwd=cwd, input_text=input_text)
