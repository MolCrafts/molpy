"""
LAMMPS molecular dynamics engine.

This module provides the LAMMPSEngine class for running LAMMPS calculations.
"""

import subprocess
from pathlib import Path
from typing import Any

from .base import Engine


class LAMMPSEngine(Engine):
    """
    LAMMPS molecular dynamics engine.

    This engine runs LAMMPS calculations with input scripts.

    Example:
        >>> from molpy.core.script import Script
        >>> from molpy.engine import LAMMPSEngine
        >>>
        >>> # Create input script
        >>> script = Script.from_text(
        ...     name="input",
        ...     text="units real\\natom_style full\\n",
        ...     language="other"
        ... )
        >>>
        >>> # Create engine and prepare
        >>> engine = LAMMPSEngine(executable="lmp")
        >>> engine.prepare(work_dir="./calc", scripts=script)
        >>>
        >>> # Run calculation
        >>> result = engine.run()
        >>> print(result.returncode)
        0
    """

    @property
    def name(self) -> str:
        """Return engine name."""
        return "LAMMPS"

    def _get_default_extension(self) -> str:
        """Get default file extension for LAMMPS input files."""
        return ".lmp"

    def run(
        self,
        input_file: str | Path | None = None,
        log_file: str | Path | None = None,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """
        Execute LAMMPS calculation.

        Args:
            input_file: Name of the input script file. If None, uses the input script
                       from prepare() (default: None)
            log_file: Name of the log file. If None, uses default "log.lammps"
                     (default: None)
            **kwargs: Additional arguments passed to subprocess.run
                     (e.g., capture_output, text, check, etc.)

        Returns:
            CompletedProcess object with execution results

        Raises:
            RuntimeError: If engine is not prepared (prepare() not called)
        """
        if not hasattr(self, "work_dir"):
            raise RuntimeError("Engine not prepared. Call prepare() first.")

        # Use input script from prepare() if not specified
        if input_file is None:
            if (
                not hasattr(self, "input_script")
                or self.input_script is None
                or self.input_script.path is None
            ):
                raise RuntimeError(
                    "No input file specified and no input script found. "
                    "Either specify input_file or ensure prepare() was called with a script."
                )
            input_file = self.input_script.path.name
        else:
            input_file = Path(input_file).name

        # Default log file name
        log_file = "log.lammps" if log_file is None else Path(log_file).name

        # Build command
        command = [self.executable, "-in", str(input_file), "-log", str(log_file)]

        # Default subprocess arguments
        run_kwargs = {
            "cwd": self.work_dir,
            "capture_output": True,
            "text": True,
            "check": False,
        }
        run_kwargs.update(kwargs)

        return subprocess.run(command, **run_kwargs)

    def get_log_file(self) -> Path | None:
        """
        Get the LAMMPS log file path.

        Returns:
            Path to the log file or None if not found
        """
        log_path = self.work_dir / "log.lammps"
        if log_path.exists():
            return log_path
        return None
