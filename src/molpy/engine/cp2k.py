"""
CP2K quantum chemistry engine.

This module provides the CP2KEngine class for running CP2K calculations.
"""

import subprocess
from pathlib import Path
from typing import Any

from .base import Engine


class CP2KEngine(Engine):
    """
    CP2K quantum chemistry engine.

    This engine runs CP2K calculations with input scripts.

    Example:
        >>> from molpy.core.script import Script
        >>> from molpy.engine import CP2KEngine
        >>>
        >>> # Create input script
        >>> script = Script.from_text(
        ...     name="input",
        ...     text="&GLOBAL\\n  PROJECT water\\n&END GLOBAL\\n",
        ...     language="other"
        ... )
        >>>
        >>> # Create engine and prepare
        >>> engine = CP2KEngine(executable="cp2k.psmp")
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
        return "CP2K"

    def _get_default_extension(self) -> str:
        """Get default file extension for CP2K input files."""
        return ".inp"

    def run(
        self,
        input_file: str | Path | None = None,
        output_file: str | Path | None = None,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """
        Execute CP2K calculation.

        Args:
            input_file: Name of the input file. If None, uses the input script
                       from prepare() (default: None)
            output_file: Name of the output file. If None, uses default "cp2k.out"
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

        # Default output file name
        output_file = "cp2k.out" if output_file is None else Path(output_file).name

        # Build command
        command = [self.executable, "-i", str(input_file), "-o", str(output_file)]

        # Default subprocess arguments
        run_kwargs = {
            "cwd": self.work_dir,
            "capture_output": True,
            "text": True,
            "check": False,
        }
        run_kwargs.update(kwargs)

        return subprocess.run(command, **run_kwargs)

    def get_output_file(self, name: str | None = None) -> Path | None:
        """
        Get the CP2K output file path.

        Args:
            name: Name of the output file. If None, uses default "cp2k.out".

        Returns:
            Path to the output file or None if not found
        """
        if name is None:
            name = "cp2k.out"

        output_path = self.work_dir / name
        if output_path.exists():
            return output_path
        return None
