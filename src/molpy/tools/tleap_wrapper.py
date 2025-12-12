"""Wrapper for the 'tleap' binary.

This module provides a thin wrapper around the tleap script-driven binary.
It knows how to run tleap on a given script file, but does NOT build the script itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .wrapper import Wrapper


@dataclass
class TLeapWrapper(Wrapper):
    """Wrapper for the 'tleap' binary.

    This wrapper knows how to run tleap on a given script file.
    It does NOT build the script itself; that is the responsibility of Compute nodes.

    **What this wrapper does:**
    - Knows the executable name (exe = "tleap")
    - Writes script text to a file
    - Runs 'tleap -f <script_name>'
    - Manages working directory

    **What this wrapper does NOT do:**
    - Decide which source leaprc.* lines to use
    - Choose units to load
    - Build saveamberparm commands
    - Pick output filenames
    - Interpret tleap output

    All of the above belong in Compute nodes.

    Example:
        >>> from molpy.tools import TLeapWrapper
        >>> from pathlib import Path
        >>>
        >>> # Create wrapper
        >>> tleap = TLeapWrapper(
        ...     name="tleap",
        ...     exe="tleap",
        ...     workdir=Path("tmp_tleap")
        ... )
        >>>
        >>> # Compute node builds script
        >>> script = \"\"\"
        ... source leaprc.protein.ff14SB
        ... loadamberparams lig.frcmod
        ... lig = loadmol2 lig.mol2
        ... saveamberparm lig lig.prmtop lig.inpcrd
        ... quit
        ... \"\"\"
        >>>
        >>> # Wrapper runs script
        >>> proc = tleap.run_script(script_text=script, script_name="build.in")
        >>> if proc.returncode != 0:
        ...     # Compute node handles error
        ...     raise RuntimeError(f"TLeap failed: {proc.stderr}")
    """

    exe: str = "tleap"

    def run_script(
        self,
        script_text: str,
        *,
        script_name: str = "tleap.in",
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Write a tleap script and run 'tleap -f <script_name>'.

        The caller (a Compute node) is responsible for constructing script_text.
        This method only handles file I/O and process invocation.

        Args:
            script_text: The tleap script content as a string.
            script_name: Name of the script file to write (default: "tleap.in").
            cwd: Working directory (overrides self.workdir if provided).

        Returns:
            CompletedProcess object with execution results.

        Raises:
            ValueError: If no working directory is available (neither cwd nor workdir).

        Example:
            >>> tleap = TLeapWrapper(name="tleap", workdir=Path("work"))
            >>> script = "source leaprc.gaff\\nquit\\n"
            >>> proc = tleap.run_script(script_text=script)
        """
        real_cwd = cwd or self.workdir
        if real_cwd is None:
            raise ValueError(
                "TLeapWrapper requires a working directory. "
                "Set workdir in constructor or provide cwd parameter."
            )

        real_cwd.mkdir(parents=True, exist_ok=True)
        script_path = real_cwd / script_name

        # Write script to file
        script_path.write_text(script_text)

        # tleap expects: tleap -f script_name
        return self.run(args=["-f", script_name], cwd=real_cwd)

