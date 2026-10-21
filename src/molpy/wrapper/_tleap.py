"""Wrapper for the 'tleap' binary.

This wrapper runs ``tleap`` on a generated script file.
"""

from __future__ import annotations

import subprocess

from molpy.wrapper._wrapper import Wrapper


class TleapWrapper(Wrapper):
    """Wrapper for the 'tleap' binary (``[wrapper.tleap]``)."""

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
            check: Raise :class:`~molpy.wrapper.ToolError` on a non-zero exit.

        Returns:
            The completed process result.

        Raises:
            ValueError: If no workdir is set.
            ToolError: tleap failed (with *check*).
        """
        if self.workdir is None:
            raise ValueError("TleapWrapper requires a working directory. Set workdir.")

        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / script_name).write_text(script_text, encoding="utf-8")
        return self.run(["-f", script_name], check=check)
