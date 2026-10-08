"""Wrapper for the 'tleap' binary.

This wrapper runs ``tleap`` on a generated script file.
"""

from __future__ import annotations

import subprocess


from ._wrapper import Wrapper


class TleapWrapper(Wrapper):
    tool = "tleap"

    def run_from_script(
        self,
        script_text: str,
        *,
        script_name: str = "tleap.in",
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Execute tleap from a script text.

        Args:
            script_text: The tleap script content.
            script_name: Name of the script file to create (in workdir).
            check: If True, raise ``CalledProcessError`` on a non-zero exit.

        Returns:
            The completed process result.

        Raises:
            ValueError: If no workdir is set.
        """
        if self.workdir is None:
            raise ValueError("TleapWrapper requires a working directory. Set workdir.")

        self.workdir.mkdir(parents=True, exist_ok=True)
        script_path = self.workdir / script_name

        script_path.write_text(script_text, encoding="utf-8")

        return self.run(args=["-f", script_name], check=check)
