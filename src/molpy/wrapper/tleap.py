"""Wrapper for the 'tleap' binary.

This wrapper runs ``tleap`` on a generated script file.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from molpy.core.forcefield import ForceField
from molpy.core.frame import Frame
from molpy.io.readers import read_amber_prmtop
from molpy.io.utils import ensure_parent_dir

from .base import Wrapper


@dataclass
class TLeapWrapper(Wrapper):
    exe: str = "tleap"

    def run_from_script(
        self,
        script_text: str,
        *,
        script_name: str = "tleap.in",
    ) -> subprocess.CompletedProcess[str]:
        """Execute tleap from a script text.

        Args:
            script_text: The tleap script content.
            script_name: Name of the script file to create (in workdir).

        Returns:
            The completed process result.

        Raises:
            ValueError: If no workdir is set.
        """
        if self.workdir is None:
            raise ValueError("TLeapWrapper requires a working directory. Set workdir.")

        self.workdir.mkdir(parents=True, exist_ok=True)
        script_path = self.workdir / script_name

        script_path.write_text(script_text)

        return self.run(args=["-f", script_name])


def read_tleap_outputs(
    prmtop_path: Path, inpcrd_path: Path
) -> tuple[Frame, ForceField]:
    """Read tleap output files (prmtop and inpcrd) into Frame and ForceField.

    Args:
        prmtop_path: Path to the AMBER topology file (.prmtop).
        inpcrd_path: Path to the AMBER coordinate file (.inpcrd).

    Returns:
        Tuple of (Frame, ForceField) objects.
    """
    return read_amber_prmtop(prmtop_path, inpcrd_path)
