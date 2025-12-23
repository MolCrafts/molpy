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

    def run_script(
        self,
        script_text: str,
        *,
        script_name: str = "tleap.in",
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        real_cwd = cwd or self.workdir
        if real_cwd is None:
            raise ValueError(
                "TLeapWrapper requires a working directory. Set workdir or provide cwd."
            )

        real_cwd.mkdir(parents=True, exist_ok=True)
        script_path = real_cwd / script_name

        write_tleap_script(script_path, script_text)

        return self.run(args=["-f", script_name], cwd=real_cwd)


def write_tleap_script(path: Path, script_text: str) -> None:
    ensure_parent_dir(path)
    path.write_text(script_text)


def read_tleap_outputs(
    prmtop_path: Path, inpcrd_path: Path
) -> tuple[Frame, ForceField]:
    return read_amber_prmtop(prmtop_path, inpcrd_path)
