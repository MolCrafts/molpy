"""Wrapper for the 'prepgen' CLI.

Higher-level workflow decisions belong in compute nodes.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from .base import Wrapper


@dataclass
class PrepgenWrapper(Wrapper):
    """Wrapper for the 'prepgen' CLI."""

    exe: str = "prepgen"

    def run_raw(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return self.run(args=args, cwd=cwd, input_text=input_text)
