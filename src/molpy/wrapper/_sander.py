"""Wrapper for the AMBER ``sander`` binary.

Runs ``sander`` on a caller's ``mdin`` control text against a prmtop/inpcrd.
What the control text asks for (minimisation, MD, its cutoffs) is the
caller's decision; this wrapper only writes it and runs the program.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from molpy.wrapper._wrapper import Wrapper


class SanderWrapper(Wrapper):
    """Wrapper for the 'sander' binary (``[wrapper.sander]``)."""

    tool = "sander"

    def run_from_mdin(
        self,
        mdin_text: str,
        prmtop: str | Path,
        inpcrd: str | Path,
        *,
        name: str = "sander",
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Write ``<name>.in`` and run ``sander`` on it.

        Runs ``sander -O -i <name>.in -p <prmtop> -c <inpcrd> -r <name>.rst
        -o <name>.out`` in :attr:`workdir`; the restart ``<name>.rst`` and
        the output ``<name>.out`` land there.

        Args:
            mdin_text: The ``mdin`` control text (a ``&cntrl`` namelist).
            prmtop: The topology.
            inpcrd: The starting coordinates.
            name: Stem of the control, restart and output files.
            check: Raise :class:`~molpy.wrapper.ToolError` on a non-zero exit.

        Returns:
            The completed process result.

        Raises:
            ValueError: If no workdir is set.
            ToolError: sander failed (with *check*).
        """
        if self.workdir is None:
            raise ValueError("SanderWrapper requires a working directory. Set workdir.")

        self.workdir.mkdir(parents=True, exist_ok=True)
        mdin = self.workdir / f"{name}.in"
        mdin.write_text(mdin_text, encoding="utf-8")
        return self.run(
            [
                "-O",
                "-i",
                mdin.name,
                "-p",
                str(Path(prmtop).absolute()),
                "-c",
                str(Path(inpcrd).absolute()),
                "-r",
                f"{name}.rst",
                "-o",
                f"{name}.out",
            ],
            check=check,
        )
