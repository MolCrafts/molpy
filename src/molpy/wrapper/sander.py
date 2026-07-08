"""Wrapper for the AMBER ``sander`` binary — energy minimization / MD.

Runs ``sander`` on a generated ``mdin`` control file against a prmtop/inpcrd and
returns the restart (``.rst``) coordinate file it writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import Wrapper


@dataclass
class SanderWrapper(Wrapper):
    exe: str = "sander"

    def minimize(
        self,
        prmtop: Path,
        inpcrd: Path,
        *,
        max_iter: int = 500,
        name: str = "min",
        check: bool = False,
    ) -> Path:
        """Run a non-periodic energy minimization; return the restart ``.rst``.

        ``ncyc`` steepest-descent steps then conjugate gradient to ``maxcyc``
        (``imin=1``); ``ntb=0``/``igb=0`` with no cutoff (a gas-phase fragment or
        single molecule). The ``.rst`` shares the inpcrd coordinate format, so
        :func:`molpy.io.readers.read_amber_inpcrd` reads the relaxed coordinates.
        """
        if self.workdir is None:
            raise ValueError("SanderWrapper requires a working directory. Set workdir.")
        self.workdir.mkdir(parents=True, exist_ok=True)

        mdin = self.workdir / f"{name}.in"
        rst = self.workdir / f"{name}.rst"
        mdout = self.workdir / f"{name}.out"
        ncyc = max(1, max_iter // 2)
        # ntxo=1 forces an ASCII restart (sander defaults to NetCDF), so the
        # relaxed coordinates read back through the plain inpcrd reader.
        mdin.write_text(
            f"minimize\n&cntrl\n  imin=1, maxcyc={max_iter}, ncyc={ncyc},\n"
            f"  ntb=0, igb=0, cut=999.0, ntpr={ncyc}, ntxo=1,\n/\n"
        )

        result = self.run(
            args=[
                "-O",
                "-i",
                mdin.name,
                "-p",
                str(Path(prmtop).absolute()),
                "-c",
                str(Path(inpcrd).absolute()),
                "-r",
                rst.name,
                "-o",
                mdout.name,
            ],
            check=check,
        )
        if result.returncode != 0 or not rst.exists():
            raise RuntimeError(
                f"sander minimize failed:\n{result.stderr or result.stdout}"
            )
        return rst
