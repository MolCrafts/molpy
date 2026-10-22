"""The AmberTools steps, and the one policy for the outputs they leave behind.

Every step deletes the files it writes before it runs, then requires them
afterwards, so a program that exits 0 without writing raises instead of
leaving an earlier run's file to be read. Whether a step runs at all is the
caller's choice (its caching): the typifiers run every step, and
:class:`molpy.builder.AmberPolymerBuilder` skips a step whose inputs have
not changed.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from molpy.tool.ambertools.inputs import ParameterSet
from molpy.tool.ambertools.programs import (
    Antechamber,
    Parmchk2,
    Prepgen,
    Tleap,
    run_step,
)
from molpy.tool.process import ToolError
from molpy.tool.tool import Tool

if TYPE_CHECKING:
    from molcfg import Config

#: The antechamber ``-fo`` / parmchk2 ``-f`` format of each file suffix.
_FORMATS = {".ac": "ac", ".mol2": "mol2"}


def digest(*parts: str | bytes) -> str:
    """The SHA-1 hex digest of ``parts``, each length-prefixed."""
    sha = hashlib.sha1()
    for part in parts:
        data = part.encode() if isinstance(part, str) else part
        sha.update(len(data).to_bytes(8, "little"))
        sha.update(data)
    return sha.hexdigest()


def run_directory(root: Path, *parts: str | bytes) -> Path:
    """``root/<digest of parts>``, created.

    ``parts`` are everything a run's result depends on (the input's text and
    every setting handed to a program), so a run with other settings never
    lands in the directory of an earlier one.
    """
    directory = root / digest(*parts)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _format(path: Path) -> str:
    try:
        return _FORMATS[path.suffix]
    except KeyError:
        raise ValueError(f"{path.name}: an AmberTools file is .ac or .mol2") from None


def _fresh(
    tool: Tool,
    outputs: tuple[Path, ...],
    call: Callable[[], subprocess.CompletedProcess[str]],
) -> None:
    """Delete ``outputs``, run ``call``, and require every output."""
    for path in outputs:
        path.unlink(missing_ok=True)
    result = run_step(tool, outputs[0], call)
    for path in outputs[1:]:
        if not path.is_file():
            raise ToolError(
                tool.key,
                f"{tool.executable} left {path.name} unwritten",
                returncode=result.returncode,
            )


@dataclass(frozen=True)
class AmberSteps:
    """antechamber (types and charges, or AM1 charges alone), parmchk2, prepgen and tleap, run in one directory.

    Attributes:
        directory: Where every program runs.
        config: molpy's configuration: the executables and their
            environment (``[ambertools]``, ``[ambertools.<tool>]``).
    """

    directory: Path
    config: Config

    def antechamber(
        self,
        source: Path,
        output: Path,
        *,
        parameter_set: ParameterSet,
        charge_method: str,
        net_charge: int,
    ) -> None:
        """Types and charges: ``source`` (mol2) → ``output`` (.ac or .mol2)."""
        output_format = _format(output)
        tool = Antechamber(self.directory, config=self.config)
        _fresh(
            tool,
            (output,),
            lambda: tool.atomtype_assign(
                source,
                output,
                input_format="mol2",
                output_format=output_format,  # type: ignore[arg-type]
                charge_method=charge_method,  # type: ignore[arg-type]
                parameter_set=parameter_set,
                net_charge=net_charge,
            ),
        )

    def am1(self, source: Path, ac: Path, *, net_charge: int) -> None:
        """sqm's AM1 Mulliken charges: ``source`` (mol2) → ``ac``.

        antechamber ``-c mul`` writes the very sqm input ``-c bcc`` writes, and
        hands its Mulliken charges back unequivalenced (``-eq 0``): the QM step
        of AM1-BCC and ABCG2 and nothing after it. ``-at bcc`` only labels the
        ac; its types are not read.
        """
        tool = Antechamber(self.directory, config=self.config)
        _fresh(
            tool,
            (ac,),
            lambda: tool.atomtype_assign(
                source,
                ac,
                input_format="mol2",
                output_format="ac",
                charge_method="mul",
                parameter_set="bcc",
                net_charge=net_charge,
            ),
        )

    def parmchk2(
        self, typed: Path, frcmod: Path, *, parameter_set: ParameterSet
    ) -> None:
        """The parameters ``parameter_set`` lacks for ``typed`` (.ac or .mol2)."""
        input_format = _format(typed)
        tool = Parmchk2(self.directory, config=self.config)
        _fresh(
            tool,
            (frcmod,),
            lambda: tool.generate_parameters(
                typed,
                frcmod,
                input_format=input_format,  # type: ignore[arg-type]
                parameter_set=parameter_set,
            ),
        )

    def prepgen(self, ac: Path, prepi: Path, control: Path, residue: str) -> None:
        """One residue cut from ``ac`` by ``control``.

        prepgen reads only an ac file. The files are named relative to
        :attr:`directory`, where all of them are.
        """
        tool = Prepgen(self.directory, config=self.config)
        _fresh(
            tool,
            (prepi,),
            lambda: tool.generate_residue(
                input_file=ac.name,
                output_file=prepi.name,
                control_file=control.name,
                residue_name=residue,
            ),
        )

    def tleap(
        self,
        script: str,
        prmtop: Path,
        inpcrd: Path,
        *,
        script_name: str = "tleap.in",
    ) -> None:
        """Run a :func:`~molpy.tool.ambertools.inputs.leap_script` that saves ``prmtop``."""
        tool = Tleap(self.directory, config=self.config)
        _fresh(
            tool,
            (prmtop, inpcrd),
            lambda: tool.run_from_script(script, script_name=script_name),
        )
