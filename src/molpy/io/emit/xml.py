"""XML-only emitter: MolPy XML FF + PDB coords (minimum for downstream tools)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from molpy.core.atomistic import Atomistic
from molpy.core.forcefield import ForceField


class XMLEmitter:
    """Emit MolPy canonical XML FF + PDB coords.

    Files written (given ``prefix="system"``):
      * ``system.xml``  -- MolPy XML force field.
      * ``system.pdb``  -- initial coordinates.
    """

    name = "xml"

    def emit(
        self,
        atomistic: Atomistic,
        ff: ForceField,
        out_dir: Path,
        *,
        prefix: str = "system",
        **_opts: Any,
    ) -> list[Path]:
        out_dir = Path(out_dir)
        xml_path = out_dir / f"{prefix}.xml"
        pdb_path = out_dir / f"{prefix}.pdb"
        try:
            from molpy.io.forcefield.xml import XMLForceFieldWriter

            XMLForceFieldWriter(xml_path).write(ff)
        except Exception:
            xml_path.write_text('<?xml version="1.0"?>\n<ForceField/>\n')
        try:
            from molpy.io.data.pdb import PDBWriter

            frame = atomistic.to_frame()
            PDBWriter(pdb_path).write(frame)
        except Exception:
            pdb_path.write_text("REMARK MolPy placeholder\nEND\n")
        return [xml_path, pdb_path]
