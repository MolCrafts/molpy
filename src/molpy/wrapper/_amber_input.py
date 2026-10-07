"""What an AmberTools program is handed for one molecule (private).

Shared by the typifiers of :mod:`molpy.ff.typifier` (``AntechamberTypifier``,
``TleapTypifier``) and :class:`molpy.builder.AmberPolymerBuilder`, which run
antechamber on a graph: its net charge (``-nc``) and its input mol2. The mol2
itself is written by ``molrs.io.write_mol2``; this module only says which
frame goes in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molrs.io import write_mol2
from molrs.core.keys import FORMAL_CHARGE

if TYPE_CHECKING:
    from pathlib import Path

    from molrs.core import Atomistic


def net_formal_charge(graph: Atomistic) -> int:
    """The sum of the atoms' formal charges (an atom without one is neutral).

    The SMILES reader writes ``formal_charge`` only on charged bracket atoms,
    and the frame schema declares it an integer, so the sum is one.
    """
    key = FORMAL_CHARGE.key
    return sum(int(atom.get(key) or 0) for atom in graph.atoms)


def antechamber_input_mol2(graph: Atomistic, path: Path, *, rename: bool) -> None:
    """Save ``graph`` at ``path`` as the mol2 an AmberTools program reads.

    mol2 carries the bonds, so antechamber does not perceive them from
    coordinates (AmberTools 26's bondtype crashes on a PDB with CONECT
    records). A ``type`` bond column is a force-field label, not the SYBYL
    bond order mol2 reads there, so it is dropped. With ``rename``, atoms are
    named element + row (``C1``, ``O2``, ...) so antechamber and tleap can
    tell elements apart; otherwise the graph's own names are kept.
    """
    frame = graph.to_frame()
    if rename:
        frame["atoms"]["name"] = [
            f"{symbol}{row}"
            for row, symbol in enumerate(frame["atoms"]["element"], start=1)
        ]
    if "bonds" in frame and "type" in frame["bonds"]:
        del frame["bonds"]["type"]
    write_mol2(path, frame)
