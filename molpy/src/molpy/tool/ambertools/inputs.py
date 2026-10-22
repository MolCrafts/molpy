"""The files an AmberTools program reads: the input mol2, the leap script, the prepgen control file.

Every leap script molpy runs is rendered by :func:`leap_script`, which
quotes each path it names, so a work directory with a space in it reaches
tleap as one word.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Literal

from molrs.core.keys import FORMAL_CHARGE
from molrs.io import write_mol2
from molrs.core import Element

if TYPE_CHECKING:
    from molrs.core import Atomistic

#: The GAFF generation: antechamber's ``-at``, parmchk2's ``-s`` and the
#: ``leaprc.<parameter_set>`` tleap sources (molrs's ``parameter_set``).
ParameterSet = Literal["gaff", "gaff2"]

#: The tleap unit every script builds and saves.
UNIT = "MOL"


def net_formal_charge(graph: Atomistic) -> int:
    """The sum of the atoms' formal charges (an atom without one is neutral).

    The SMILES reader writes ``formal_charge`` only on charged bracket atoms,
    and the frame schema declares it an integer, so the sum is one.
    """
    key = FORMAL_CHARGE.key
    return sum(int(atom.get(key) or 0) for atom in graph.atoms)


def input_mol2(graph: Atomistic, *, rename: bool) -> str:
    """The text of the mol2 an AmberTools program reads for ``graph``.

    The caller writes it, and keys its work directory by it: the text holds
    everything of the graph a run depends on (elements, names, coordinates,
    bonds, and the types and charges of a typed graph).

    mol2 carries the bonds, so antechamber does not perceive them from
    coordinates (AmberTools 26's bondtype crashes on a PDB with CONECT
    records). The bond orders are the canonical ``bond_type`` /
    ``bond_number`` columns (every graph bond carries them), which
    ``write_mol2_str`` writes as SYBYL codes (``1``, ``2``, ``ar``, ...).
    ``write_mol2_str`` prefers a ``type`` bond column over them, and on a
    typed graph that column is a force-field label (``c3-os``), so ``type``
    alone is dropped. With ``rename``, atoms are named element + row (``C1``,
    ``O2``, ...) so antechamber and tleap can tell elements apart; otherwise
    the graph's own names are kept.
    """
    frame = graph.to_frame()
    Element.fill_symbols(frame)
    if "type" not in frame["atoms"]:
        frame["atoms"]["type"] = frame["atoms"]["element"]
    if rename:
        frame["atoms"]["name"] = [
            f"{symbol}{row}"
            for row, symbol in enumerate(frame["atoms"]["element"], start=1)
        ]
    if "bonds" in frame and "type" in frame["bonds"]:
        del frame["bonds"]["type"]
    buffer = bytearray()
    write_mol2(buffer, frame)
    return buffer.decode("utf-8")


def leap_path(path: PurePath) -> str:
    """``path`` as one tleap word: forward slashes and double quotes.

    Raises:
        ValueError: The path holds a double quote, which tleap cannot escape.
    """
    text = path.as_posix()
    if '"' in text:
        raise ValueError(f"tleap cannot read a path with a double quote: {text!r}")
    return f'"{text}"'


def leap_mol2(path: PurePath) -> str:
    """The unit expression that loads a mol2."""
    return f"loadmol2 {leap_path(path)}"


def leap_sequence(residues: Sequence[str]) -> str:
    """The unit expression that joins prepgen residues into a chain."""
    return f"sequence {{ {' '.join(residues)} }}"


def leap_script(
    parameter_set: ParameterSet,
    unit: str,
    prmtop: Path,
    inpcrd: Path,
    *,
    frcmods: Sequence[Path] = (),
    preps: Sequence[Path] = (),
) -> str:
    """The tleap script that builds :data:`UNIT` and saves its prmtop.

    The leaprc, then every frcmod and prep file, then the unit
    (:func:`leap_mol2` or :func:`leap_sequence`), then ``saveamberparm``.

    Args:
        parameter_set: The GAFF generation; tleap sources ``leaprc.<set>``.
        unit: The expression :data:`UNIT` is assigned.
        prmtop: Where tleap saves the topology.
        inpcrd: Where tleap saves the coordinates.
        frcmods: Parameter files loaded on top of the leaprc.
        preps: prepgen residue files.
    """
    lines = [f"source leaprc.{parameter_set}"]
    lines += [f"loadamberparams {leap_path(path)}" for path in frcmods]
    lines += [f"loadamberprep {leap_path(path)}" for path in preps]
    lines += [
        f"{UNIT} = {unit}",
        f"saveamberparm {UNIT} {leap_path(prmtop)} {leap_path(inpcrd)}",
        "quit",
    ]
    return "\n".join(lines) + "\n"


def prepgen_control_text(
    *,
    variant: Literal["chain", "head", "tail"],
    head_name: str | None = None,
    tail_name: str | None = None,
    head_type: str | None = None,
    tail_type: str | None = None,
    omit_names: Sequence[str] = (),
    charge: int = 0,
) -> str:
    """The text of a prepgen control file (``-m``) for one residue variant.

    Args:
        variant: ``"chain"`` has a HEAD and a TAIL connection, ``"head"``
            (a chain start) only a TAIL, ``"tail"`` (a chain end) only a HEAD.
        head_name: Atom name for HEAD connection (required for chain/tail).
        tail_name: Atom name for TAIL connection (required for chain/head).
        head_type: Atom type for PRE_HEAD_TYPE (e.g., "c3").
        tail_type: Atom type for POST_TAIL_TYPE (e.g., "os").
        omit_names: Atom names to omit (leaving groups).
        charge: Net charge of the residue.

    Raises:
        ValueError: A connection atom the variant needs is missing.
    """
    lines: list[str] = []
    if variant == "chain":
        if head_name is None or tail_name is None:
            raise ValueError("chain variant requires both head_name and tail_name")
        lines += [f"HEAD_NAME {head_name}", f"TAIL_NAME {tail_name}"]
        if head_type:
            lines.append(f"PRE_HEAD_TYPE {head_type}")
        if tail_type:
            lines.append(f"POST_TAIL_TYPE {tail_type}")
    elif variant == "head":
        # Only the TAIL connects. When the tail atom is a graph terminus,
        # prepgen would root the residue tree there and attach the next
        # residue at the wrong atom; a HEAD_NAME (the tail atom's neighbour)
        # pins the main chain. The head connection is unused at a chain start.
        if tail_name is None:
            raise ValueError("head variant requires tail_name")
        if head_name:
            lines.append(f"HEAD_NAME {head_name}")
        lines.append(f"TAIL_NAME {tail_name}")
        if tail_type:
            lines.append(f"POST_TAIL_TYPE {tail_type}")
    elif variant == "tail":
        # Only the HEAD connects; a TAIL_NAME pins the main chain, as above.
        if head_name is None:
            raise ValueError("tail variant requires head_name")
        lines.append(f"HEAD_NAME {head_name}")
        if tail_name:
            lines.append(f"TAIL_NAME {tail_name}")
        if head_type:
            lines.append(f"PRE_HEAD_TYPE {head_type}")
    lines += [f"OMIT_NAME {name}" for name in omit_names]
    lines.append(f"CHARGE {charge}")
    return "\n".join(lines) + "\n"
