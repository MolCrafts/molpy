"""One prepgen cut, and the chain :class:`AmberPolymerBuilder` returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from molrs import Atomistic
from molrs.ff import ForceField


@dataclass(frozen=True)
class AmberCut:
    """One prepgen residue, written out as a control file.

    The names are atoms of the oligomer the user built. ``omit`` is dropped.
    ``head`` / ``tail`` are ``HEAD_NAME`` / ``TAIL_NAME``. ``pre_head`` and
    ``post_tail`` are the atoms across those bonds; their GAFF types, read
    from the ac file, become ``PRE_HEAD_TYPE`` / ``POST_TAIL_TYPE``. Pass
    ``pre_head_type`` / ``post_tail_type`` to write a type yourself.

    A head residue has a tail connection. A tail residue has a head
    connection. A chain residue has both. A head residue may still name a
    ``head`` (and a tail residue a ``tail``): prepgen then roots its main
    chain there, which pins the connection atom when it is a terminus.

    Attributes:
        omit: Atoms prepgen drops; their charge is spread over the rest.
        head: ``HEAD_NAME``, the atom bonded to the previous residue.
        tail: ``TAIL_NAME``, the atom bonded to the next residue.
        pre_head: The oligomer atom across the head bond, whose type
            becomes ``PRE_HEAD_TYPE``.
        post_tail: The oligomer atom across the tail bond, whose type
            becomes ``POST_TAIL_TYPE``.
        pre_head_type: ``PRE_HEAD_TYPE`` written as given.
        post_tail_type: ``POST_TAIL_TYPE`` written as given.
        charge: The residue's net charge (``CHARGE``).
    """

    omit: tuple[str, ...] = ()
    head: str | None = None
    tail: str | None = None
    pre_head: str | None = None
    post_tail: str | None = None
    pre_head_type: str | None = None
    post_tail_type: str | None = None
    charge: int = 0


@dataclass
class AmberBuildResult:
    """Chain built by :meth:`AmberPolymerBuilder.assemble`.

    Coordinates, types, charges and parameters are what tleap wrote. They
    are assigned the way :class:`~molpy.typifier.AntechamberTypifier`
    assigns its prmtop, so ``chain`` is a typed graph like a typifier's
    output and ``forcefield`` merges with a typifier's force field.

    Attributes:
        chain: The typed polymer (coordinates, types, charges, bonded terms).
        forcefield: The parameters of the types ``chain`` uses, with AMBER
            units and 1-4 scaling declared.
        prmtop_path: The AMBER topology file.
        inpcrd_path: The AMBER coordinate file.
        monomer_count: Number of sites in the chain.
    """

    chain: Atomistic
    forcefield: ForceField
    prmtop_path: Path
    inpcrd_path: Path
    monomer_count: int
