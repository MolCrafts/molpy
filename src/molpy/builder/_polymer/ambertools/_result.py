"""The chain :class:`AmberPolymerBuilder` returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from molrs.core import Atomistic
from molrs.ff.forcefield import ForceField


@dataclass
class AmberBuildResult:
    """Chain built by :meth:`AmberPolymerBuilder.assemble`.

    Coordinates, types, charges and parameters are what tleap wrote. They
    are assigned the way :class:`~molpy.ff.typifier.AntechamberTypifier`
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
