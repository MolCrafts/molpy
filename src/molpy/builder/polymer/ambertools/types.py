"""
Type definitions for AmberTools polymer builder.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AmberBuildResult:
    """Result of AmberPolymerBuilder.build().

    This result contains the built polymer structure and force field parameters,
    along with paths to intermediate files for debugging.

    Attributes:
        frame: The polymer as a Frame object (coordinates + topology).
        forcefield: The ForceField with all parameters.
        prmtop_path: Path to the generated AMBER topology file.
        inpcrd_path: Path to the generated AMBER coordinate file.
        pdb_path: Path to the generated PDB file (optional).
        monomer_count: Total number of monomers in the polymer.
        cgsmiles: The CGSmiles string used to build the polymer.
    """

    frame: Any  # Frame (avoid circular import)
    forcefield: Any  # ForceField (avoid circular import)
    prmtop_path: Path
    inpcrd_path: Path
    pdb_path: Path | None
    monomer_count: int
    cgsmiles: str | None
