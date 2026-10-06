"""AmberTools polymer builder.

Public API:
    - AmberCut: one prepgen residue, as the user specifies it
    - AmberPieces: head, repeat and tail SMILES → the oligomer and its cuts
    - AmberPolymerBuilder: oligomer plus cuts in, tleap ``sequence`` out
    - AmberBuildResult: the chain frame, its force field, and the file paths
"""

from .amber_builder import AmberPolymerBuilder
from .oligomer import AmberPieces
from .types import AmberBuildResult, AmberCut

__all__ = [
    "AmberBuildResult",
    "AmberCut",
    "AmberPieces",
    "AmberPolymerBuilder",
]
