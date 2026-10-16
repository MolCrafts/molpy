"""AmberTools polymer builder.

Public API:
    - AmberCut: one prepgen residue, as the user specifies it
    - AmberPieces: head, repeat and tail SMILES → the oligomer and its cuts
    - AmberPolymerBuilder: oligomer plus cuts in, tleap ``sequence`` out
    - AmberBuildResult: the chain frame, its force field, and the file paths
"""

from molpy.builder._polymer.ambertools._cut import AmberCut
from molpy.builder._polymer.ambertools._result import AmberBuildResult
from molpy.builder._polymer.ambertools.amber_builder import AmberPolymerBuilder
from molpy.builder._polymer.ambertools.oligomer import AmberPieces

__all__ = [
    "AmberBuildResult",
    "AmberCut",
    "AmberPieces",
    "AmberPolymerBuilder",
]
