"""Coarse-grained molecular structure: identity re-exports of the native graph.

``molpy.CoarseGrain is molrs.CoarseGrain``; ``Bead`` and ``CGBond`` are the
native ``molrs.views`` handle views. This module stays so internal
``from molpy.core.cg import ...`` paths keep resolving.
"""

from molrs.views import Bead, CGBond, CoarseGrain

__all__ = ["Bead", "CGBond", "CoarseGrain"]
