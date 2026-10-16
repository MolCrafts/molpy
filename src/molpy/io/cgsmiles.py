"""CGsmiles notation — :mod:`molrs.io.cgsmiles`, mirrored by identity.

``CgSmilesIr`` and the records it hands out (``CgGraph``, ``CgNode``,
``CgEdge``, ``CgFragmentDef``, ``ResolvedPair``, ``PairEnd``). One molecule
straight from a CGsmiles string is ``CgSmilesIr.parse(text).to_atomistic()``.
"""

from molrs.io.cgsmiles import *  # noqa: F403
from molrs.io.cgsmiles import __all__ as __all__
