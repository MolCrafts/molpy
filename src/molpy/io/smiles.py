"""SMILES notation — :mod:`molrs.io.smiles`, mirrored by identity.

``SmilesIr`` (the parsed text), ``SmilesError`` (every refusal of the
SMILES family) and ``BondingDescriptor``. One molecule straight to a graph is
``mp.io.read_smiles_str``; one written back is ``mp.io.write_smiles_str``.
SMARTS is :mod:`molpy.perceive`'s.
"""

from molrs.io.smiles import *  # noqa: F403
from molrs.io.smiles import __all__ as __all__
