"""SMILES notation — :mod:`molrs.io.smiles`, mirrored by identity.

``SmilesIr`` (the parsed text), ``SmilesError`` (every refusal of the
SMILES family) and ``BondingDescriptor``. Parse a molecule with
``SmilesIr.parse(text).to_atomistic()`` and write it with
``str(SmilesIr.from_atomistic(mol))``.
SMARTS is :mod:`molpy.perceive`'s.
"""

from molrs.io.smiles import *  # noqa: F403
from molrs.io.smiles import __all__ as __all__
