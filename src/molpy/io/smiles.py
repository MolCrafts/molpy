"""SMILES notation — :mod:`molrs.io.smiles`, mirrored by identity.

``SmilesIr`` (the parsed text), ``SmilesError`` (every refusal of the
SMILES family) and ``BondingDescriptor``. Parse a molecule with
``read_smiles(text).to_atomistic()`` and write it with
``write_smiles(mol)``.
SMARTS is :mod:`molpy.perceive`'s.
"""

from molrs.io.smiles import *  # noqa: F403
from molrs.io.smiles import __all__ as _native

__all__ = [*_native]
