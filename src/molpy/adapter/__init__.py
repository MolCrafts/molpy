"""RDKit adapter for MolPy.

Provides bidirectional conversion between RDKit molecules and MolPy structures,
as well as utilities for 3D generation, visualization, and optimization.
"""

from .rdkit_adapter import (
    RDKitWrapper,
    atomistic_to_mol,
    mol_to_atomistic,
    monomer_to_mol,
    smilesir_to_atomistic,
    smilesir_to_mol,
)

__all__ = [
    "RDKitWrapper",
    "atomistic_to_mol",
    "mol_to_atomistic",
    "monomer_to_mol",
    "smilesir_to_atomistic",
    "smilesir_to_mol",
]
