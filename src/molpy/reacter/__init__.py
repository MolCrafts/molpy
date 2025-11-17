"""
Programmable Reacter Module for Chemical Transformations.

This module provides a framework for defining and executing chemical reactions
within the molpy framework, following SMIRKS-style semantics but working
entirely on native data structures (Atom, Bond, Struct, Monomer).

Core Concepts:
--------------
- **Reacter**: Represents a single chemical reaction type
- **ProductSet**: Container for reaction products and metadata
- **Selectors**: Functions that identify anchor atoms and leaving groups
- **Transformers**: Functions that create or modify bonds

Example Usage:
--------------
```python
from molpy.reacter import Reacter, port_anchor_selector, remove_one_H, make_single_bond

# Define a C-C coupling reaction
cc_coupling = Reacter(
    name="C-C_coupling_with_H_loss",
    anchor_left=port_anchor_selector,
    anchor_right=port_anchor_selector,
    leaving_left=remove_one_H,
    leaving_right=remove_one_H,
    bond_maker=make_single_bond,
)

# Execute reaction between two monomers
product = cc_coupling.run(monoA, monoB, port_L="1", port_R="2")
print(f"Removed atoms: {product.notes['removed_atoms']}")
print(f"New bonds: {product.notes['new_bonds']}")
```

Design Goals:
-------------
- Pure Python, framework-native (no RDKit)
- Composable: reaction logic = modular functions
- Stable indexing: atom deletion doesn't shift IDs
- Single responsibility: one Reacter = one reaction type
- Extensible: easy to subclass for specialized reactions
- Auditable: all changes recorded in ProductSet.notes
"""

from .base import AtomId, ProductSet, Reacter
from .connector import ReacterConnector
from .selectors import (
    no_leaving_group,
    port_anchor_selector,
    remove_all_H,
    remove_dummy_atoms,
    remove_OH,
    remove_one_H,
    remove_water,
)
from .transformers import (
    break_bond,
    make_aromatic_bond,
    make_bond_by_order,
    make_double_bond,
    make_single_bond,
    make_triple_bond,
    no_new_bond,
)
from .utils import create_atom_mapping, find_neighbors

__all__ = [
    "AtomId",
    "ProductSet",
    # Core classes
    "Reacter",
    # Connectors
    "ReacterConnector",
    "break_bond",
    "create_atom_mapping",
    # Utilities
    "find_neighbors",
    "make_aromatic_bond",
    "make_bond_by_order",
    "make_double_bond",
    # Transformers
    "make_single_bond",
    "make_triple_bond",
    "no_leaving_group",
    "no_new_bond",
    # Selectors
    "port_anchor_selector",
    "remove_OH",
    "remove_all_H",
    "remove_dummy_atoms",
    "remove_one_H",
    "remove_water",
]
