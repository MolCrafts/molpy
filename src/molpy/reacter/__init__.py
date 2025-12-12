"""
Programmable Reacter Module for Chemical Transformations.

This module provides a framework for defining and executing chemical reactions
within the molpy framework, following SMIRKS-style semantics but working
entirely on native data structures (Atom, Bond, Atomistic).

Core Concepts:
--------------
- **Reacter**: Represents a single chemical reaction type
- **ReactionResult**: Container for reaction products and metadata
- **Selectors**: Functions that identify port atoms and leaving groups
- **Transformers**: Functions that create or modify bonds

Example Usage:
--------------
```python
from molpy.reacter import Reacter, select_port_atom, select_one_hydrogen, form_single_bond

# Define a C-C coupling reaction
cc_coupling = Reacter(
    name="C-C_coupling_with_H_loss",
    port_selector_left=select_port_atom,
    port_selector_right=select_port_atom,
    leaving_selector_left=select_one_hydrogen,
    leaving_selector_right=select_one_hydrogen,
    bond_former=form_single_bond,
)

# Mark ports on atoms
atom_a["port"] = "1"
atom_b["port"] = "2"

# Execute reaction between two Atomistic structures
product = cc_coupling.run(left=struct_a, right=struct_b, port_L="1", port_R="2")
print(f"Removed atoms: {product.removed_atoms}")
print(f"New bonds: {product.new_bonds}")
```

Design Goals:
-------------
- Pure Python, framework-native (no RDKit)
- Composable: reaction logic = modular functions
- Stable indexing: atom deletion doesn't shift IDs
- Single responsibility: one Reacter = one reaction type
- Extensible: easy to subclass for specialized reactions
- Auditable: all changes recorded in ReactionResult.notes
"""

from .base import (
    ProductInfo,
    ReactantInfo,
    Reacter,
    ReactionMetadata,
    ReactionResult,
    TopologyChanges,
)
from .connector import MonomerLinker
from .selectors import (
    select_all_hydrogens,
    select_dummy_atoms,
    select_hydroxyl_group,
    select_none,
    select_one_hydrogen,
    select_port_atom,
    select_prev_atom,
)
from .topology_detector import TopologyDetector
from .transformers import (
    break_bond,
    create_bond_former,
    form_aromatic_bond,
    form_double_bond,
    form_single_bond,
    form_triple_bond,
    skip_bond_formation,
)
from .utils import create_atom_mapping, find_neighbors

__all__ = [
    # Core classes
    "MonomerLinker",
    "ProductInfo",
    "ReactantInfo",
    "ReactionMetadata",
    "ReactionResult",
    "Reacter",
    "TopologyChanges",
    "TopologyDetector",
    # Transformers (Bond Formers)
    "break_bond",
    "create_bond_former",
    # Utilities
    "create_atom_mapping",
    "find_neighbors",
    "form_aromatic_bond",
    "form_double_bond",
    "form_single_bond",
    "form_triple_bond",
    # Selectors
    "select_all_hydrogens",
    "select_dummy_atoms",
    "select_hydroxyl_group",
    "select_none",
    "select_one_hydrogen",
    "select_port_atom",
    "select_prev_atom",
    "skip_bond_formation",
]
