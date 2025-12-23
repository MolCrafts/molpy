"""
Programmable Reacter Module for Chemical Transformations.

This module provides a framework for defining and executing chemical reactions
within the molpy framework, following SMIRKS-style semantics but working
entirely on native data structures (Atom, Bond, Atomistic).

Core Concepts:
--------------
- **Reacter**: Represents a single chemical reaction type
- **ReactionResult**: Container for reaction products and metadata
- **Selectors**: Functions that map port atoms to anchors and choose leaving groups
- **Transformers**: Functions that create or modify bonds

Example Usage:
--------------
```python
from molpy.reacter import Reacter, select_port, select_one_hydrogen, form_single_bond

# Define a C-C coupling reaction
cc_coupling = Reacter(
    name="C-C_coupling_with_H_loss",
    anchor_selector_left=select_port,
    anchor_selector_right=select_port,
    leaving_selector_left=select_one_hydrogen,
    leaving_selector_right=select_one_hydrogen,
    bond_former=form_single_bond,
)

# Find port atoms first
port_atom_a = find_port_atom(struct_a, "1")
port_atom_b = find_port_atom(struct_b, "2")

# Execute reaction
result = cc_coupling.run(
    left=struct_a, right=struct_b,
    port_atom_L=port_atom_a, port_atom_R=port_atom_b
)
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
from .template import TemplateReacter, TemplateResult, write_template_files
from .selectors import (
    # Anchor selectors (transform port_atom to anchor atom)
    select_port,
    select_c_neighbor,
    select_o_neighbor,
    select_dehydration_left,
    select_dehydration_right,
    # Leaving selectors (identify atoms to remove)
    select_all_hydrogens,
    select_dummy_atoms,
    select_hydroxyl_group,
    select_hydroxyl_h_only,
    select_none,
    select_one_hydrogen,
    # Utilities
    find_port_atom,
    find_port_atom_by_node,
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
    "TemplateReacter",
    "TemplateResult",
    "TopologyChanges",
    "TopologyDetector",
    "write_template_files",
    # Transformers (Bond Formers)
    "break_bond",
    "create_bond_former",
    "form_aromatic_bond",
    "form_double_bond",
    "form_single_bond",
    "form_triple_bond",
    "skip_bond_formation",
    # Utilities
    "create_atom_mapping",
    "find_neighbors",
    "find_port_atom",
    "find_port_atom_by_node",
    # Anchor selectors
    "select_port",
    "select_c_neighbor",
    "select_o_neighbor",
    "select_dehydration_left",
    "select_dehydration_right",
    # Leaving selectors
    "select_all_hydrogens",
    "select_dummy_atoms",
    "select_hydroxyl_group",
    "select_hydroxyl_h_only",
    "select_none",
    "select_one_hydrogen",
]
