"""Chemical reactions on native molpy structures — start here.

Entry points (most users need only these):

- :class:`Reacter` — execute one reaction type between two structures;
  compose it from anchor selectors, leaving selectors, and a bond former
- :func:`find_port` — locate a port-marked atom to pass into
  :meth:`Reacter.run`
- :class:`BondReactReacter` — :class:`Reacter` plus LAMMPS
  ``fix bond/react`` template generation (serialize with
  :func:`molpy.io.write_lammps_bond_react_system`)

Building blocks:

- **Anchor selectors** (``select_self``, ``select_neighbor``, …) map a
  port atom to the atom that actually forms the bond
- **Leaving selectors** (``select_hydrogens``, ``select_hydroxyl_group``,
  ``select_none``, …) choose atoms to remove
- **Bond formers** (``form_single_bond``, ``form_double_bond``, …)
  create the new bond

Example::

    from molpy.reacter import (
        Reacter, find_port, form_single_bond,
        select_hydrogens, select_self,
    )

    cc_coupling = Reacter(
        name="C-C_coupling_with_H_loss",
        anchor_selector_left=select_self,
        anchor_selector_right=select_self,
        leaving_selector_left=select_hydrogens(1),
        leaving_selector_right=select_hydrogens(1),
        bond_former=form_single_bond,
    )
    result = cc_coupling.run(
        left=struct_a, right=struct_b,
        port_atom_L=find_port(struct_a, ">"),
        port_atom_R=find_port(struct_b, "<"),
    )
    product = result.product  # caller inputs are never mutated

Design goals: pure Python and framework-native (no RDKit); composable
modular reaction logic; deterministic, explicit port selection; one
Reacter = one reaction type.
"""

from .base import (
    Reacter,
    ReactionResult,
)
from .bond_react import (
    BondReactReacter,
    BondReactResult,
    BondReactTemplate,
)
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
    find_port,
    find_port_atom_by_node,
    # High-level convenience selectors
    Selector,
    select_self,
    select_hydrogens,
    select_neighbor,
)

from .topology_detector import TopologyDetector
from .utils import (
    break_bond,
    create_atom_mapping,
    create_bond_former,
    find_neighbors,
    form_aromatic_bond,
    form_double_bond,
    form_single_bond,
    form_triple_bond,
    skip_bond_formation,
)

__all__ = [
    # Core classes
    "ReactionResult",
    "Reacter",
    "BondReactReacter",
    "BondReactResult",
    "BondReactTemplate",
    "TopologyDetector",
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
    "find_port",
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
    # High-level convenience selectors
    "Selector",
    "select_self",
    "select_hydrogens",
    "select_neighbor",
    "find_port",
]
