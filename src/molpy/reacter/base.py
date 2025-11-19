"""
Core Reacter implementation for chemical transformations.

This module defines the base Reacter class and ProductSet dataclass,
providing the foundation for SMIRKS-style reaction semantics.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from molpy import Atomistic
from molpy.core.entity import Entity
from molpy.core.wrappers.monomer import Monomer

# Type aliases for clarity
AtomId = Entity  # Direct reference to an Atom entity

# Callable type signatures for reaction components
AnchorSelector = Callable[[Monomer, str], AtomId]
"""
Select anchor atom from a monomer given a port name.

Args:
    monomer: The monomer to select from
    port_name: Name of the port to use for selection

Returns:
    AtomId: The anchor atom entity
"""

LeavingSelector = Callable[[Monomer, AtomId], list[AtomId]]
"""
Select leaving group atoms given an anchor atom.

Args:
    monomer: The monomer containing the atoms
    anchor: The anchor atom entity

Returns:
    List of atom entities to be removed
"""

BondMaker = Callable[[Atomistic, AtomId, AtomId], None]
"""
Create or modify bonds between two atoms in an assembly.

Args:
    assembly: The atomistic assembly to modify
    i: First atom entity
    j: Second atom entity

Side effects:
    Adds bonds to assembly.links
"""


@dataclass
class ProductSet:
    """
    Container for reaction products and metadata.

    Attributes:
        product: The resulting Atomistic assembly after reaction
        notes: Dictionary containing execution metadata:
            - 'reaction_name': str
            - 'removed_atoms': List of removed atom entities
            - 'removed_count': int
            - 'anchor_left': Entity
            - 'anchor_right': Entity
            - 'entity_maps': List of entity mappings from merge
            - 'new_bonds': List of newly created bonds
            - 'new_angles': List of newly created angles (if computed)
            - 'new_dihedrals': List of newly created dihedrals (if computed)
            - 'modified_atoms': List of atoms whose types may have changed
            - 'needs_retypification': bool indicating if retypification needed
    """

    product: Atomistic
    notes: dict[str, Any] = field(default_factory=dict)


class Reacter:
    """
    Programmable chemical reaction executor.

    A Reacter represents one specific chemical reaction type by composing:
    1. Anchor selectors - identify reactive atoms via ports
    2. Leaving group selectors - identify atoms to remove
    3. Bond maker - create new bonds between anchors

    The reaction is executed on copies of input monomers, ensuring
    original structures remain unchanged.

    **Port Selection Philosophy:**
    Reacter does NOT handle port selection. The caller (e.g., ReacterConnector)
    must explicitly specify which ports to connect via port_L and port_R.
    This makes the reaction execution deterministic and explicit.

    Attributes:
        name: Descriptive name for this reaction type
        anchor_left: Function to select left anchor atom
        anchor_right: Function to select right anchor atom
        leaving_left: Function to select left leaving group
        leaving_right: Function to select right leaving group
        bond_maker: Function to create bond between anchors

    Example:
        >>> from molpy.reacter import Reacter, port_anchor_selector, remove_one_H, make_single_bond
        >>>
        >>> cc_coupling = Reacter(
        ...     name="C-C_coupling_with_H_loss",
        ...     anchor_left=port_anchor_selector,
        ...     anchor_right=port_anchor_selector,
        ...     leaving_left=remove_one_H,
        ...     leaving_right=remove_one_H,
        ...     bond_maker=make_single_bond,
        ... )
        >>>
        >>> # Port selection is explicit!
        >>> product = cc_coupling.run(monomerA, monomerB, port_L="1", port_R="2")
        >>> print(product.notes['removed_atoms'])  # [H1, H2]
    """

    def __init__(
        self,
        name: str,
        anchor_left: AnchorSelector,
        anchor_right: AnchorSelector,
        leaving_left: LeavingSelector,
        leaving_right: LeavingSelector,
        bond_maker: BondMaker,
    ):
        """
        Initialize a Reacter with reaction components.

        Args:
            name: Descriptive name for this reaction
            anchor_left: Selector for left anchor atom
            anchor_right: Selector for right anchor atom
            leaving_left: Selector for left leaving group
            leaving_right: Selector for right leaving group
            bond_maker: Function to create bond between anchors
        """
        self.name = name
        self.anchor_left = anchor_left
        self.anchor_right = anchor_right
        self.leaving_left = leaving_left
        self.leaving_right = leaving_right
        self.bond_maker = bond_maker

    def run(
        self,
        left: Monomer,
        right: Monomer,
        port_L: str,
        port_R: str,
        compute_topology: bool = True,
        record_intermediates: bool = False,
    ) -> ProductSet:
        """
        Execute the reaction between two monomers.

        **IMPORTANT: port_L and port_R must be explicitly specified.**
        No automatic port selection is performed.

        Workflow (STRICT ORDER):
        1. Check ports exist on monomers
        2. Select anchors via ports
        3. Merge right into left (direct transfer, no copy)
        4. Create bond between anchors
        5. Remove leaving groups from MERGED assembly
        6. (Optional) Compute new angles/dihedrals
        7. Return ProductSet with metadata

        Args:
            left: Left reactant monomer
            right: Right reactant monomer
            port_L: Port name on left monomer (REQUIRED - must be explicit)
            port_R: Port name on right monomer (REQUIRED - must be explicit)
            compute_topology: If True, compute new angles/dihedrals (default True)
            record_intermediates: If True, record intermediate states in notes

        Returns:
            ProductSet containing:
                - product: Final product assembly
                - notes: Metadata including intermediate states if requested

        Raises:
            ValueError: If ports not found or anchors invalid
        """
        intermediates: list[dict] | None = [] if record_intermediates else None

        # Step 1: Check ports exist on monomers
        left_port = left.get_port_def(port_L)
        right_port = right.get_port_def(port_R)

        if left_port is None:
            raise ValueError(f"Port '{port_L}' not found on left monomer")
        if right_port is None:
            raise ValueError(f"Port '{port_R}' not found on right monomer")

        # Step 2: Select anchors via ports
        anchor_L = left_port.target
        anchor_R = right_port.target

        # Get unwrapped assemblies
        left_asm = left
        right_asm = right

        if intermediates is not None:
            # Record initial state (copy first, then gen_topo on the copy!)
            left_copy = left_asm.copy()
            right_copy = right_asm.copy()
            if compute_topology:
                left_copy.get_topo(gen_angle=True, gen_dihe=True)
                right_copy.get_topo(gen_angle=True, gen_dihe=True)

            intermediates.append(
                {
                    "step": "initial",
                    "description": "Initial reactants (Step 1-2: validated ports and anchors)",
                    "left": left_copy,
                    "right": right_copy,
                }
            )

        # Step 3: Merge right into left (direct transfer, entities are SAME objects)
        left_asm.merge(right_asm)

        if intermediates is not None:
            product_copy = left_asm.copy()
            if compute_topology:
                product_copy.get_topo(gen_angle=True, gen_dihe=True)

            intermediates.append(
                {
                    "step": "merge",
                    "description": "After merging right into left (Step 3)",
                    "product": product_copy,
                }
            )

        # Step 4: Create bond between anchors
        new_bond = self.bond_maker(left_asm, anchor_L, anchor_R)
        if new_bond is not None:
            left_asm.add_link(new_bond, include_endpoints=False)

        if intermediates is not None:
            product_copy = left_asm.copy()
            if compute_topology:
                product_copy.get_topo(gen_angle=True, gen_dihe=True)

            intermediates.append(
                {
                    "step": "bond_formation",
                    "description": "After forming new bond between anchors (Step 4)",
                    "product": product_copy,
                    "new_bond": new_bond,
                }
            )

        # Step 5: Remove leaving groups from MERGED assembly
        # IMPORTANT: After merge, right_asm's entities are moved to left_asm,
        # so we need to wrap left_asm in a temporary monomer for the selector
        merged_monomer = left_asm
        leaving_L = self.leaving_left(merged_monomer, anchor_L)
        leaving_R = self.leaving_right(merged_monomer, anchor_R)

        if intermediates is not None:
            intermediates.append(
                {
                    "step": "identify_leaving",
                    "description": f"Identified leaving groups (Step 5a): {len(leaving_L)} from left anchor, {len(leaving_R)} from right anchor",
                    "leaving_L": leaving_L,
                    "leaving_R": leaving_R,
                }
            )

        removed_atoms = []
        if leaving_L:
            left_asm.remove_entity(*leaving_L, drop_incident_links=True)
            removed_atoms.extend(leaving_L)
        if leaving_R:
            left_asm.remove_entity(*leaving_R, drop_incident_links=True)
            removed_atoms.extend(leaving_R)

        if intermediates is not None:
            product_copy = left_asm.copy()
            if compute_topology:
                product_copy.get_topo(gen_angle=True, gen_dihe=True)

            intermediates.append(
                {
                    "step": "remove_leaving",
                    "description": f"After removing {len(removed_atoms)} leaving atoms (Step 5b)",
                    "product": product_copy,
                }
            )

        # Step 6: (Optional) Compute new angles/dihedrals
        if compute_topology:
            topo = left_asm.get_topo(gen_angle=True, gen_dihe=True)

            if intermediates is not None:
                product_copy = left_asm.copy()
                product_copy.get_topo(gen_angle=True, gen_dihe=True)

                intermediates.append(
                    {
                        "step": "topology",
                        "description": "Final topology computation (Step 6)",
                        "product": product_copy,
                        "n_angles": topo.n_angles,
                        "n_dihedrals": topo.n_dihedrals,
                    }
                )

        # Step 7: Return ProductSet with metadata
        notes = {
            "reaction_name": self.name,
            "removed_atoms": removed_atoms,
            "removed_count": len(removed_atoms),
            "anchor_left": anchor_L,
            "anchor_right": anchor_R,
            "port_L": port_L,
            "port_R": port_R,
            "new_bonds": [new_bond] if new_bond else [],
            "needs_retypification": True,
        }

        if record_intermediates:
            notes["intermediates"] = intermediates

        return ProductSet(product=left_asm, notes=notes)

    def __repr__(self) -> str:
        return f"Reacter(name={self.name!r})"
