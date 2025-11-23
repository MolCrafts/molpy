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
AtomEntity = Entity  # Direct reference to an Atom entity

# Callable type signatures for reaction components
PortSelector = Callable[[Monomer, str], AtomEntity]
"""
Select port atom from a monomer given a port name.

Args:
    monomer: The monomer to select from
    port_name: Name of the port to use for selection

Returns:
    AtomEntity: The port atom entity
"""

LeavingSelector = Callable[[Monomer, AtomEntity], list[AtomEntity]]
"""
Select leaving group atoms given a port atom.

Args:
    monomer: The monomer containing the atoms
    port_atom: The port atom entity

Returns:
    List of atom entities to be removed
"""

BondFormer = Callable[[Atomistic, AtomEntity, AtomEntity], None]
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
class ReactionProduct:
    """
    Container for reaction products and metadata.

    Attributes:
        product: The resulting Atomistic assembly after reaction
        notes: Dictionary containing execution metadata:
            - 'reaction_name': str
            - 'eliminated_atoms': List of eliminated atom entities
            - 'n_eliminated': int
            - 'port_L': Entity (left port atom)
            - 'port_R': Entity (right port atom)
            - 'entity_maps': List of entity mappings from merge
            - 'formed_bonds': List of newly formed bonds
            - 'new_angles': List of newly created angles (if computed)
            - 'new_dihedrals': List of newly created dihedrals (if computed)
            - 'modified_atoms': List of atoms whose types may have changed
            - 'requires_retype': bool indicating if retypification needed
    """

    product: Atomistic
    notes: dict[str, Any] = field(default_factory=dict)


class Reacter:
    """
    Programmable chemical reaction executor.

    A Reacter represents one specific chemical reaction type by composing:
    1. Port selectors - identify reactive atoms via ports
    2. Leaving selectors - identify atoms to remove
    3. Bond former - create new bonds between port atoms

    The reaction is executed on copies of input monomers, ensuring
    original structures remain unchanged.

    **Port Selection Philosophy:**
    Reacter does NOT handle port selection. The caller (e.g., MonomerLinker)
    must explicitly specify which ports to connect via port_L and port_R.
    This makes the reaction execution deterministic and explicit.

    Attributes:
        name: Descriptive name for this reaction type
        port_selector_left: Function to select left port atom
        port_selector_right: Function to select right port atom
        leaving_selector_left: Function to select left leaving group
        leaving_selector_right: Function to select right leaving group
        bond_former: Function to create bond between port atoms

    Example:
        >>> from molpy.reacter import Reacter, select_port_atom, select_one_hydrogen, form_single_bond
        >>>
        >>> cc_coupling = Reacter(
        ...     name="C-C_coupling_with_H_loss",
        ...     port_selector_left=select_port_atom,
        ...     port_selector_right=select_port_atom,
        ...     leaving_selector_left=select_one_hydrogen,
        ...     leaving_selector_right=select_one_hydrogen,
        ...     bond_former=form_single_bond,
        ... )
        >>>
        >>> # Port selection is explicit!
        >>> product = cc_coupling.run(monomerA, monomerB, port_L="1", port_R="2")
        >>> print(product.notes['eliminated_atoms'])  # [H1, H2]
    """

    def __init__(
        self,
        name: str,
        port_selector_left: PortSelector,
        port_selector_right: PortSelector,
        leaving_selector_left: LeavingSelector,
        leaving_selector_right: LeavingSelector,
        bond_former: BondFormer,
    ):
        """
        Initialize a Reacter with reaction components.

        Args:
            name: Descriptive name for this reaction
            port_selector_left: Selector for left port atom
            port_selector_right: Selector for right port atom
            leaving_selector_left: Selector for left leaving group
            leaving_selector_right: Selector for right leaving group
            bond_former: Function to create bond between port atoms
        """
        self.name = name
        self.port_selector_left = port_selector_left
        self.port_selector_right = port_selector_right
        self.leaving_selector_left = leaving_selector_left
        self.leaving_selector_right = leaving_selector_right
        self.bond_former = bond_former

    def run(
        self,
        left: Monomer,
        right: Monomer,
        port_L: str,
        port_R: str,
        compute_topology: bool = True,
        record_intermediates: bool = False,
    ) -> ReactionProduct:
        """
        Execute the reaction between two monomers.

        **IMPORTANT: port_L and port_R must be explicitly specified.**
        No automatic port selection is performed.

        Workflow (STRICT ORDER):
        1. Check ports exist on monomers
        2. Select port atoms via ports
        3. Merge right into left (direct transfer, no copy)
        4. Create bond between port atoms
        5. Remove leaving groups from MERGED assembly
        6. (Optional) Compute new angles/dihedrals
        7. Return ReactionProduct with metadata

        Args:
            left: Left reactant monomer
            right: Right reactant monomer
            port_L: Port name on left monomer (REQUIRED - must be explicit)
            port_R: Port name on right monomer (REQUIRED - must be explicit)
            compute_topology: If True, compute new angles/dihedrals (default True)
            record_intermediates: If True, record intermediate states in notes

        Returns:
            ReactionProduct containing:
                - product: Final product assembly
                - notes: Metadata including intermediate states if requested

        Raises:
            ValueError: If ports not found or port atoms invalid
        """
        intermediates: list[dict] | None = [] if record_intermediates else None

        # Step 1: Check ports exist on monomers
        port_atom_L = self.port_selector_left(left, port_L)  # Validate left port
        port_atom_R = self.port_selector_right(right, port_R)  # Validate right port
        if intermediates is not None:
            # Record initial state (copy first, then gen_topo on the copy!)
            left_copy = left.copy()
            right_copy = right.copy()
            if compute_topology:
                left_copy.get_topo(gen_angle=True, gen_dihe=True)
                right_copy.get_topo(gen_angle=True, gen_dihe=True)

            intermediates.append(
                {
                    "step": "initial",
                    "description": "Initial reactants (Step 1-2: validated ports and port atoms)",
                    "left": left_copy,
                    "right": right_copy,
                }
            )

        # Step 3: Merge right into left (direct transfer, entities are SAME objects)
        left.merge(right)

        # Step 4: Create bond between port atoms
        # Bond former is responsible for adding the bond to the assembly
        new_bond = self.bond_former(left, port_atom_L, port_atom_R)
        # Note: we don't call left.add_link(new_bond) here because
        # standard bond formers already do it. If a custom bond former returns
        # a bond but doesn't add it, it won't be in the assembly.

        if intermediates is not None:
            product_copy = left.copy()
            if compute_topology:
                product_copy.get_topo(gen_angle=True, gen_dihe=True)

            intermediates.append(
                {
                    "step": "bond_formation",
                    "description": "After forming new bond between port atoms (Step 4)",
                    "product": product_copy,
                    "new_bond": new_bond,
                }
            )

        # Step 5: Remove leaving groups from MERGED assembly
        # IMPORTANT: After merge, right's entities are moved to left,
        # so we need to wrap left in a temporary monomer for the selector
        merged_monomer = left
        leaving_L = self.leaving_selector_left(merged_monomer, port_atom_L)
        leaving_R = self.leaving_selector_right(merged_monomer, port_atom_R)

        removed_atoms = []
        if leaving_L:
            left.remove_entity(*leaving_L, drop_incident_links=True)
            removed_atoms.extend(leaving_L)
        if leaving_R:
            left.remove_entity(*leaving_R, drop_incident_links=True)
            removed_atoms.extend(leaving_R)

        if intermediates is not None:
            product_copy = left.copy()
            if compute_topology:
                product_copy.get_topo(gen_angle=True, gen_dihe=True)

            removed_meta = [
                {"symbol": a.get("symbol"), "id": a.get("id"), "ref": a}
                for a in removed_atoms
            ]
            intermediates.append(
                {
                    "step": "remove_leaving",
                    "description": f"After removing {len(removed_atoms)} leaving atoms (Step 5b)",
                    "product": product_copy,
                    "removed_atoms": removed_atoms,
                    "removed_meta": removed_meta,
                }
            )

        # Step 6: (Optional) Compute new angles/dihedrals
        if compute_topology:
            # Sanitize links: remove any self-loop links (endpoints duplicated)
            doomed: list = []
            for lcls in list(left.links.classes()):
                bucket = left.links.bucket(lcls)
                for l in list(bucket):
                    if len(set(l.endpoints)) != len(l.endpoints):
                        doomed.append(l)
            if doomed:
                left.remove_link(*doomed)

            topo = left.get_topo(gen_angle=True, gen_dihe=True)

            if intermediates is not None:
                product_copy = left.copy()
                product_copy.get_topo(gen_angle=True, gen_dihe=True)

                intermediates.append(
                    {
                        "step": "final",
                        "description": "Final topology computation (Step 6)",
                        "product": product_copy,
                        "n_angles": topo.n_angles,
                        "n_dihedrals": topo.n_dihedrals,
                    }
                )

        # Step 7: Return ReactionProduct with metadata
        notes = {
            "reaction_name": self.name,
            "eliminated_atoms": removed_atoms,
            "n_eliminated": len(removed_atoms),
            "port_L": port_atom_L,
            "port_R": port_atom_R,
            "port_name_L": port_L,
            "port_name_R": port_R,
            "formed_bonds": [new_bond] if new_bond else [],
            "requires_retype": True,
        }

        if record_intermediates:
            notes["intermediates"] = intermediates

        return ReactionProduct(product=left, notes=notes)

    def __repr__(self) -> str:
        return f"Reacter(name={self.name!r})"
