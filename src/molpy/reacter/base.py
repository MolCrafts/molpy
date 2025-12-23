"""
Core Reacter implementation for chemical transformations.

This module defines the base Reacter class and ProductSet dataclass,
providing the foundation for SMIRKS-style reaction semantics.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from molpy.core.atomistic import Angle, Atom, Atomistic, Bond, Dihedral
from molpy.core.entity import Entity
from molpy.reacter.topology_detector import TopologyDetector
from molpy.typifier.atomistic import TypifierBase

# Callable type signatures for reaction components
AnchorSelector = Callable[[Atomistic, Atom], Atom]
"""
Select **anchor atom** given a **port atom**.

Port atoms are SMILES-marked connection sites (e.g. $, *, <, >) stored on
atoms via the "port" / "ports" attributes. An *anchor* is the actual atom
that participates in bond formation.

Args:
    assembly: The Atomistic structure to select from
    port_atom: The port atom entity (SMILES-marked atom)

Returns:
    Atom: The anchor atom where the new bond should be formed
"""

LeavingSelector = Callable[[Atomistic, Atom], list[Atom]]
"""
Select leaving group atoms given an **anchor atom**.

Args:
    assembly: The Atomistic structure containing the atoms
    anchor_atom: The anchor atom entity (actual reaction site)

Returns:
    List of atom entities to be removed
"""

BondFormer = Callable[[Atomistic, Atom, Atom], Bond | None]
"""
Create or modify bonds between two **anchor atoms** in an assembly.

Args:
    assembly: The atomistic assembly to modify
    i: First anchor atom
    j: Second anchor atom

Side effects:
    Adds or updates bonds in assembly.links
"""


@dataclass
class ReactantInfo:
    """Information about the reactants in a reaction.

    Ports vs anchors:
    - **Ports** are SMILES-marked connection atoms (e.g. $, *, <, >)
    - **Anchors** are the actual atoms where bonds are formed

    This dataclass tracks the merged reactants and the original **port atoms**
    on each side before the reaction is executed.
    """

    merged_reactants: Atomistic
    port_atom_L: Entity | None = None
    port_atom_R: Entity | None = None


@dataclass
class ProductInfo:
    """Information about the reaction product.

    This captures the final product structure and the **anchor atoms**
    where the new bond was formed.
    """

    product: Atomistic
    anchor_L: Atom | None = None
    anchor_R: Atom | None = None


@dataclass
class TopologyChanges:
    """Topology changes resulting from the reaction."""

    new_bonds: list[Any] = field(default_factory=list)
    new_angles: list[Angle] = field(default_factory=list)
    new_dihedrals: list[Dihedral] = field(default_factory=list)
    removed_angles: list[Angle] = field(default_factory=list)
    removed_dihedrals: list[Dihedral] = field(default_factory=list)
    removed_atoms: list[Atom] = field(default_factory=list)
    modified_atoms: set[Atom] = field(default_factory=set)


@dataclass
class ReactionMetadata:
    """Metadata about the reaction."""

    reaction_name: str
    requires_retype: bool = False
    entity_maps: list[dict[Entity, Entity]] = field(default_factory=list)
    intermediates: list[dict] = field(default_factory=list)


@dataclass
class ReactionResult:
    """
    Container for reaction products and metadata with organized structure.

    This class organizes reaction information into logical groups:

    - ``reactant_info``: Information about the reactants and their **port atoms**
    - ``product_info``: Information about the product and the **anchor atoms**
    - ``topology_changes``: All topology changes (bonds, angles, dihedrals)
    - ``metadata``: Reaction metadata (name, retyping info, etc.)
    """

    reactant_info: ReactantInfo
    product_info: ProductInfo
    topology_changes: TopologyChanges
    metadata: ReactionMetadata


class Reacter:
    """
    Programmable chemical reaction executor.

    A Reacter represents one specific chemical reaction type by composing:
    1. Anchor selectors - map **port atoms** to **anchor atoms**
    2. Leaving selectors - identify atoms to remove
    3. Bond former - create new bonds between anchor atoms

    The reaction is executed on copies of input monomers, ensuring
    original structures remain unchanged.

    **Port Selection Philosophy:**
    Reacter does NOT handle port selection. The caller (e.g., MonomerLinker)
    must explicitly specify which ports to connect via port_L and port_R.
    Ports are marked directly on atoms using the "port" or "ports" attribute.
    This makes the reaction execution deterministic and explicit.

    Attributes:
        name: Descriptive name for this reaction type
        anchor_selector_left: Function to map left port atom to anchor atom
        anchor_selector_right: Function to map right port atom to anchor atom
        leaving_selector_left: Function to select left leaving group from left anchor
        leaving_selector_right: Function to select right leaving group from right anchor
        bond_former: Function to create bond between anchor atoms

    Example:
        >>> from molpy.reacter import Reacter, select_port_atom, select_one_hydrogen, form_single_bond
        >>> from molpy import Atomistic
        >>>
        >>> # Mark ports on atoms
        >>> atom_a["port"] = "1"
        >>> atom_b["port"] = "2"
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
        >>> product = cc_coupling.run(structA, structB, port_L="1", port_R="2")
        >>> print(product.removed_atoms)  # [H1, H2]
    """

    def __init__(
        self,
        name: str,
        anchor_selector_left: AnchorSelector,
        anchor_selector_right: AnchorSelector,
        leaving_selector_left: LeavingSelector,
        leaving_selector_right: LeavingSelector,
        bond_former: BondFormer,
    ):
        """
        Initialize a Reacter with reaction components.

        Args:
            name: Descriptive name for this reaction
            anchor_selector_left: Selector that maps left **port atom** to **anchor atom**
            anchor_selector_right: Selector that maps right **port atom** to **anchor atom**
            leaving_selector_left: Selector for left leaving group (from left anchor)
            leaving_selector_right: Selector for right leaving group (from right anchor)
            bond_former: Function to create bond between anchor atoms
        """
        self.name = name
        self.anchor_selector_left = anchor_selector_left
        self.anchor_selector_right = anchor_selector_right
        self.leaving_selector_left = leaving_selector_left
        self.leaving_selector_right = leaving_selector_right
        self.bond_former = bond_former

    def _prepare_reactants(
        self,
        left: Atomistic,
        right: Atomistic,
    ) -> tuple[Atomistic, Atomistic, dict[Entity, Entity], dict[Entity, Entity]]:
        """
        Prepare reactants by copying and building entity maps.

        Args:
            left: Left reactant structure
            right: Right reactant structure

        Returns:
            Tuple of (copied_left, copied_right, left_entity_map, right_entity_map)
        """
        # Save original atoms for entity mapping
        original_left_atoms = list(left.atoms)
        original_right_atoms = list(right.atoms)

        left_copy = left.copy()
        right_copy = right.copy()

        # Build entity map: original -> copied
        left_entity_map: dict[Entity, Entity] = {}
        copied_left_atoms = list(left_copy.atoms)
        if len(original_left_atoms) == len(copied_left_atoms):
            for orig, copied in zip(original_left_atoms, copied_left_atoms):
                left_entity_map[orig] = copied

        right_entity_map: dict[Entity, Entity] = {}
        copied_right_atoms = list(right_copy.atoms)
        if len(original_right_atoms) == len(copied_right_atoms):
            for orig, copied in zip(original_right_atoms, copied_right_atoms):
                right_entity_map[orig] = copied

        return left_copy, right_copy, left_entity_map, right_entity_map

    def _merge_structures(
        self,
        left: Atomistic,
        right: Atomistic,
    ) -> Atomistic:
        """
        Merge right structure into left structure.

        Args:
            left: Left structure
            right: Right structure

        Returns:
            Merged structure with updated atom IDs
        """
        merged = left.merge(right)
        for i, atom in enumerate(merged.atoms, start=1):
            atom["id"] = i
        return merged

    def _execute_reaction(
        self,
        assembly: Atomistic,
        anchor_L: Atom,
        anchor_R: Atom,
        leaving_L: list[Atom],
        leaving_R: list[Atom],
    ) -> tuple[Bond | None, list[Atom]]:
        """
        Execute the reaction core steps: bond formation and leaving group removal.

        Args:
            assembly: The merged assembly
            anchor_L: Left anchor atom (actual reaction site)
            anchor_R: Right anchor atom (actual reaction site)
            leaving_L: Left leaving group atoms
            leaving_R: Right leaving group atoms

        Returns:
            Tuple of (new_bond, removed_atoms)
        """
        # Form bond between anchors
        new_bond = self.bond_former(assembly, anchor_L, anchor_R)

        # Clear any port markers from anchors
        # (They may have already been cleared by selectors)
        if "port" in anchor_L:
            del anchor_L["port"]
        if "ports" in anchor_L:
            del anchor_L["ports"]

        if "port" in anchor_R:
            del anchor_R["port"]
        if "ports" in anchor_R:
            del anchor_R["ports"]

        # Remove leaving groups
        removed_atoms = []
        if leaving_L:
            assembly.remove_entity(*leaving_L, drop_incident_links=True)
            removed_atoms.extend(leaving_L)
        if leaving_R:
            assembly.remove_entity(*leaving_R, drop_incident_links=True)
            removed_atoms.extend(leaving_R)

        return new_bond, removed_atoms

    def _detect_and_update_topology(
        self,
        assembly: Atomistic,
        new_bonds: list[Bond],
        removed_atoms: list[Atom],
    ) -> tuple[list[Angle], list[Dihedral], list[Angle], list[Dihedral]]:
        """
        Detect and update topology changes using TopologyDetector.

        Args:
            assembly: The assembly structure
            new_bonds: List of newly formed bonds
            removed_atoms: List of removed atoms

        Returns:
            Tuple of (new_angles, new_dihedrals, removed_angles, removed_dihedrals)
        """
        return TopologyDetector.detect_and_update_topology(
            assembly, new_bonds, removed_atoms
        )

    def _build_entity_maps(
        self,
        left_entity_map: dict[Entity, Entity],
        right_entity_map: dict[Entity, Entity],
        product: Atomistic,
    ) -> dict[Entity, Entity]:
        """
        Build final entity map from original atoms to product atoms.

        Args:
            left_entity_map: Map from original left atoms to copied atoms
            right_entity_map: Map from original right atoms to copied atoms
            product: The final product assembly

        Returns:
            Final entity map from original to product atoms
        """
        final_entity_map: dict[Entity, Entity] = {}
        product_atoms = set(product.atoms)

        # Map left atoms: original -> copied -> product (if still present)
        for orig_left, copied_left in left_entity_map.items():
            if copied_left in product_atoms:
                final_entity_map[orig_left] = copied_left

        # Map right atoms: original -> copied -> product (if still present)
        for orig_right, copied_right in right_entity_map.items():
            if copied_right in product_atoms:
                final_entity_map[orig_right] = copied_right

        return final_entity_map

    def run(
        self,
        left: Atomistic,
        right: Atomistic,
        port_atom_L: Entity,
        port_atom_R: Entity,
        compute_topology: bool = True,
        record_intermediates: bool = False,
        typifier: TypifierBase | None = None,
    ) -> ReactionResult:
        """
        Execute the reaction between two Atomistic structures.

        **IMPORTANT: port_atom_L and port_atom_R must be explicit Atom objects.**
        Use find_port_atom() or find_port_atom_by_node() to get them first.

        Workflow:
        1. Transform port atoms to reaction sites via port selectors
        2. Merge structures (or work on single copy for ring closure)
        3. Select leaving groups from reaction sites
        4. Create bond between reaction sites
        5. Remove leaving groups
        6. (Optional) Compute new angles/dihedrals
        7. Return ReactionResult

        Args:
            left: Left reactant Atomistic structure
            right: Right reactant Atomistic structure
            port_atom_L: Port atom from left structure (the atom with port marker)
            port_atom_R: Port atom from right structure (the atom with port marker)
            compute_topology: If True, compute new angles/dihedrals (default True)
            record_intermediates: If True, record intermediate states
            typifier: Optional typifier for incremental retypification

        Returns:
            ReactionResult containing product and metadata

        Raises:
            ValueError: If port atoms invalid
        """
        intermediates: list[dict] = []

        # Check if this is a ring closure (left and right are the same object)
        is_ring_closure = left is right

        if is_ring_closure:
            # For ring closure, work directly on a single copy of the structure
            merged = left.copy()
            for i, atom in enumerate(merged.atoms, start=1):
                atom["id"] = i

            # Build entity map: original -> copied
            original_atoms = list(left.atoms)
            copied_atoms = list(merged.atoms)
            entity_map: dict[Entity, Entity] = {}
            if len(original_atoms) == len(copied_atoms):
                for orig, copied in zip(original_atoms, copied_atoms):
                    entity_map[orig] = copied

            left_entity_map = entity_map
            right_entity_map = entity_map

            # Map port atoms to their copies
            copied_port_L = entity_map.get(port_atom_L, port_atom_L)
            copied_port_R = entity_map.get(port_atom_R, port_atom_R)

            # Transform port atoms to anchors
            anchor_L = self.anchor_selector_left(merged, copied_port_L)
            anchor_R = self.anchor_selector_right(merged, copied_port_R)

            # Select leaving groups from anchors
            leaving_L = self.leaving_selector_left(merged, anchor_L)
            leaving_R = self.leaving_selector_right(merged, anchor_R)
        else:
            # Normal case: prepare reactants (creates copies)
            left_copy, right_copy, left_entity_map, right_entity_map = (
                self._prepare_reactants(left, right)
            )

            # Map port atoms to their copies
            copied_port_L = left_entity_map.get(port_atom_L, port_atom_L)
            copied_port_R = right_entity_map.get(port_atom_R, port_atom_R)

            # Transform port atoms to anchors
            anchor_L = self.anchor_selector_left(left_copy, copied_port_L)
            anchor_R = self.anchor_selector_right(right_copy, copied_port_R)

            # Select leaving groups from anchors
            leaving_L = self.leaving_selector_left(left_copy, anchor_L)
            leaving_R = self.leaving_selector_right(right_copy, anchor_R)

            # Merge structures
            merged = self._merge_structures(left_copy, right_copy)

        # Save merged reactants BEFORE reaction (for template generation)
        merged_reactants_before_reaction = merged.copy()
        merged_copy = merged.copy()

        if record_intermediates:
            if compute_topology:
                merged.get_topo(gen_angle=True, gen_dihe=True)
            intermediates.append(
                {
                    "step": "reactants",
                    "description": "Reactants before bond formation",
                    "product": merged_copy,
                }
            )

        # Step 4: Execute reaction (bond formation and leaving group removal)
        new_bond, removed_atoms = self._execute_reaction(
            merged,
            anchor_L,
            anchor_R,
            leaving_L,
            leaving_R,
        )

        if record_intermediates:
            product_copy = merged_copy.copy()
            if compute_topology:
                product_copy.get_topo(gen_angle=True, gen_dihe=True)

            intermediates.append(
                {
                    "step": "bond_formation",
                    "description": "After forming new bond between port atoms",
                    "product": product_copy,
                    "new_bond": new_bond,
                }
            )

            intermediates.append(
                {
                    "step": "remove_leaving",
                    "description": f"After removing {len(removed_atoms)} leaving atoms",
                    "product": product_copy,
                    "removed_atoms": removed_atoms,
                }
            )

        # Step 5: Detect and update topology if requested
        new_angles: list[Angle] = []
        new_dihedrals: list[Dihedral] = []
        removed_angles: list[Angle] = []
        removed_dihedrals: list[Dihedral] = []

        if compute_topology and new_bond:
            new_angles, new_dihedrals, removed_angles, removed_dihedrals = (
                self._detect_and_update_topology(merged, [new_bond], removed_atoms)
            )

        # Step 6: Build entity maps
        final_entity_map = self._build_entity_maps(
            left_entity_map, right_entity_map, merged
        )

        # Step 7: Determine if retypification is needed
        requires_retype = bool(new_bond or removed_atoms)

        # Step 8: Build result structure
        # Use merged reactants saved BEFORE reaction (with all atoms including removed_atoms)
        reactant_info = ReactantInfo(
            merged_reactants=merged_reactants_before_reaction,
            port_atom_L=port_atom_L,
            port_atom_R=port_atom_R,
        )

        product_info = ProductInfo(product=merged, anchor_L=anchor_L, anchor_R=anchor_R)

        topology_changes = TopologyChanges(
            new_bonds=[new_bond] if new_bond else [],
            new_angles=new_angles,
            new_dihedrals=new_dihedrals,
            removed_angles=removed_angles,
            removed_dihedrals=removed_dihedrals,
            removed_atoms=removed_atoms,
            modified_atoms=({anchor_L, anchor_R} if anchor_L and anchor_R else set()),
        )

        metadata = ReactionMetadata(
            reaction_name=self.name,
            requires_retype=requires_retype,
            entity_maps=[final_entity_map],
            intermediates=intermediates,
        )

        result = ReactionResult(
            reactant_info=reactant_info,
            product_info=product_info,
            topology_changes=topology_changes,
            metadata=metadata,
        )

        # Step 9: Perform incremental typification if requested
        if typifier:
            self._incremental_typify(merged, result, typifier)

        return result

    def _incremental_typify(
        self,
        assembly: Atomistic,
        reaction_result: ReactionResult,
        typifier: TypifierBase,
    ) -> None:
        """
        Perform incremental typification using exact information from reaction result.

        Types only the specific atoms and topology items that were affected:
        - Modified atoms (anchors where bonds were formed)
        - New bonds, angles, and dihedrals
        - Existing bonds/angles/dihedrals involving modified atoms

        Args:
            assembly: The product assembly structure (will be modified)
            reaction_result: Result from the reaction containing exact topology changes
            typifier: OPLS typifier for assigning types
        """
        modified_atoms = reaction_result.topology_changes.modified_atoms
        new_bonds = reaction_result.topology_changes.new_bonds
        new_angles = reaction_result.topology_changes.new_angles
        new_dihedrals = reaction_result.topology_changes.new_dihedrals

        # Step 1: Re-type modified atoms (port atoms where bonds were formed)
        if hasattr(typifier, "atom_typifier") and typifier.atom_typifier:
            # Clear types from modified atoms so they can be re-typed
            for atom in modified_atoms:
                if "type" in atom.data:
                    del atom.data["type"]

            # Re-type all atoms (graph matching needs full structure)
            typifier.atom_typifier.typify(assembly)

        # Step 2: Update pair types (charge, sigma, epsilon) for modified atoms
        if hasattr(typifier, "pair_typifier") and typifier.pair_typifier:
            from molpy.core.atomistic import Atom

            for atom in modified_atoms:
                if isinstance(atom, Atom):
                    typifier.pair_typifier.typify(atom)

        # Step 3: Type new bonds
        if hasattr(typifier, "bond_typifier") and typifier.bond_typifier:
            for bond in new_bonds:
                typifier.bond_typifier.typify(bond)

            # Re-type existing bonds involving modified atoms
            new_bonds_set = set(new_bonds)
            for bond in assembly.bonds:
                if bond in new_bonds_set:
                    continue  # Already typed above
                if bond.itom in modified_atoms or bond.jtom in modified_atoms:
                    if "type" in bond.data:
                        del bond.data["type"]
                    typifier.bond_typifier.typify(bond)

        # Step 4: Type new angles
        if hasattr(typifier, "angle_typifier") and typifier.angle_typifier:
            for angle in new_angles:
                typifier.angle_typifier.typify(angle)

            # Re-type existing angles involving modified atoms
            new_angles_set = set(new_angles)
            modified_atoms_set = modified_atoms
            for angle in assembly.angles:
                if angle in new_angles_set:
                    continue  # Already typed above
                if (
                    angle.itom in modified_atoms_set
                    or angle.jtom in modified_atoms_set
                    or angle.ktom in modified_atoms_set
                ):
                    if "type" in angle.data:
                        del angle.data["type"]
                    typifier.angle_typifier.typify(angle)

        # Step 5: Type new dihedrals
        if hasattr(typifier, "dihedral_typifier") and typifier.dihedral_typifier:
            for dihedral in new_dihedrals:
                typifier.dihedral_typifier.typify(dihedral)

            # Re-type existing dihedrals involving modified atoms
            new_dihedrals_set = set(new_dihedrals)
            modified_atoms_set = modified_atoms
            for dihedral in assembly.dihedrals:
                if dihedral in new_dihedrals_set:
                    continue  # Already typed above
                if (
                    dihedral.itom in modified_atoms_set
                    or dihedral.jtom in modified_atoms_set
                    or dihedral.ktom in modified_atoms_set
                    or dihedral.ltom in modified_atoms_set
                ):
                    if "type" in dihedral.data:
                        del dihedral.data["type"]
                    typifier.dihedral_typifier.typify(dihedral)

    def __repr__(self) -> str:
        return f"Reacter(name={self.name!r})"
