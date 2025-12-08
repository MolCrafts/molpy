"""
Core Reacter implementation for chemical transformations.

This module defines the base Reacter class and ProductSet dataclass,
providing the foundation for SMIRKS-style reaction semantics.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from molpy.core.atomistic import Angle, Atomistic, Bond, Dihedral, Atom
from molpy.core.entity import Entity
from molpy.reacter.topology_detector import TopologyDetector

from molpy.typifier.atomistic import TypifierBase

# Callable type signatures for reaction components
PortSelector = Callable[[Atomistic, str], Atom]
"""
Select port atom from an Atomistic structure given a port name.

The port name should be stored as a "port" attribute on the atom,
or atoms can have multiple ports stored as a "ports" list.

Args:
    assembly: The Atomistic structure to select from
    port_name: Name of the port to use for selection

Returns:
    Atom: The port atom entity
"""

LeavingSelector = Callable[[Atomistic, Atom], list[Atom]]
"""
Select leaving group atoms given a port atom.

Args:
    assembly: The Atomistic structure containing the atoms
    port_atom: The port atom entity

Returns:
    List of atom entities to be removed
"""

BondFormer = Callable[[Atomistic, Atom, Atom], Bond | None]
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
class ReactantInfo:
    """Information about the reactants in a reaction."""

    merged_reactants: Atomistic  # Merged reactants with reassigned IDs
    port_L: str
    port_R: str


@dataclass
class ProductInfo:
    """Information about the reaction product."""

    product: Atomistic
    port_L_atom: Atom | None = None
    port_R_atom: Atom | None = None


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

    **Important: port_L_atom and port_R_atom are actual reaction sites.**

    The `port_L_atom` and `port_R_atom` attributes contain the actual reaction site atoms
    selected by the port selector functions. For example, in EO polymerization,
    the actual reaction site is C or H.

    This class organizes reaction information into logical groups:
    - reactant_info: Information about the reactants
    - product_info: Information about the product
    - topology_changes: All topology changes (bonds, angles, dihedrals)
    - metadata: Reaction metadata (name, retyping info, etc.)

    For backward compatibility, the original flat structure is also available
    through properties that access the nested structures.
    """

    reactant_info: ReactantInfo
    product_info: ProductInfo
    topology_changes: TopologyChanges
    metadata: ReactionMetadata

    # Backward compatibility properties
    @property
    def reactants(self) -> Atomistic:
        """Backward compatibility: get reactants (merged left and right)."""
        return self.reactant_info.merged_reactants

    @property
    def product(self) -> Atomistic:
        """Backward compatibility: get product."""
        return self.product_info.product

    @property
    def reaction_name(self) -> str:
        """Backward compatibility: get reaction name."""
        return self.metadata.reaction_name

    @property
    def port_L(self) -> Atom | None:
        """Backward compatibility: get left port atom."""
        return self.product_info.port_L_atom

    @property
    def port_R(self) -> Atom | None:
        """Backward compatibility: get right port atom."""
        return self.product_info.port_R_atom

    @property
    def removed_atoms(self) -> list[Atom]:
        """Backward compatibility: get removed atoms."""
        return self.topology_changes.removed_atoms

    @property
    def new_bonds(self) -> list[Any]:
        """Backward compatibility: get new bonds."""
        return self.topology_changes.new_bonds

    @property
    def new_angles(self) -> list[Any]:
        """Backward compatibility: get new angles."""
        return self.topology_changes.new_angles

    @property
    def new_dihedrals(self) -> list[Any]:
        """Backward compatibility: get new dihedrals."""
        return self.topology_changes.new_dihedrals

    @property
    def modified_atoms(self) -> set[Atom]:
        """Backward compatibility: get modified atoms."""
        return self.topology_changes.modified_atoms

    @property
    def requires_retype(self) -> bool:
        """Backward compatibility: get retyping requirement."""
        return self.metadata.requires_retype

    @property
    def entity_maps(self) -> list[dict[Entity, Entity]]:
        """Backward compatibility: get entity maps."""
        return self.metadata.entity_maps

    @property
    def intermediates(self) -> list[dict]:
        """Backward compatibility: get intermediates."""
        return self.metadata.intermediates

    @property
    def removed_angles(self) -> list[Angle]:
        """Get removed angles (new property)."""
        return self.topology_changes.removed_angles

    @property
    def removed_dihedrals(self) -> list[Dihedral]:
        """Get removed dihedrals (new property)."""
        return self.topology_changes.removed_dihedrals


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
    Ports are marked directly on atoms using the "port" or "ports" attribute.
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
        port_atom_L: Atom,
        port_atom_R: Atom,
        leaving_L: list[Atom],
        leaving_R: list[Atom],
        port_L: str,
        port_R: str,
    ) -> tuple[Bond | None, list[Atom]]:
        """
        Execute the reaction core steps: bond formation and leaving group removal.

        Args:
            assembly: The merged assembly
            port_atom_L: Left port atom
            port_atom_R: Right port atom
            leaving_L: Left leaving group atoms
            leaving_R: Right leaving group atoms
            port_L: Left port name
            port_R: Right port name

        Returns:
            Tuple of (new_bond, removed_atoms)
        """
        # Form bond
        new_bond = self.bond_former(assembly, port_atom_L, port_atom_R)

        # Remove port markers
        if port_atom_L.get("port") == port_L:
            del port_atom_L["port"]
        if "ports" in port_atom_L:
            ports_list = port_atom_L.get("ports", [])
            if isinstance(ports_list, list):
                ports_list = [p for p in ports_list if p != port_L]
                if ports_list:
                    port_atom_L["ports"] = ports_list
                else:
                    del port_atom_L["ports"]

        if port_atom_R.get("port") == port_R:
            del port_atom_R["port"]
        if "ports" in port_atom_R:
            ports_list = port_atom_R.get("ports", [])
            if isinstance(ports_list, list):
                ports_list = [p for p in ports_list if p != port_R]
                if ports_list:
                    port_atom_R["ports"] = ports_list
                else:
                    del port_atom_R["ports"]

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
        port_L: str,
        port_R: str,
        compute_topology: bool = True,
        record_intermediates: bool = False,
        typifier: TypifierBase | None = None,
    ) -> ReactionResult:
        """
        Execute the reaction between two Atomistic structures.

        **IMPORTANT: port_L and port_R must be explicitly specified.**
        No automatic port selection is performed.

        Workflow (STRICT ORDER):
        1. Check ports exist on structures (atoms must have port markers)
        2. Select port atoms via port selectors
        3. Merge right into left (direct transfer, no copy)
        4. Create bond between port atoms
        5. Remove leaving groups from MERGED assembly
        6. (Optional) Compute new angles/dihedrals
        7. Return ReactionResult with metadata

        Args:
            left: Left reactant Atomistic structure
            right: Right reactant Atomistic structure
            port_L: Port name on left structure (REQUIRED - must be explicit)
            port_R: Port name on right structure (REQUIRED - must be explicit)
            compute_topology: If True, compute new angles/dihedrals (default True)
            record_intermediates: If True, record intermediate states in notes
            typifier: Optional typifier for incremental retypification.
                                 If provided, will retype only affected atoms and
                                 topology items (new bonds/angles/dihedrals and modified atoms)
                                 after the reaction (default None)

        Returns:
            ReactionResult containing:
                - product: Final product assembly
                - notes: Metadata including intermediate states if requested

        Raises:
            ValueError: If ports not found or port atoms invalid
        """
        intermediates: list[dict] = []

        # Step 1: Prepare reactants
        left_copy, right_copy, left_entity_map, right_entity_map = (
            self._prepare_reactants(left, right)
        )

        # Step 2: Select port atoms and leaving groups
        port_atom_L = self.port_selector_left(left_copy, port_L)
        port_atom_R = self.port_selector_right(right_copy, port_R)

        leaving_L = self.leaving_selector_left(left_copy, port_atom_L)
        leaving_R = self.leaving_selector_right(right_copy, port_atom_R)

        # Step 3: Merge structures
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
            port_atom_L,
            port_atom_R,
            leaving_L,
            leaving_R,
            port_L,
            port_R,
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
            port_L=port_L,
            port_R=port_R,
        )

        product_info = ProductInfo(
            product=merged,
            port_L_atom=port_atom_L,
            port_R_atom=port_atom_R,
        )

        topology_changes = TopologyChanges(
            new_bonds=[new_bond] if new_bond else [],
            new_angles=new_angles,
            new_dihedrals=new_dihedrals,
            removed_angles=removed_angles,
            removed_dihedrals=removed_dihedrals,
            removed_atoms=removed_atoms,
            modified_atoms=(
                {port_atom_L, port_atom_R} if port_atom_L and port_atom_R else set()
            ),
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
        - Modified atoms (port atoms where bonds were formed)
        - New bonds, angles, and dihedrals
        - Existing bonds/angles/dihedrals involving modified atoms

        Args:
            assembly: The product assembly structure (will be modified)
            reaction_result: Result from the reaction containing exact topology changes
            typifier: OPLS typifier for assigning types
        """
        modified_atoms = reaction_result.modified_atoms
        new_bonds = reaction_result.new_bonds
        new_angles = reaction_result.new_angles
        new_dihedrals = reaction_result.new_dihedrals

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
