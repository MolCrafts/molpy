"""
TemplateReacter: Specialized Reacter for LAMMPS fix bond/react template generation.

This module provides a wrapper around the base Reacter that handles:
1. react_id assignment for atom tracking across reactions
2. Pre/post template extraction with correct atom/bond references
3. Writing LAMMPS molecule and map files
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List

import molpy as mp
from molpy.core.atomistic import Atom, Atomistic, Bond
from molpy.core.entity import Entity
from molpy.reacter.base import (
    AnchorSelector,
    BondFormer,
    LeavingSelector,
    Reacter,
    ReactionResult,
)
from molpy.typifier.atomistic import TypifierBase


@dataclass
class TemplateResult:
    """Result of template generation."""

    pre: Atomistic  # Pre-reaction subgraph (deep copied)
    post: Atomistic  # Post-reaction subgraph (deep copied)
    init_atoms: List[Atom]  # Initiator atoms (port atoms) in pre
    edge_atoms: List[Atom]  # Boundary atoms in pre
    removed_atoms: List[Atom]  # Atoms to delete
    pre_react_id_to_atom: dict  # react_id -> atom in pre
    post_react_id_to_atom: dict  # react_id -> atom in post


class TemplateReacter:
    """
    Reacter wrapper for LAMMPS fix bond/react template generation.

    Composes a Reacter internally and adds:
    1. react_id assignment before reaction
    2. Subgraph extraction for templates
    3. Correct atom/bond reference handling

    The function signature matches Reacter, but run_with_template() returns
    both ReactionResult and TemplateResult.

    Example:
        >>> from molpy.reacter import select_identity, select_one_hydrogen, form_single_bond
        >>> template_reacter = TemplateReacter(
        ...     name="C-C_coupling",
        ...     anchor_selector_left=select_identity,
        ...     anchor_selector_right=select_identity,
        ...     leaving_selector_left=select_one_hydrogen,
        ...     leaving_selector_right=select_one_hydrogen,
        ...     bond_former=form_single_bond,
        ...     radius=4
        ... )
        >>> result, template = template_reacter.run_with_template(
        ...     left, right, port_atom_L, port_atom_R
        ... )
        >>> template.pre  # Pre-reaction subgraph
        >>> template.post  # Post-reaction subgraph
    """

    def __init__(
        self,
        name: str,
        anchor_selector_left: AnchorSelector,
        anchor_selector_right: AnchorSelector,
        leaving_selector_left: LeavingSelector,
        leaving_selector_right: LeavingSelector,
        bond_former: BondFormer,
        radius: int = 4,
    ):
        """
        Initialize TemplateReacter.

        Args:
            name: Descriptive name for this reaction
            anchor_selector_left: Selector that maps left port atom to anchor atom
            anchor_selector_right: Selector that maps right port atom to anchor atom
            leaving_selector_left: Selector for left leaving group (from left anchor)
            leaving_selector_right: Selector for right leaving group (from right anchor)
            bond_former: Function to create bond between anchor atoms
            radius: Topological radius for subgraph extraction
        """
        # Create internal Reacter instance
        self.reacter = Reacter(
            name=name,
            anchor_selector_left=anchor_selector_left,
            anchor_selector_right=anchor_selector_right,
            leaving_selector_left=leaving_selector_left,
            leaving_selector_right=leaving_selector_right,
            bond_former=bond_former,
        )
        self.radius = radius
        self._react_id_counter = 0

    def run_with_template(
        self,
        left: Atomistic,
        right: Atomistic,
        port_atom_L: Entity,
        port_atom_R: Entity,
        compute_topology: bool = True,
        record_intermediates: bool = False,
        typifier: TypifierBase | None = None,
    ) -> tuple[ReactionResult, TemplateResult]:
        """
        Run reaction and generate templates.

        Function signature matches Reacter.run() but returns both
        ReactionResult and TemplateResult.

        Args:
            left: Left reactant structure
            right: Right reactant structure
            port_atom_L: Port atom in left structure
            port_atom_R: Port atom in right structure
            compute_topology: If True, compute new angles/dihedrals (default True)
            record_intermediates: If True, record intermediate states
            typifier: Optional typifier for incremental retypification

        Returns:
            Tuple of (ReactionResult, TemplateResult)
        """
        # Step 1: Assign react_id to ALL atoms BEFORE reaction
        self._assign_react_ids(left)
        self._assign_react_ids(right)

        # Verify port atoms have react_id
        if "react_id" not in port_atom_L.data:
            raise ValueError("port_atom_L missing react_id after assignment!")
        if "react_id" not in port_atom_R.data:
            raise ValueError("port_atom_R missing react_id after assignment!")

        # Step 2: Run the underlying reaction
        result = self.reacter.run(
            left,
            right,
            port_atom_L,
            port_atom_R,
            compute_topology=compute_topology,
            record_intermediates=record_intermediates,
            typifier=typifier,
        )

        # Step 3: Generate templates from result
        template = self._generate_template(result, port_atom_L, port_atom_R, typifier)

        return result, template

    def _assign_react_ids(self, struct: Atomistic) -> None:
        """Assign unique react_id to each atom that doesn't have one."""
        for atom in struct.atoms:
            if "react_id" not in atom.data:
                self._react_id_counter += 1
                atom["react_id"] = self._react_id_counter

    def _generate_template(
        self,
        result: ReactionResult,
        port_atom_L: Entity,
        port_atom_R: Entity,
        typifier: TypifierBase | None = None,
    ) -> TemplateResult:
        """
        Generate pre/post templates from reaction result.

        The key insight is:
        - Pre comes from merged_reactants (before reaction)
        - Post has the same atoms (by react_id) but from product + removed_atoms

        Note: Reacter already handles typification of product via _incremental_typify.
        We need to ensure reactants have topology and typification before extracting subgraph.
        """
        reactants = result.reactant_info.merged_reactants
        product = result.product_info.product
        removed_atoms = result.topology_changes.removed_atoms

        # Note: Reacter.run() already handles topology generation and typification
        # for reactants and product, so we don't need to do it again here

        # Get port atom react_ids
        port_L_rid = port_atom_L["react_id"]
        port_R_rid = port_atom_R["react_id"]

        # Step 1: Find port atoms in reactants by react_id
        port_atoms_in_reactants = self._find_atoms_by_react_id(
            reactants, [port_L_rid, port_R_rid]
        )

        if len(port_atoms_in_reactants) != 2:
            raise ValueError(
                f"Could not find port atoms in reactants! "
                f"Looking for react_ids {port_L_rid}, {port_R_rid}, "
                f"found {len(port_atoms_in_reactants)} atoms"
            )

        # Step 2: Extract pre subgraph from reactants (creates deep copies)
        # CRITICAL: extract_subgraph should use anchor atoms, not port atoms!
        # Anchor atoms are the actual reaction sites, and radius is topological distance from anchors
        anchor_L = self.reacter.anchor_selector_left(
            reactants, port_atoms_in_reactants[0]
        )
        anchor_R = self.reacter.anchor_selector_right(
            reactants, port_atoms_in_reactants[1]
        )
        anchor_atoms_in_reactants = [anchor_L, anchor_R]

        pre, pre_edge_entities = reactants.extract_subgraph(
            center_entities=anchor_atoms_in_reactants,
            radius=self.radius,
            entity_type=Atom,
            link_type=Bond,
        )

        # Typify pre (topology already preserved by extract_subgraph)
        if typifier:
            _ensure_typified(pre, typifier)

        # Build pre react_id -> atom mapping
        pre_react_id_to_atom = {}
        for atom in pre.atoms:
            if "react_id" not in atom.data:
                raise ValueError(
                    f"Atom in pre missing react_id! symbol={atom.get('symbol')}"
                )
            pre_react_id_to_atom[atom["react_id"]] = atom

        pre_react_ids = set(pre_react_id_to_atom.keys())
        # Keep ordered list for post to match pre's atom order
        pre_react_ids_ordered = [atom["react_id"] for atom in pre.atoms]

        # Collect removed atom react_ids
        removed_react_ids = set()
        for atom in removed_atoms:
            if "react_id" not in atom.data:
                raise ValueError(f"Removed atom missing react_id!")
            removed_react_ids.add(atom["react_id"])

        # Step 3: Build post by finding atoms via react_id
        # Pass reactants for topology sourcing
        post, post_react_id_to_atom = self._build_post(
            pre_react_ids_ordered,  # Ordered list to maintain atom order
            pre_react_ids,  # Set for quick lookups
            product,
            removed_atoms,
            removed_react_ids,
            pre,
            reactants=reactants,  # Source of original topology
        )

        # Step 4: Find init atoms in pre
        pre_init_atoms = []
        for rid in [port_L_rid, port_R_rid]:
            if rid in pre_react_id_to_atom:
                pre_init_atoms.append(pre_react_id_to_atom[rid])

        # Step 5: Typify post
        # Note: Reacter already removed all connections involving removed atoms,
        # so post should not have any bonds/angles/dihedrals between deleted and non-deleted atoms
        if typifier:
            _ensure_typified(post, typifier)

        return TemplateResult(
            pre=pre,
            post=post,
            init_atoms=pre_init_atoms,
            edge_atoms=list(pre_edge_entities),
            removed_atoms=removed_atoms,
            pre_react_id_to_atom=pre_react_id_to_atom,
            post_react_id_to_atom=post_react_id_to_atom,
        )

    def _find_atoms_by_react_id(
        self, struct: Atomistic, react_ids: List[int]
    ) -> List[Atom]:
        """Find atoms in struct by their react_id."""
        result = []
        react_id_set = set(react_ids)
        for atom in struct.atoms:
            if atom.get("react_id") in react_id_set:
                result.append(atom)
        return result

    def _build_post(
        self,
        pre_react_ids_ordered: List[
            int
        ],  # Ordered list of react_ids (same order as pre.atoms)
        pre_react_ids: set,  # Set for quick lookups
        product: Atomistic,
        removed_atoms: List[Atom],
        removed_react_ids: set,
        pre: Atomistic,
        reactants: Atomistic | None = None,
    ) -> tuple[Atomistic, dict]:
        """
        Build post template matching pre's atoms by react_id.

        Post template gets topology from:
        1. REACTANTS: for existing topology (not involving deleted atoms)
        2. PRODUCT: for new bonds/angles/dihedrals formed by reaction

        This hybrid approach ensures:
        - Original angles/dihedrals are preserved (from reactants)
        - New topology from reaction is included (from product)
        - Deleted atoms have no bonds in post

        Args:
            pre_react_ids: Set of react_ids in pre template
            product: Product structure (non-removed atoms only)
            removed_atoms: List of removed atoms
            removed_react_ids: Set of removed atom react_ids
            pre: Pre template (for reference)
            reactants: Merged reactants (source of original topology)

        Returns:
            Tuple of (post Atomistic, react_id -> atom mapping)
        """
        # Use pre if reactants not provided (backward compatibility)
        if reactants is None:
            reactants = pre

        # Build lookup tables
        product_by_rid = {a["react_id"]: a for a in product.atoms}
        removed_by_rid = {a["react_id"]: a for a in removed_atoms}

        # Build post atoms and mapping - iterate in pre's atom order!
        post_atoms = []
        post_react_id_to_atom = {}

        for rid in pre_react_ids_ordered:
            if rid in removed_react_ids:
                source = removed_by_rid.get(rid)
                if not source:
                    raise ValueError(
                        f"react_id {rid} marked as removed but not in removed_atoms!"
                    )
            else:
                source = product_by_rid.get(rid)
                if not source:
                    raise ValueError(f"react_id {rid} not found in product!")

            copied = Atom(deepcopy(source.data))
            post_atoms.append(copied)
            post_react_id_to_atom[rid] = copied

        # Build post structure
        post = Atomistic()
        post.add_atoms(post_atoms)

        # Helper to add topology items
        def _add_topology_item(
            endpoints_rids: List[int], data: dict, item_type: str
        ) -> bool:
            """Add a topology item if all endpoints are in template and none are deleted."""
            # Skip if any endpoint is not in our template
            if not all(rid in pre_react_ids for rid in endpoints_rids):
                return False

            # Skip if any endpoint is deleted (deleted atoms have no bonds in post)
            if any(rid in removed_react_ids for rid in endpoints_rids):
                return False

            # Get corresponding atoms in post
            post_eps = [post_react_id_to_atom[rid] for rid in endpoints_rids]

            # Check if already exists
            if item_type == "bond":
                exists = any(set(b.endpoints) == set(post_eps) for b in post.bonds)
                if not exists:
                    post.def_bond(*post_eps, **data)
                    return True
            elif item_type == "angle":
                # For angles, order matters: i-j-k
                exists = any(tuple(a.endpoints) == tuple(post_eps) for a in post.angles)
                if not exists:
                    post.def_angle(*post_eps, **data)
                    return True
            elif item_type == "dihedral":
                # For dihedrals, order matters: i-j-k-l
                exists = any(
                    tuple(d.endpoints) == tuple(post_eps) for d in post.dihedrals
                )
                if not exists:
                    post.def_dihedral(*post_eps, **data)
                    return True
            return False

        # Step 1: Get topology from REACTANTS (original topology, excluding deleted atoms)
        for bond in reactants.bonds:
            rids = [ep.get("react_id") for ep in bond.endpoints]
            _add_topology_item(rids, bond.data, "bond")

        for angle in reactants.angles:
            rids = [ep.get("react_id") for ep in angle.endpoints]
            _add_topology_item(rids, angle.data, "angle")

        for dihedral in reactants.dihedrals:
            rids = [ep.get("react_id") for ep in dihedral.endpoints]
            _add_topology_item(rids, dihedral.data, "dihedral")

        # Step 2: Get NEW topology from PRODUCT (new bonds/angles/dihedrals from reaction)
        # These are bonds/angles/dihedrals that don't exist in reactants
        for bond in product.bonds:
            rids = [ep.get("react_id") for ep in bond.endpoints]
            _add_topology_item(rids, bond.data, "bond")

        for angle in product.angles:
            rids = [ep.get("react_id") for ep in angle.endpoints]
            _add_topology_item(rids, angle.data, "angle")

        for dihedral in product.dihedrals:
            rids = [ep.get("react_id") for ep in dihedral.endpoints]
            _add_topology_item(rids, dihedral.data, "dihedral")

        return post, post_react_id_to_atom


def write_template_files(
    base_path: Path, template: TemplateResult, typifier=None
) -> None:
    """
    Write LAMMPS fix bond/react template files.

    Args:
        base_path: Base path for files (e.g., "rxn1" -> rxn1_pre.mol, rxn1_post.mol, rxn1.map)
        template: TemplateResult from TemplateReacter
        typifier: Optional typifier to ensure types are set
    """
    pre_path = Path(f"{base_path}_pre.mol")
    post_path = Path(f"{base_path}_post.mol")
    map_path = Path(f"{base_path}.map")

    pre = template.pre
    post = template.post

    # Note: Topology and typification should already be done in TemplateReacter._generate_template
    # This is just a safety check for backward compatibility
    if typifier:
        _ensure_typified(pre, typifier)
        _ensure_typified(post, typifier)

    # Build react_id to index mappings and set atom IDs
    pre_react_id_to_idx = {}
    for i, atom in enumerate(pre.atoms, start=1):
        if "react_id" not in atom.data:
            raise ValueError(f"Atom {i} in pre missing react_id!")
        pre_react_id_to_idx[atom["react_id"]] = i
        atom["id"] = i

    post_react_id_to_idx = {}
    for i, atom in enumerate(post.atoms, start=1):
        if "react_id" not in atom.data:
            raise ValueError(f"Atom {i} in post missing react_id!")
        post_react_id_to_idx[atom["react_id"]] = i
        atom["id"] = i

    # Verify pre and post have same atoms
    pre_react_ids = set(pre_react_id_to_idx.keys())
    post_react_ids = set(post_react_id_to_idx.keys())

    if pre_react_ids != post_react_ids:
        raise ValueError(
            f"Pre and post have different atoms!\n"
            f"  Missing in post: {pre_react_ids - post_react_ids}\n"
            f"  Missing in pre: {post_react_ids - pre_react_ids}"
        )

    # Build equivalences
    equiv = [
        (pre_react_id_to_idx[rid], post_react_id_to_idx[rid]) for rid in pre_react_ids
    ]

    # Get special atom indices
    initiator_ids = []
    for a in template.init_atoms:
        rid = a.get("react_id")
        if rid and rid in pre_react_id_to_idx:
            initiator_ids.append(pre_react_id_to_idx[rid])

    # Edge atoms: atoms in subgraph that have neighbors outside subgraph
    # Exclude initiator atoms from edge atoms (they are reaction sites, not boundaries)
    edge_ids = []
    initiator_rids = {a.get("react_id") for a in template.init_atoms}
    for a in template.edge_atoms:
        rid = a.get("react_id")
        # Don't mark initiator atoms as edge atoms
        if rid and rid in pre_react_id_to_idx and rid not in initiator_rids:
            edge_ids.append(pre_react_id_to_idx[rid])

    deleted_ids = []
    for a in template.removed_atoms:
        rid = a.get("react_id")
        # Mark all removed atoms for deletion, even if they are also initiators.
        # This is necessary for reactions like dehydration where:
        # - Initiator atoms are port atoms (two O atoms)
        # - Anchor atoms are the actual reaction sites (left: C neighbor, right: O port)
        # - Left port O is both an initiator AND part of the leaving group (O+H)
        # In such cases, the port O must be deleted even though it's an initiator
        if rid and rid in pre_react_id_to_idx:
            deleted_ids.append(pre_react_id_to_idx[rid])

    # Write map file
    with map_path.open("w", encoding="utf-8") as f:
        f.write("# auto-generated map file for fix bond/react\n\n")
        f.write(f"{len(equiv)} equivalences\n")
        f.write(f"{len(edge_ids)} edgeIDs\n")
        f.write(f"{len(deleted_ids)} deleteIDs\n\n")
        f.write("InitiatorIDs\n\n")
        for idx in initiator_ids:
            f.write(f"{idx}\n")
        f.write("\nEdgeIDs\n\n")
        for idx in edge_ids:
            f.write(f"{idx}\n")
        f.write("\nDeleteIDs\n\n")
        for idx in deleted_ids:
            f.write(f"{idx}\n")
        f.write("\nEquivalences\n\n")
        for pre_id, post_id in sorted(equiv):
            f.write(f"{pre_id}   {post_id}\n")

    # Write mol files with UNIFIED type mappings
    # CRITICAL: Pre and post must use the same type-to-ID mapping so LAMMPS
    # sees consistent types. Otherwise edge atoms may appear to have type changes.
    pre_frame = pre.to_frame()
    post_frame = post.to_frame()

    if "charge" in pre_frame["atoms"]:
        pre_frame["atoms"]["q"] = pre_frame["atoms"]["charge"]
    if "charge" in post_frame["atoms"]:
        post_frame["atoms"]["q"] = post_frame["atoms"]["charge"]

    # Create unified type mappings from BOTH pre and post
    _unify_type_mappings(pre_frame, post_frame)

    # Write with pre-converted types (won't trigger _convert_types_to_ids again)
    mp.io.write_lammps_molecule(pre_path, pre_frame)
    mp.io.write_lammps_molecule(post_path, post_frame)

    print(f"✅ Written: {pre_path.name}, {post_path.name}, {map_path.name}")


def _unify_type_mappings(pre_frame, post_frame) -> None:
    """
    Create unified type-to-ID mappings for both pre and post frames.

    LAMMPS fix bond/react requires that atoms with the same type name have
    the same numeric type ID in both pre and post templates. This function
    collects all unique types from both frames and applies a shared mapping.

    Args:
        pre_frame: Frame for pre-reaction template (modified in-place)
        post_frame: Frame for post-reaction template (modified in-place)
    """
    import numpy as np

    # Collect all unique atom types from both frames
    all_atom_types = set()
    for frame in [pre_frame, post_frame]:
        if "atoms" in frame and "type" in frame["atoms"]:
            for t in frame["atoms"]["type"]:
                all_atom_types.add(str(t))

    # Create unified atom type mapping
    atom_type_to_id = {t: i + 1 for i, t in enumerate(sorted(all_atom_types))}

    # Apply to both frames - create new integer array
    for frame in [pre_frame, post_frame]:
        if "atoms" in frame and "type" in frame["atoms"]:
            atoms = frame["atoms"]
            new_types = np.array(
                [
                    atom_type_to_id[str(atoms["type"][idx])]
                    for idx in range(atoms.nrows)
                ],
                dtype=np.int64,
            )
            atoms["type"] = new_types

    # Process connectivity types (bonds, angles, dihedrals, impropers)
    for section in ["bonds", "angles", "dihedrals", "impropers"]:
        # Collect all unique types from both frames for this section
        all_types = set()
        for frame in [pre_frame, post_frame]:
            if section in frame and frame[section].nrows > 0:
                block = frame[section]
                if "type" in block:
                    for t in block["type"]:
                        all_types.add(str(t))

        if not all_types:
            continue

        # Create unified mapping
        type_to_id = {t: i + 1 for i, t in enumerate(sorted(all_types))}

        # Apply to both frames - create new integer array
        for frame in [pre_frame, post_frame]:
            if section in frame and frame[section].nrows > 0:
                block = frame[section]
                if "type" in block:
                    new_types = np.array(
                        [
                            type_to_id[str(block["type"][idx])]
                            for idx in range(block.nrows)
                        ],
                        dtype=np.int64,
                    )
                    block["type"] = new_types


def _ensure_typified(struct: Atomistic, typifier) -> None:
    """Ensure all atoms and topology in struct are typified."""
    typifier.atom_typifier.typify(struct)
    for bond in struct.bonds:
        typifier.bond_typifier.typify(bond)
    for angle in struct.angles:
        typifier.angle_typifier.typify(angle)
    for dihedral in struct.dihedrals:
        typifier.dihedral_typifier.typify(dihedral)
