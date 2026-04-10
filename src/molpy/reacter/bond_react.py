"""LAMMPS fix bond/react template generation.

Provides :class:`BondReactReacter`, a :class:`Reacter` subclass that
generates pre/post molecule templates and map files for the LAMMPS
``fix bond/react`` command.

See: https://docs.lammps.org/fix_bond_react.html
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np

import molpy as mp
from molpy.core.atomistic import Atom, Atomistic, Bond
from molpy.core.entity import Entity
from molpy.reacter.base import Reacter, ReactionResult
from molpy.reacter.utils import AnchorSelector, BondFormer, LeavingSelector

if TYPE_CHECKING:
    from molpy.typifier.atomistic import TypifierBase


# ===================================================================
#                       BondReactTemplate
# ===================================================================


@dataclass
class BondReactTemplate:
    """LAMMPS fix bond/react template data.

    Contains pre/post reaction subgraphs and atom mapping needed to
    generate the three files required by ``fix bond/react``:

    - ``{name}_pre.mol`` — pre-reaction molecule template
    - ``{name}_post.mol`` — post-reaction molecule template
    - ``{name}.map`` — atom equivalence / edge / delete map

    Attributes:
        pre: Pre-reaction subgraph (deep copy of local environment).
        post: Post-reaction subgraph (same atoms, new topology).
        initiator_atoms: The pair of atoms that trigger the reaction
            (LAMMPS ``InitiatorIDs``).
        edge_atoms: Boundary atoms connected to topology outside the
            template (LAMMPS ``EdgeIDs``).
        deleted_atoms: Atoms removed by the reaction
            (LAMMPS ``DeleteIDs``).
        pre_react_id_to_atom: Mapping from react_id to atom in pre.
        post_react_id_to_atom: Mapping from react_id to atom in post.
    """

    pre: Atomistic
    post: Atomistic
    initiator_atoms: List[Atom]
    edge_atoms: List[Atom]
    deleted_atoms: List[Atom]
    pre_react_id_to_atom: dict
    post_react_id_to_atom: dict

    def write(
        self, base_path: str | Path, typifier: TypifierBase | None = None
    ) -> None:
        """Write pre.mol, post.mol, and .map files for ``fix bond/react``.

        This is a convenience method for standalone use. When writing a
        complete fix bond/react system (data + ff + templates), prefer
        :func:`molpy.io.write_lammps_bond_react_system` which ensures
        type IDs are consistent across all files.

        Args:
            base_path: Stem for output files (e.g. ``"04_output/rxn1"``
                produces ``rxn1_pre.mol``, ``rxn1_post.mol``, ``rxn1.map``).
            typifier: Optional typifier to ensure atom/bond types are set.
        """
        base_path = Path(base_path)

        pre = self.pre
        post = self.post

        if typifier:
            _ensure_typified(pre, typifier)
            _ensure_typified(post, typifier)

        self._assign_atom_ids()
        self.write_map(base_path)

        # Write .mol files with unified type mappings (pre/post only)
        pre_frame = pre.to_frame()
        post_frame = post.to_frame()
        _unify_type_mappings(pre_frame, post_frame)

        mp.io.write_lammps_molecule(Path(f"{base_path}_pre.mol"), pre_frame)
        mp.io.write_lammps_molecule(Path(f"{base_path}_post.mol"), post_frame)

    def write_map(self, base_path: str | Path) -> None:
        """Write only the .map file for ``fix bond/react``.

        The .map file contains atom equivalences, initiator IDs, edge IDs,
        and delete IDs. It is purely topological and independent of type
        numbering.

        Args:
            base_path: Stem path; produces ``{base_path}.map``.
        """
        self._assign_atom_ids()

        pre_rid_to_idx = {
            atom["react_id"]: i for i, atom in enumerate(self.pre.atoms, start=1)
        }
        post_rid_to_idx = {
            atom["react_id"]: i for i, atom in enumerate(self.post.atoms, start=1)
        }

        pre_rids = set(pre_rid_to_idx)
        post_rids = set(post_rid_to_idx)
        if pre_rids != post_rids:
            raise ValueError(
                f"Pre and post have different atoms!\n"
                f"  Missing in post: {pre_rids - post_rids}\n"
                f"  Missing in pre: {post_rids - pre_rids}"
            )

        equiv = [(pre_rid_to_idx[rid], post_rid_to_idx[rid]) for rid in pre_rids]

        initiator_rids = {a.get("react_id") for a in self.initiator_atoms}
        initiator_ids = [
            pre_rid_to_idx[rid]
            for rid in initiator_rids
            if rid and rid in pre_rid_to_idx
        ]
        edge_ids = [
            pre_rid_to_idx[a.get("react_id")]
            for a in self.edge_atoms
            if a.get("react_id")
            and a.get("react_id") in pre_rid_to_idx
            and a.get("react_id") not in initiator_rids
        ]
        deleted_ids = [
            pre_rid_to_idx[a.get("react_id")]
            for a in self.deleted_atoms
            if a.get("react_id") and a.get("react_id") in pre_rid_to_idx
        ]

        map_path = Path(f"{base_path}.map")
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

    def _assign_atom_ids(self) -> None:
        """Assign 1-based atom IDs to pre and post atoms."""
        for i, atom in enumerate(self.pre.atoms, start=1):
            atom["id"] = i
        for i, atom in enumerate(self.post.atoms, start=1):
            atom["id"] = i


# ===================================================================
#                       BondReactResult
# ===================================================================


@dataclass
class BondReactResult(ReactionResult):
    """Reaction result with an attached bond/react template.

    Extends :class:`ReactionResult` with a ``template`` field
    containing the :class:`BondReactTemplate` for LAMMPS export.
    """

    template: BondReactTemplate | None = None


# ===================================================================
#                       BondReactReacter
# ===================================================================


class BondReactReacter(Reacter):
    """Reacter subclass for LAMMPS ``fix bond/react`` template generation.

    Extends :class:`Reacter` with:

    1. ``react_id`` assignment for atom tracking across reactions
    2. Subgraph extraction around the reaction site
    3. Pre/post template generation attached to :attr:`ReactionResult.template`

    The ``radius`` parameter controls how many bonds away from the anchor
    atoms the extracted subgraph extends.

    Example::

        reacter = BondReactReacter(
            name="dehydration",
            anchor_selector_left=select_neighbor("C"),
            anchor_selector_right=select_self,
            leaving_selector_left=select_hydroxyl_group,
            leaving_selector_right=select_hydrogens(1),
            bond_former=form_single_bond,
            radius=3,
        )

        result = reacter.run(left, right, port_L, port_R, compute_topology=True)
        result.template.write("output/rxn1", typifier=typifier)
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
        super().__init__(
            name=name,
            anchor_selector_left=anchor_selector_left,
            anchor_selector_right=anchor_selector_right,
            leaving_selector_left=leaving_selector_left,
            leaving_selector_right=leaving_selector_right,
            bond_former=bond_former,
        )
        self.radius = radius
        self._react_id_counter = 0

    def run(
        self,
        left: Atomistic,
        right: Atomistic,
        port_atom_L: Entity,
        port_atom_R: Entity,
        compute_topology: bool = True,
        record_intermediates: bool = False,
        typifier: TypifierBase | None = None,
    ) -> BondReactResult:
        """Run reaction and generate bond/react template.

        Returns :class:`BondReactResult` with ``template`` populated.
        """
        # 1. Assign react_ids before reaction
        self._assign_react_ids(left)
        self._assign_react_ids(right)

        # 2. Run the base reaction
        base_result = super().run(
            left,
            right,
            port_atom_L,
            port_atom_R,
            compute_topology=compute_topology,
            record_intermediates=record_intermediates,
            typifier=typifier,
        )

        # 3. Generate template and wrap as BondReactResult
        template = self._generate_template(
            base_result, port_atom_L, port_atom_R, typifier
        )

        # Copy all base result fields into BondReactResult
        result_dict = {
            f.name: getattr(base_result, f.name) for f in fields(base_result)
        }
        return BondReactResult(**result_dict, template=template)

    # ── private helpers ──────────────────────────────────────────

    def _assign_react_ids(self, struct: Atomistic) -> None:
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
    ) -> BondReactTemplate:
        reactants = result.reactants
        product = result.product
        removed_atoms = result.removed_atoms

        port_L_rid = port_atom_L["react_id"]
        port_R_rid = port_atom_R["react_id"]

        # Find port atoms in reactants
        port_atoms_in_reactants = self._find_atoms_by_react_id(
            reactants, [port_L_rid, port_R_rid]
        )
        if len(port_atoms_in_reactants) != 2:
            raise ValueError(
                f"Could not find port atoms in reactants! "
                f"Looking for react_ids {port_L_rid}, {port_R_rid}"
            )

        # Use anchor atoms as subgraph centers
        anchor_L = self.anchor_selector_left(reactants, port_atoms_in_reactants[0])
        anchor_R = self.anchor_selector_right(reactants, port_atoms_in_reactants[1])

        # Extract pre subgraph
        pre, pre_edge_entities = reactants.extract_subgraph(
            center_entities=[anchor_L, anchor_R],
            radius=self.radius,
            entity_type=Atom,
            link_type=Bond,
        )

        # Note: do NOT call _ensure_typified() here.  Atom types are
        # already set by _incremental_typify (product) and deep-copied
        # by _build_post (post).  Re-running the typifier can reorder
        # atoms inside the Atomistic, which breaks the equivalence map.

        # Build pre react_id mapping
        pre_react_id_to_atom = {}
        for atom in pre.atoms:
            pre_react_id_to_atom[atom["react_id"]] = atom
        pre_react_ids = set(pre_react_id_to_atom)
        pre_react_ids_ordered = [atom["react_id"] for atom in pre.atoms]

        removed_react_ids = {a["react_id"] for a in removed_atoms}

        # Build post
        post, post_react_id_to_atom = self._build_post(
            pre_react_ids_ordered,
            pre_react_ids,
            product,
            removed_atoms,
            removed_react_ids,
            pre,
            reactants=reactants,
        )

        # Initiator atoms in pre — use anchors (the atoms that form the
        # new bond), not the port atoms (which may be in the leaving group).
        initiator_atoms = []
        for anchor in [anchor_L, anchor_R]:
            rid = anchor["react_id"]
            if rid in pre_react_id_to_atom:
                initiator_atoms.append(pre_react_id_to_atom[rid])

        return BondReactTemplate(
            pre=pre,
            post=post,
            initiator_atoms=initiator_atoms,
            edge_atoms=list(pre_edge_entities),
            deleted_atoms=removed_atoms,
            pre_react_id_to_atom=pre_react_id_to_atom,
            post_react_id_to_atom=post_react_id_to_atom,
        )

    def _find_atoms_by_react_id(
        self, struct: Atomistic, react_ids: list[int]
    ) -> list[Atom]:
        react_id_set = set(react_ids)
        return [a for a in struct.atoms if a.get("react_id") in react_id_set]

    def _build_post(
        self,
        pre_react_ids_ordered: list[int],
        pre_react_ids: set,
        product: Atomistic,
        removed_atoms: list[Atom],
        removed_react_ids: set,
        pre: Atomistic,
        reactants: Atomistic | None = None,
    ) -> tuple[Atomistic, dict]:
        if reactants is None:
            reactants = pre

        product_by_rid = {a["react_id"]: a for a in product.atoms}
        removed_by_rid = {a["react_id"]: a for a in removed_atoms}

        post_atoms = []
        post_react_id_to_atom = {}

        for rid in pre_react_ids_ordered:
            source = (
                removed_by_rid.get(rid)
                if rid in removed_react_ids
                else product_by_rid.get(rid)
            )
            if not source:
                raise ValueError(
                    f"react_id {rid} not found in product or removed_atoms!"
                )
            copied = Atom(deepcopy(source.data))
            post_atoms.append(copied)
            post_react_id_to_atom[rid] = copied

        post = Atomistic()
        post.add_atoms(post_atoms)

        def _add_topo(endpoints_rids, data, item_type):
            if not all(rid in pre_react_ids for rid in endpoints_rids):
                return False
            if any(rid in removed_react_ids for rid in endpoints_rids):
                return False
            post_eps = [post_react_id_to_atom[rid] for rid in endpoints_rids]
            if item_type == "bond":
                if not any(set(b.endpoints) == set(post_eps) for b in post.bonds):
                    post.def_bond(*post_eps, **data)
                    return True
            elif item_type == "angle":
                if not any(tuple(a.endpoints) == tuple(post_eps) for a in post.angles):
                    post.def_angle(*post_eps, **data)
                    return True
            elif item_type == "dihedral":
                if not any(
                    tuple(d.endpoints) == tuple(post_eps) for d in post.dihedrals
                ):
                    post.def_dihedral(*post_eps, **data)
                    return True
            return False

        for bond in reactants.bonds:
            _add_topo([ep.get("react_id") for ep in bond.endpoints], bond.data, "bond")
        for angle in reactants.angles:
            _add_topo(
                [ep.get("react_id") for ep in angle.endpoints], angle.data, "angle"
            )
        for dihedral in reactants.dihedrals:
            _add_topo(
                [ep.get("react_id") for ep in dihedral.endpoints],
                dihedral.data,
                "dihedral",
            )

        for bond in product.bonds:
            _add_topo([ep.get("react_id") for ep in bond.endpoints], bond.data, "bond")
        for angle in product.angles:
            _add_topo(
                [ep.get("react_id") for ep in angle.endpoints], angle.data, "angle"
            )
        for dihedral in product.dihedrals:
            _add_topo(
                [ep.get("react_id") for ep in dihedral.endpoints],
                dihedral.data,
                "dihedral",
            )

        return post, post_react_id_to_atom


# ===================================================================
#                       Internal helpers
# ===================================================================


def _unify_type_mappings(pre_frame, post_frame) -> None:
    """Create unified type-to-ID mappings for both pre and post frames."""
    all_atom_types = set()
    for frame in [pre_frame, post_frame]:
        if "atoms" in frame and "type" in frame["atoms"]:
            for t in frame["atoms"]["type"]:
                all_atom_types.add(str(t))

    atom_type_to_id = {t: i + 1 for i, t in enumerate(sorted(all_atom_types))}

    for frame in [pre_frame, post_frame]:
        if "atoms" in frame and "type" in frame["atoms"]:
            atoms = frame["atoms"]
            atoms["type"] = np.array(
                [
                    atom_type_to_id[str(atoms["type"][idx])]
                    for idx in range(atoms.nrows)
                ],
                dtype=np.int64,
            )

    for section in ["bonds", "angles", "dihedrals", "impropers"]:
        all_types = set()
        for frame in [pre_frame, post_frame]:
            if (
                section in frame
                and frame[section].nrows > 0
                and "type" in frame[section]
            ):
                for t in frame[section]["type"]:
                    all_types.add(str(t))
        if not all_types:
            continue
        type_to_id = {t: i + 1 for i, t in enumerate(sorted(all_types))}
        for frame in [pre_frame, post_frame]:
            if (
                section in frame
                and frame[section].nrows > 0
                and "type" in frame[section]
            ):
                block = frame[section]
                block["type"] = np.array(
                    [type_to_id[str(block["type"][idx])] for idx in range(block.nrows)],
                    dtype=np.int64,
                )


def _ensure_typified(struct: Atomistic, typifier) -> None:
    """Ensure all atoms and topology in struct are typified."""
    typifier.atom_typifier.typify(struct)
    for bond in struct.bonds:
        typifier.bond_typifier.typify(bond)
    for angle in struct.angles:
        typifier.angle_typifier.typify(angle)
    for dihedral in struct.dihedrals:
        typifier.dihedral_typifier.typify(dihedral)
