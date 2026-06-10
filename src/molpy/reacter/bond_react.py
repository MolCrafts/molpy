"""LAMMPS fix bond/react template generation.

Provides :class:`BondReactReacter`, a :class:`Reacter` subclass that
generates pre/post molecule templates and map files for the LAMMPS
``fix bond/react`` command.

See: https://docs.lammps.org/fix_bond_react.html
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

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
    """LAMMPS fix bond/react template data (pure data object).

    Contains pre/post reaction subgraphs and the atom mapping needed to
    serialize the files required by ``fix bond/react``:

    - ``{name}_pre.mol`` — pre-reaction molecule template
    - ``{name}_post.mol`` — post-reaction molecule template
    - ``{name}.map`` — atom equivalence / edge / delete map

    Serialization lives in the io layer: write a complete system with
    :func:`molpy.io.write_lammps_bond_react_system`, or just the map
    file with :func:`molpy.io.write_bond_react_map`.

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
    initiator_atoms: list[Atom]
    edge_atoms: list[Atom]
    deleted_atoms: list[Atom]
    pre_react_id_to_atom: dict
    post_react_id_to_atom: dict

    def assign_atom_ids(self) -> None:
        """Assign deterministic 1-based ``id`` values to pre/post atoms.

        Iterates ``self.pre.atoms`` and ``self.post.atoms`` in their
        stable insertion order and assigns ``atom["id"] = i`` (1-based).
        This ordering convention defines the template-local indices used
        by the ``.map`` file, so io writers call this before serializing
        templates.
        """
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
        mp.io.write_lammps_bond_react_system(
            "output", frame, forcefield, templates={"rxn1": result.template}
        )
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

        # Note: do NOT re-run the typifier here.  Atom types are
        # already set by _incremental_typify (product) and deep-copied
        # by _build_post (post).  Re-typing can reorder atoms inside
        # the Atomistic, which breaks the equivalence map.

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
