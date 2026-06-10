"""LAMMPS fix bond/react template generation.

Provides :class:`BondReactReacter`, a :class:`Reacter` subclass that
generates pre/post molecule templates and map files for the LAMMPS
``fix bond/react`` command.

See: https://docs.lammps.org/fix_bond_react.html
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from molpy.core.atomistic import Atom, Atomistic, Bond
from molpy.core.entity import Entity
from molpy.reacter.base import Reacter, ReactionResult
from molpy.reacter.utils import AnchorSelector, BondFormer, LeavingSelector

if TYPE_CHECKING:
    from molpy.typifier.atomistic import TypifierBase

logger = logging.getLogger(__name__)

#: Tolerance for the pre/post total-charge conservation check, in
#: elementary charge units (e). Fixed point-charge force fields assign
#: charges per type, so a reaction template never redistributes total
#: charge; a difference beyond this tolerance indicates an inconsistent
#: template and is logged as a warning (not raised).
CHARGE_CONSERVATION_TOL: float = 1e-6


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


def validate_bond_react_template(template: BondReactTemplate) -> None:
    """Validate REACTER map invariants on a bond/react template.

    Enforces the template requirements of LAMMPS ``fix bond/react``
    (https://docs.lammps.org/fix_bond_react.html) and the REACTER
    protocol (Gissinger et al., Polymer 128 (2017) 211-217,
    DOI: 10.1016/j.polymer.2017.06.038; Gissinger et al.,
    Macromolecules 53 (2020) 9953-9961,
    DOI: 10.1021/acs.macromol.0c02012):

    - exactly 2 initiator atoms (the bond-forming anchors);
    - no initiator may sit on the template boundary (edge atom) — its
      first neighbor shell must be inside the template;
    - edge atoms must be identical pre vs post in ``type`` and
      ``charge`` (fix bond/react rejects maps otherwise);
    - total charge (elementary charge units, e) should be conserved
      within :data:`CHARGE_CONSERVATION_TOL`; violations are logged as
      a warning, not raised, since fixed point-charge force fields
      assign charge per type and templates never redistribute it.

    Args:
        template: The template to validate.

    Raises:
        ValueError: On initiator-count, initiator-on-edge, or edge
            pre/post mismatch violations. The message names the
            offending atom and suggests increasing ``radius``.
    """
    initiators = template.initiator_atoms
    if len(initiators) != 2:
        names = [
            f"react_id={a.get('react_id')}, element={a.get('element')}"
            for a in initiators
        ]
        raise ValueError(
            f"fix bond/react requires exactly 2 initiator atoms, got "
            f"{len(initiators)} ({names}). An anchor may have fallen outside "
            f"the template radius; increase the reacter's radius."
        )

    edge_rids = {a.get("react_id") for a in template.edge_atoms}
    for anchor in initiators:
        rid = anchor.get("react_id")
        if rid in edge_rids:
            raise ValueError(
                f"Initiator atom (react_id={rid}, "
                f"element={anchor.get('element')}) lies on the template "
                f"boundary (edge atom), so its first neighbor shell is not "
                f"contained in the template; increase the reacter's radius."
            )

    for edge_atom in template.edge_atoms:
        rid = edge_atom.get("react_id")
        pre_atom = template.pre_react_id_to_atom.get(rid)
        post_atom = template.post_react_id_to_atom.get(rid)
        if pre_atom is None or post_atom is None:
            # Pre/post atom-set mismatches are reported by the map writer.
            continue
        for field_name in ("type", "charge"):
            pre_value = pre_atom.get(field_name)
            post_value = post_atom.get(field_name)
            if pre_value != post_value:
                raise ValueError(
                    f"Edge atom (react_id={rid}, "
                    f"element={pre_atom.get('element')}) differs between pre "
                    f"and post in {field_name!r}: pre={pre_value!r}, "
                    f"post={post_value!r}. fix bond/react requires edge atoms "
                    f"to be identical; increase the reacter's radius so the "
                    f"retyped shell is inside the template."
                )

    def _total_charge(atoms: list[Atom]) -> tuple[float, int]:
        total = 0.0
        missing = 0
        for atom in atoms:
            charge = atom.get("charge")
            if charge is None:
                missing += 1
            else:
                total += float(charge)
        return total, missing

    pre_q, pre_missing = _total_charge(list(template.pre.atoms))
    post_q, post_missing = _total_charge(list(template.post.atoms))
    delta = post_q - pre_q
    if abs(delta) > CHARGE_CONSERVATION_TOL:
        missing_note = ""
        if pre_missing or post_missing:
            missing_note = (
                f" (atoms without a 'charge' field treated as 0: "
                f"{pre_missing} pre, {post_missing} post)"
            )
        logger.warning(
            "Charge not conserved in bond/react template: "
            "sum(q_post) - sum(q_pre) = %.6g e exceeds tolerance %.1g e%s.",
            delta,
            CHARGE_CONSERVATION_TOL,
            missing_note,
        )


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
       (stamped on internal copies only — caller-owned inputs are never
       mutated)
    2. Subgraph extraction around the reaction site
    3. Pre/post template generation attached to :attr:`ReactionResult.template`

    The ``radius`` parameter controls how many bonds away from the anchor
    atoms the extracted subgraph extends.

    Generated templates are validated against the REACTER protocol
    (Gissinger et al., Polymer 128 (2017) 211-217,
    DOI: 10.1016/j.polymer.2017.06.038; Gissinger et al., Macromolecules
    53 (2020) 9953-9961, DOI: 10.1021/acs.macromol.0c02012) and the
    LAMMPS ``fix bond/react`` contract
    (https://docs.lammps.org/fix_bond_react.html):

    - Equivalences are a pre→post bijection over template atoms;
    - exactly 2 ordered initiator atoms (left anchor first), never on
      the template boundary;
    - edge atoms identical pre vs post in type and charge (elementary
      charge units, e);
    - impropers propagate into the post template so sp2 centers keep
      their planarity terms;
    - total charge conserved within :data:`CHARGE_CONSERVATION_TOL`
      (warning on violation).

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

        Caller-owned ``left``/``right`` are never mutated: ``react_id``
        markers are stamped on internal copies only (port atoms are
        resolved into the copies positionally, mirroring
        ``Reacter._prepare_reactants``).

        Returns :class:`BondReactResult` with ``template`` populated.
        """
        # 1. Copy caller inputs and resolve port atoms into the copies,
        #    so react_id stamping never touches caller-owned structures.
        is_ring_closure = left is right
        left_work = left.copy()
        left_map: dict[Entity, Entity] = dict(
            zip(list(left.atoms), list(left_work.atoms))
        )
        if is_ring_closure:
            right_work = left_work
            right_map = left_map
        else:
            right_work = right.copy()
            right_map = dict(zip(list(right.atoms), list(right_work.atoms)))

        port_atom_L = left_map.get(port_atom_L, port_atom_L)
        port_atom_R = right_map.get(port_atom_R, port_atom_R)

        # 2. Assign react_ids on the working copies, then run the base
        #    reaction (its internal copy deep-copies data, so react_id
        #    survives into reactants/product).
        self._assign_react_ids(left_work)
        self._assign_react_ids(right_work)

        base_result = super().run(
            left_work,
            right_work,
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

        template = BondReactTemplate(
            pre=pre,
            post=post,
            initiator_atoms=initiator_atoms,
            edge_atoms=list(pre_edge_entities),
            deleted_atoms=removed_atoms,
            pre_react_id_to_atom=pre_react_id_to_atom,
            post_react_id_to_atom=post_react_id_to_atom,
        )
        validate_bond_react_template(template)
        return template

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
        """Build the post-reaction template in pre atom order.

        Atoms are emitted in ``pre_react_ids_ordered`` order so the
        pre→post equivalence map is positional, as required by the
        REACTER protocol (Gissinger et al., Polymer 128 (2017) 211-217,
        DOI: 10.1016/j.polymer.2017.06.038; Gissinger et al.,
        Macromolecules 53 (2020) 9953-9961,
        DOI: 10.1021/acs.macromol.0c02012). Bonds, angles, dihedrals,
        AND impropers are copied from both ``reactants`` (topology
        untouched by the reaction) and ``product`` (topology rebuilt by
        TopologyDetector) — post-side impropers are required because
        LAMMPS ``fix bond/react``
        (https://docs.lammps.org/fix_bond_react.html) replaces pre
        topology with post topology, so any improper missing here would
        be physically deleted from the simulation. Topology whose
        endpoints include deleted atoms is skipped (deleted atoms stay
        in the template but carry no post topology).

        Args:
            pre_react_ids_ordered: react_ids in pre-template atom order.
            pre_react_ids: Set view of the same ids.
            product: Post-reaction assembly.
            removed_atoms: Atoms deleted by the reaction.
            removed_react_ids: Their react_ids.
            pre: The pre template.
            reactants: Reactant snapshot (defaults to ``pre``).

        Returns:
            Tuple of (post template, react_id → post atom mapping).
        """
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
            elif item_type == "improper":
                # Impropers compare by center (i position) + unordered wings
                if not any(
                    imp.endpoints[0] is post_eps[0]
                    and set(imp.endpoints[1:]) == set(post_eps[1:])
                    for imp in post.impropers
                ):
                    post.def_improper(*post_eps, **data)
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
        for improper in reactants.impropers:
            _add_topo(
                [ep.get("react_id") for ep in improper.endpoints],
                improper.data,
                "improper",
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
        for improper in product.impropers:
            _add_topo(
                [ep.get("react_id") for ep in improper.endpoints],
                improper.data,
                "improper",
            )

        return post, post_react_id_to_atom
