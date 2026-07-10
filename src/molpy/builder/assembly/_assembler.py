"""The one assembly kernel: paste, match, pair, edit locally, repair locally.

Growing a chain, crosslinking a melt and closing a macrocycle differ only in
which sites pair up. That difference is a :class:`~molpy.builder.assembly._selector.Selector`
handed to :meth:`GraphAssembler.assemble`; everything else is this one code path.

Nothing here walks the whole system per edit. Adding the thousandth bond costs
what the second one cost, because the only atoms examined are the ones inside the
ball the typifier's :class:`~molpy.typifier.scope.TypeScope` calls for.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import molrs
from molpy.builder.assembly._context import MatchContext
from molpy.builder.assembly._selector import Binding, Selector
from molpy.core import fields
from molpy.core.atomistic import Atomistic
from molpy.typifier.region import RegionTypifier

if TYPE_CHECKING:
    from molrs.fields import FieldSpec

    from molpy.builder.assembly._placer import Placer
    from molpy.core.affected_region import AffectedRegion

#: Net charge drift that counts as zero (elementary charge).
_CHARGE_TOL = 1e-9


class GraphAssembler:
    """Apply a reaction wherever a selector says, then repair types locally.

    Use it directly to crosslink an existing graph; subclass it to add an input
    format, as :class:`~molpy.builder.assembly._polymer.PolymerBuilder` does for
    CGSmiles + a monomer library.

    ``typifier=None`` is a named mode — *pure-topology assembly* — not a
    fallback: it skips retyping and nothing else. A typifier that declares no
    receptive field is rejected at construction, because there is no honest way
    to type a region with it.
    """

    def __init__(
        self,
        reaction: molrs.Reaction,
        *,
        typifier: RegionTypifier | None = None,
        placer: Placer | None = None,
        label_field: FieldSpec = fields.SITE,
    ) -> None:
        if not isinstance(reaction, molrs.Reaction):
            raise TypeError(
                "reaction must be a molpy.Reaction instance, not "
                f"{type(reaction).__name__}; build it once with mp.Reaction(smirks)"
            )
        if typifier is not None and not isinstance(typifier, RegionTypifier):
            raise TypeError(
                f"{type(typifier).__name__} declares no receptive field (scope) and "
                "cannot type a region. There is no whole-graph fallback."
            )
        self._reaction = reaction
        self._typifier = typifier
        self._placer = placer
        self._label_field = label_field

        forming = reaction.forming_bonds
        if not forming:
            raise ValueError("reaction forms no bond; nothing to assemble")
        self._map_a, self._map_b = forming[0]
        label_sets = self._component_label_sets()
        self._comp_a = self._find_component(label_sets, self._map_a)
        self._comp_b = self._find_component(label_sets, self._map_b)

        #: Memoised across ``assemble`` calls: it keys on the typifier and the
        #: region's structure, never on the selector, so one builder can grow a
        #: hundred chains of different lengths and type each junction once.
        self._cache = self._new_cache()

    # -- the verb ------------------------------------------------------------

    def assemble(self, world: Atomistic, selector: Selector) -> Atomistic:
        """Return a new graph with the selector's bindings reacted.

        ``world`` is never mutated.
        """
        work = world.copy()
        labels = self._labels(work)
        occurrences = self._match(work, labels)
        context = MatchContext(
            world=work,
            occurrences=occurrences,
            map_a=self._map_a,
            map_b=self._map_b,
            comp_a=self._comp_a,
            comp_b=self._comp_b,
        )

        bindings = list(selector.select(context))
        if not bindings:
            warnings.warn(
                f"{type(selector).__name__} selected no bindings from "
                f"{sum(len(o) for o in occurrences)} matched sites; "
                "the world is returned unchanged",
                stacklevel=2,
            )
            return work
        self._assert_disjoint(bindings)

        if self._placer is not None:
            self._placer.place(work, bindings)

        charge_before = self._total_charge(work)
        formed: list[tuple[int, int]] = []
        touched_sets: list[list[int]] = []
        for binding in bindings:
            # refresh=False: skip molrs' per-apply whole-graph angle/dihedral +
            # aromaticity rebuild (O(N) each). Matching and region extraction
            # need only bonds, which update in place.
            touched = self._reaction.apply(work, binding, labels, refresh=False)
            self._assert_touched_covers_forming_bond(binding, touched)
            formed.append((binding[self._map_a], binding[self._map_b]))
            touched_sets.append(list(touched))
        self._assert_charge_conserved(work, charge_before)

        if self._typifier is None:
            # Pure-topology mode: nothing declares a region radius, so the new
            # bonded terms are perceived with one whole-graph pass.
            work.generate_topology(gen_angle=True, gen_dihedral=True)
            molrs.perceive_aromaticity(work)
            return work

        # Regions are built AFTER every apply, so each one sees the final graph.
        # Two overlapping regions then agree on a shared interior atom's type.
        scope = self._typifier.scope
        regions = [scope.region(work, touched) for touched in touched_sets]
        inserted: set[tuple[str, tuple[int, ...]]] = set()
        for region, bond in zip(regions, formed, strict=True):
            self._cache.retype_and_apply(region)
            self._insert_new_terms(work, region, bond, inserted)
        return work

    # -- construction helpers ------------------------------------------------

    def _new_cache(self):
        from molpy.typifier.cache import RetypeCache

        return RetypeCache(self._typifier) if self._typifier is not None else None

    def _component_label_sets(self) -> list[set[int]]:
        return [
            {
                label
                for i in range(pattern.num_query_atoms)
                if (label := pattern.map_label(i)) is not None
            }
            for pattern in self._reaction.reactant_patterns
        ]

    @staticmethod
    def _find_component(label_sets: list[set[int]], map_number: int) -> int:
        """Which reactant component carries ``map_number``.

        A forming-bond map number that appears in no reactant pattern is a
        malformed reaction. Returning component 0 would pair the wrong occurrence
        lists and produce a chemically meaningless bond, silently.
        """
        for index, labels in enumerate(label_sets):
            if map_number in labels:
                return index
        raise ValueError(
            f"forming-bond map number {map_number} appears in no reactant pattern "
            f"of this reaction (patterns carry {sorted(set().union(*label_sets))})"
        )

    # -- per-assemble helpers ------------------------------------------------

    def _labels(self, graph: Atomistic) -> dict[int, str]:
        """``{handle: label}`` for the ``%LABEL`` predicates, from ``label_field``.

        Only atoms carrying a non-empty label enter the table: an empty string
        means *unmarked* and must not match a ``%site`` predicate.
        """
        key = self._label_field.key
        return {
            atom.handle: str(value) for atom in graph.atoms if (value := atom.get(key))
        }

    def _match(self, graph: Atomistic, labels: dict[int, str]) -> list[list[Binding]]:
        """Match each reactant pattern once. O(N) — the selector never rescans."""
        return [
            pattern.find_matches(graph, labels=labels, mapped=True)
            for pattern in self._reaction.reactant_patterns
        ]

    @staticmethod
    def _assert_disjoint(bindings: list[Binding]) -> None:
        seen: dict[int, int] = {}
        for index, binding in enumerate(bindings):
            for handle in binding.values():
                if handle in seen:
                    raise ValueError(
                        f"bindings {seen[handle]} and {index} both name atom "
                        f"{handle}; every edit is applied to the same world, so "
                        "the second would act on handles the first invalidated"
                    )
                seen[handle] = index

    def _assert_touched_covers_forming_bond(
        self, binding: Binding, touched: list[int]
    ) -> None:
        """``ball(touched, reach)`` is only complete if the new bond's ends are in it."""
        expected = {binding[self._map_a], binding[self._map_b]}
        missing = expected - set(touched)
        if missing:
            raise RuntimeError(
                f"molrs.Reaction.apply omitted forming-bond endpoint(s) "
                f"{sorted(missing)} from its touched set; the affected region "
                "would be incomplete"
            )

    @staticmethod
    def _total_charge(graph: Atomistic) -> float | None:
        """Net charge, or ``None`` when the graph carries no charge column."""
        charges = [atom.get(fields.CHARGE) for atom in graph.atoms]
        if all(q is None for q in charges):
            return None
        return float(sum(q for q in charges if q is not None))

    def _assert_charge_conserved(self, graph: Atomistic, before: float | None) -> None:
        """Deleting a leaving group must not change the net charge.

        Charge is solved and frozen on the monomer template, where each cap's
        charge is folded onto the site atom it capped. After that fold every atom
        the reaction deletes carries exactly zero, so conservation is an
        accounting identity — not something a redistribution heuristic restores
        afterwards. A drift here means the templates were never frozen.
        """
        if before is None:
            return
        after = self._total_charge(graph)
        if after is None or abs(after - before) > _CHARGE_TOL:
            raise ValueError(
                f"assembly changed the net charge by {(after or 0.0) - before:+.6g} e: "
                "the reaction deleted atoms that carry charge. Freeze the monomer "
                "templates first so each cap's charge folds onto its site atom."
            )

    # -- new bonded terms ----------------------------------------------------

    def _insert_new_terms(
        self,
        world: Atomistic,
        region: AffectedRegion,
        bond: tuple[int, int],
        inserted: set[tuple[str, tuple[int, ...]]],
    ) -> None:
        """Create the angles/dihedrals the new bond brought into existence.

        Only terms that *contain* the formed bond are new: everything else either
        survived the edit untouched or was removed with a deleted atom (molrs does
        that even with ``refresh=False``). So this inserts and never deletes.

        Overlapping regions can both own a dihedral that spans two formed bonds;
        ``inserted`` makes the write idempotent.
        """
        parent_of = {
            region_atom.handle: parent_atom
            for region_atom, parent_atom in region.entity_map.items()
        }
        local_of = {parent.handle: handle for handle, parent in parent_of.items()}
        if bond[0] not in local_of or bond[1] not in local_of:
            return
        u, v = local_of[bond[0]], local_of[bond[1]]
        interior = {atom.handle for atom in region.interior}

        for kind, views, adder in (
            ("angle", region.angles, world.def_angle),
            ("dihedral", region.dihedrals, world.def_dihedral),
        ):
            for term in views:
                handles = [endpoint.handle for endpoint in term.endpoints]
                if not self._spans_bond(handles, u, v):
                    continue
                if any(handle not in interior for handle in handles):
                    continue
                parents = tuple(parent_of[handle] for handle in handles)
                key = (kind, tuple(p.handle for p in parents))
                if key in inserted or (kind, key[1][::-1]) in inserted:
                    continue
                inserted.add(key)
                adder(*parents)

    @staticmethod
    def _spans_bond(handles: list[int], u: int, v: int) -> bool:
        """True when ``(u, v)`` is one of the term's consecutive bonds."""
        return any(
            {handles[i], handles[i + 1]} == {u, v} for i in range(len(handles) - 1)
        )
