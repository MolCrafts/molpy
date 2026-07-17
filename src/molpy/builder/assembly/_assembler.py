"""The one assembly kernel: match, compile local products, then execute once.

Growing a chain, crosslinking a melt and closing a macrocycle differ only in
which sites pair up. That difference is a :class:`~molpy.builder.assembly._selector.Selector`
handed to :meth:`GraphAssembler.assemble`; everything else is this one code path.

Every prospective reaction environment is compiled while the user-defined
monomers are still intact.  The growing world is edited only once; cached typing
results write scalar per-atom data back afterwards.  Topology and bonded
parameter finalization are a separate, optional tail.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import molrs
from molpy.builder.assembly._context import MatchContext
from molpy.builder.assembly._finalize import (
    AssemblyFinalizer,
    Finalization,
)
from molpy.builder.assembly._plan import AssemblyCompiler
from molpy.builder.assembly._selector import Binding, Selector
from molpy.core import fields
from molpy.core.atomistic import Atomistic

if TYPE_CHECKING:
    from molrs.fields import FieldSpec

    from molpy.builder.assembly._placer import Placer
    from molpy.typifier.forcefield import ForceFieldParams

#: Net charge drift that counts as zero (elementary charge).
_CHARGE_TOL = 1e-9


class GraphAssembler:
    """Compile every selected local product, then apply the reaction once.

    Use it directly to crosslink an existing graph; subclass it to add an input
    format, as :class:`~molpy.builder.assembly._polymer.PolymerBuilder` does for
    CGSmiles + a monomer library.

    A typifier must be accompanied by the ``reach`` it needs.  ``finalize`` is
    orthogonal: ``"atoms"`` stops after per-atom write-back, ``"topology"``
    materializes angles/dihedrals once, and ``"bonded"`` also assigns their
    force-field parameters.
    """

    def __init__(
        self,
        reaction: molrs.Reaction,
        *,
        typifier: molrs.Typifier | None = None,
        reach: int | None = None,
        placer: Placer | None = None,
        label_field: FieldSpec = fields.SITE,
        finalize: Finalization | str = Finalization.TOPOLOGY,
        bonded: ForceFieldParams | None = None,
    ) -> None:
        if not isinstance(reaction, molrs.Reaction):
            raise TypeError(
                "reaction must be a molpy.Reaction instance, not "
                f"{type(reaction).__name__}; build it once with mp.Reaction(smirks)"
            )
        if typifier is not None and not isinstance(typifier, molrs.Typifier):
            raise TypeError(
                f"{type(typifier).__name__} is not a molrs.Typifier and "
                "cannot compile a local product. There is no whole-graph fallback."
            )
        if typifier is not None and reach is None:
            raise TypeError(
                "reach= is required alongside typifier=: it is the neighbourhood "
                "radius (in bonds) that decides one atom's type, and it fixes both "
                "the extracted ball and the write-back set. Pass reach=2 for GAFF "
                "and for SMARTS pattern sets without ring predicates."
            )
        if reach is not None and reach < 0:
            raise ValueError(f"reach must be >= 0, got {reach}")
        self._reaction = reaction
        self._typifier = typifier
        self._reach = reach
        self._placer = placer
        self._label_field = label_field
        self._finalizer = AssemblyFinalizer(Finalization(finalize), bonded)

        forming = reaction.forming_bonds
        if not forming:
            raise ValueError("reaction forms no bond; nothing to assemble")
        self._map_a, self._map_b = forming[0]
        label_sets = self._component_label_sets()
        self._comp_a = self._find_component(label_sets, self._map_a)
        self._comp_b = self._find_component(label_sets, self._map_b)

        self._compiler = AssemblyCompiler(reaction, typifier, reach)
        # Kept as a readable inspection point: the cache belongs to the
        # compiler and is shared across every build made by this assembler.
        self._cache = self._compiler.cache

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

        # This is the compile-first boundary: all local products and their
        # per-atom annotations are resolved before placement or graph mutation.
        plan = self._compiler.compile(work, bindings, labels)
        formed = [(b[self._map_a], b[self._map_b]) for b in plan.bindings]
        if self._placer is not None:
            self._placer.place(work, formed)

        charge_before = self._total_charge(work)
        # The reaction compiles every leaving group against the intact world,
        # then executes the disjoint transforms as one batch.  In particular,
        # relation tables are scanned once for the union of deleted atoms rather
        # than once per polymer bond.
        touched_sets, created_sets = self._reaction.apply_many_detailed(
            work, list(plan.bindings), labels, refresh=False
        )
        for binding, touched in zip(plan.bindings, touched_sets, strict=True):
            self._assert_touched_covers_forming_bond(binding, touched)
        self._assert_charge_conserved(work, charge_before)
        plan.write_atoms(work, created_sets)
        return self._finalizer.apply(work)

    # -- construction helpers ------------------------------------------------

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
