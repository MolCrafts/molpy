"""A hashable MolGraph subgraph produced by a graph edit.

:class:`AffectedRegion` is the radius-N ball around the atoms a graph edit
touched, extracted (via :meth:`Atomistic._extract_mapped`) at a retype-safe
radius. It **is** an :class:`~molpy.core.atomistic.Atomistic` — so it hands
straight to third-party typifiers / AmberTools — but adds region semantics
(``interior`` / ``boundary`` / ``entity_map``) and an isomorphism-invariant
structural ``__hash__`` / ``__eq__`` (via the molrs Weisfeiler–Lehman graph
hash), so identical polymer junctions dedupe to one cache key.

The region overrides hashing only at the *region* level; its member
:class:`~molpy.core.entity.Entity` / :class:`~molpy.core.entity.Link` views keep
their identity hashing (unchanged core contract).

Producers (``Reacter``, ``Crosslinker``) build a region from the atoms an edit
reports as touched. Region-scoped typing + the retype cache keyed by
``AffectedRegion.__hash__`` are ``incremental-typify-02`` (out of scope here).

The generic :class:`_RegionMixin` is shared with an eventual coarse-grained
variant; only the all-atom :class:`AffectedRegion` is instantiated here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Self

from molpy.core.atomistic import Atom, Atomistic
from molpy.core.entity import Entity

#: Retype-safe floor radius — BondReact's proven default; the extracted ball
#: must cover the typifier's SMARTS depth so boundary atoms have full context.
_FLOOR = 4


def region_radius(typifier: object | None = None) -> int:
    """Retype-safe extraction radius for an :class:`AffectedRegion`.

    ``max(typifier.context_radius, _FLOOR)`` with ``_FLOOR = 4``. A typifier may
    expose ``context_radius`` (the max path length over its SMARTS patterns);
    when it is absent — or ``typifier`` is ``None`` — the floor applies. Callers
    may override the radius per edit.
    """
    ctx = getattr(typifier, "context_radius", 0)
    return max(int(ctx), _FLOOR)


class _RegionMixin[E: Entity]:
    """Region semantics layered onto a molpy graph (all-atom or coarse-grained).

    Carries the changed ``interior`` atoms (to be retyped), the context-only
    ``boundary`` shell (never retyped), and the region → parent ``entity_map``
    (to map assigned types back). The concrete subclass supplies the structural
    ``__hash__`` / ``__eq__`` over the molrs graph hash. Populated by
    :meth:`_from`, which is the region's only constructor.
    """

    #: the changed atoms an edit touched — the ones to be retyped.
    interior: tuple[E, ...]
    #: the context-only shell (each has a neighbour outside the ball).
    boundary: tuple[E, ...]
    #: region entity → parent-graph entity (invert to map types back).
    entity_map: dict[E, E]


class AffectedRegion(_RegionMixin[Atom], Atomistic):
    """All-atom affected region: a hashable, typable, AmberTools-ready subgraph.

    Base order mirrors :class:`~molpy.core.atomistic.Atomistic`: the pyo3 native
    ``molrs.Atomistic`` remains the solid base (contributed by ``Atomistic``),
    with the plain-Python :class:`_RegionMixin` layered in front for its region
    attributes.
    """

    @classmethod
    def _from(
        cls, parent: Atomistic, touched: Iterable[Atom | int], radius: int
    ) -> Self:
        """Build the region around ``touched`` in ``parent`` at ``radius``.

        ``touched`` are the seed atoms an edit reported — :class:`Atom` views or
        raw molrs handles (as returned by ``molrs.Reaction.apply``). ``parent``
        is not mutated: the region is an induced clone with its own atom views.
        """
        centers = _resolve_centers(parent, touched)
        sub, boundary, region_to_parent = parent._extract_mapped(centers, radius, cls)
        parent_to_region = {
            parent_atom: region_atom
            for region_atom, parent_atom in region_to_parent.items()
        }
        sub.interior = tuple(
            parent_to_region[c] for c in centers if c in parent_to_region
        )
        sub.boundary = tuple(boundary)
        sub.entity_map = region_to_parent
        return sub

    def __hash__(self) -> int:
        # Isomorphism-invariant molrs Weisfeiler–Lehman hash — the dedup key.
        return self.structural_hash()

    def __eq__(self, other: object) -> bool:
        # Graph equality (resolves the rare hash collision); only regions of the
        # same kind compare — a region never equals a plain Atomistic.
        return isinstance(other, AffectedRegion) and self.is_isomorphic(other)


def _resolve_centers(parent: Atomistic, touched: Iterable[Atom | int]) -> list[Atom]:
    """Normalise ``touched`` (atoms or handles) to deduplicated parent atoms."""
    seen: set[int] = set()
    centers: list[Atom] = []
    for item in touched:
        if isinstance(item, Atom):
            atom = item
        else:
            view = parent._intern_atom(item)
            assert isinstance(view, Atom), "handle did not intern to an Atom"
            atom = view
        if atom.handle not in seen:
            seen.add(atom.handle)
            centers.append(atom)
    return centers
