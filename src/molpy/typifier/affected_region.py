"""A hashable MolGraph subgraph produced by a graph edit.

This is not a data-model type — it is the ripple a graph edit raises. That is why
it lives here and not in :mod:`molpy.core`: nothing in the data model needs it,
and everything that does is a typifier.

A region retype has **two** radii, and they are not independent degrees of
freedom — they are ``reach`` and (essentially) ``2 * reach``. Let ``reach`` be the
neighbourhood radius, in bonds, that decides one atom's type.

*Write-back set.* An edit lands on ``touched``. Atom ``a``'s type can change iff
the edit falls inside ``a``'s deciding neighbourhood, i.e. ``touched`` intersects
``ball(a, reach)``. Distance is symmetric, so the atoms that must be retyped are
exactly ``ball(touched, reach)``.

*Extraction radius.* Every atom in that write-back set still needs its own
``reach``-ball in view. The farthest one sits ``reach`` hops out, and its ball
reaches ``reach`` hops further — so the extracted ball has radius ``2 * reach``.

*Small rings are not divisible.* A radius is the wrong instrument for a ring:
cut one and the remainder is not a truncated environment, it is a **different
molecule**. Ring perception on the slice finds no ring, so aromaticity — which
the RDKit model applies to "a ring, or fused ring system" as a whole, never atom
by atom — is not perceived, and every ring-aware atom type silently becomes the
aliphatic one. So the extracted ball is closed on **ring systems**: any ring of
at most :attr:`AffectedRegion.MAX_RING_SIZE` atoms that the radius only partly
reaches arrives whole, together with everything fused or bridged to it. Those
atoms are context by construction: they sit past ``extract_radius``, so their
``hops`` can never put them in ``interior``. The write-back set is exactly what
the two radii say it is; only the view around it grew.

The size bound is load-bearing, not a convenience. "A cut ring is a different
molecule" is a claim about *small* rings; a macrocycle is locally a chain, and
closing on one would pull an unbounded number of atoms into a bounded ball —
which is exactly what a single crosslink between two chains creates. With the
bound, the closure is a local flood along bonds that lie on a small ring, so
building a region costs the ball and its chemical neighbourhood, never the size
of the parent graph.

:meth:`AffectedRegion.around` is the **only** place in molpy that does this
arithmetic. It **is** an
:class:`~molpy.core.atomistic.Atomistic` — so it hands straight to third-party
typifiers / AmberTools — but adds region semantics (``interior`` / ``boundary`` /
``hops`` / ``entity_map``) and an isomorphism-invariant structural ``__hash__`` /
``__eq__`` (via the molrs Weisfeiler–Lehman graph hash), so identical polymer
junctions dedupe to one cache key.

``interior`` is the **write-back set**: every atom within ``interior_reach`` hops
of the edit. It is *not* "everything that is not boundary" — an atom on the outer
shell of the extracted ball carries truncated context, and a truncated SMARTS
environment does not fail to match, it matches the **wrong rule**. Only ``hops``
decides what is written back.

The region overrides hashing only at the *region* level; its member
:class:`~molpy.core.entity.Entity` / :class:`~molpy.core.entity.Link` views keep
their identity hashing (unchanged core contract).

The generic :class:`_RegionMixin` is shared with an eventual coarse-grained
variant; only the all-atom :class:`AffectedRegion` is instantiated here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar, Self

from molpy.core.atomistic import Atom, Atomistic
from molpy.core.entity import Entity


class _RegionMixin[E: Entity]:
    """Region semantics layered onto a molpy graph (all-atom or coarse-grained).

    Carries the ``interior`` atoms (within ``interior_reach`` of the edit — the
    ones whose types are written back), the context-only ``boundary`` shell, the
    per-atom ``hops`` distance from the edit, and the region → parent
    ``entity_map`` (to map assigned types back). The concrete subclass supplies
    the structural ``__hash__`` / ``__eq__`` over the molrs graph hash. Populated
    by :meth:`_from`, which is the region's only constructor.
    """

    #: atoms within ``interior_reach`` hops of the edit — the write-back set.
    interior: tuple[E, ...]
    #: the outer shell (each has a neighbour outside the extracted ball).
    boundary: tuple[E, ...]
    #: region entity handle → hops from the nearest touched atom.
    hops: dict[int, int]
    #: region entity → parent-graph entity (invert to map types back).
    entity_map: dict[E, E]
    #: radius of the write-back set: ``max(reach, TERM_REACH)``.
    interior_reach: int
    #: radius of the extracted ball: ``interior_reach + reach``.
    extract_radius: int


class AffectedRegion(_RegionMixin[Atom], Atomistic):
    """All-atom affected region: a hashable, typable, AmberTools-ready subgraph.

    Base order mirrors :class:`~molpy.core.atomistic.Atomistic`: the pyo3 native
    ``molrs.Atomistic`` remains the solid base (contributed by ``Atomistic``),
    with the plain-Python :class:`_RegionMixin` layered in front for its region
    attributes.

    Build one through :meth:`around`, never by guessing a radius.
    """

    #: A dihedral / improper is a 4-body term, so a term containing a newly
    #: formed bond spans at most ``4 - 2 = 2`` hops from that bond. Those atoms
    #: must be typed for the term to be looked up, which floors the write-back
    #: radius. Not a magic number: it is the arity of the widest bonded term
    #: :class:`~molpy.core.atomistic.Atomistic` carries, minus two.
    TERM_REACH: ClassVar[int] = 2

    #: Largest ring the extraction refuses to cut. Also not a magic number: it
    #: is where "ring" stops being a *local* fact. Every explicit ring-size
    #: query in the shipped chemistry asks about a small one — MMFF's atom
    #: typing asks 3, 4 and 6, UFF asks 3 and 4, aromaticity runs Hückel on
    #: SSSR rings and fuses at most six of them — and 8 covers the largest
    #: single ring that model calls aromatic (cyclooctatetraene's antiaromatic
    #: 8). Past that, a ring is a topological fact with no chemical shadow: a
    #: 5000-membered macrocycle is locally a chain, no typifier here can tell
    #: the two apart, and closing on it would drag the whole loop into a ball
    #: that asked for nine atoms — which is precisely what one crosslink
    #: between two chains creates. Region typing exists to bound work per edit;
    #: an unbounded closure would defeat it.
    #:
    #: The cost of the bound is a bare SMARTS ``[R]`` (ring membership, any
    #: size): on an atom of a macrocycle, a region slice reads it as acyclic.
    #: No in-tree typifier depends on that; a force field that did would need
    #: its own, larger bound.
    MAX_RING_SIZE: ClassVar[int] = 8

    @classmethod
    def around(
        cls, graph: Atomistic, touched: Iterable[Atom | int], *, reach: int
    ) -> Self:
        """Extract the region a ``reach``-limited typifier needs around ``touched``.

        Args:
            graph: The parent graph an edit just modified.
            touched: Seed atoms (views or molrs handles) the edit reported.
            reach: Neighbourhood radius, in bonds, that decides one atom's type.

        Returns:
            A region whose ``interior`` is ``ball(touched, max(reach, TERM_REACH))``
            — the atoms whose types are written back — inside an extracted ball of
            radius ``interior_reach + reach``.

        Raises:
            ValueError: if ``reach < 1``, if ``touched`` is empty, or if it names
                a handle that is not a live atom of ``graph``.
        """
        if reach < 1:
            raise ValueError(f"reach must be >= 1, got {reach}")
        interior_reach = max(reach, cls.TERM_REACH)
        return cls._from(
            graph,
            touched,
            extract_radius=interior_reach + reach,
            interior_reach=interior_reach,
        )

    @classmethod
    def _from(
        cls,
        parent: Atomistic,
        touched: Iterable[Atom | int],
        *,
        extract_radius: int,
        interior_reach: int,
    ) -> Self:
        """Build the region around ``touched`` in ``parent``.

        ``touched`` are the seed atoms an edit reported — :class:`Atom` views or
        raw molrs handles (as returned by ``molrs.Reaction.apply``). ``parent``
        is not mutated: the region is an induced clone with its own atom views.

        Raises:
            ValueError: if ``touched`` is empty, or names a handle that is not a
                live atom of ``parent`` (a deleted atom was reported as touched).
        """
        if interior_reach > extract_radius:
            raise ValueError(
                f"interior_reach ({interior_reach}) exceeds extract_radius "
                f"({extract_radius}): the write-back set would reach past the "
                "extracted ball"
            )
        centers = cls._resolve_centers(parent, touched)
        sub, boundary, region_to_parent, hops = parent._extract_mapped(
            centers,
            extract_radius,
            cls,
            regenerate_topology=True,
            max_ring_size=cls.MAX_RING_SIZE,
        )
        sub.hops = hops
        sub.interior = tuple(
            atom for atom in sub.atoms if hops[atom.handle] <= interior_reach
        )
        sub.boundary = tuple(boundary)
        sub.entity_map = region_to_parent
        sub.interior_reach = interior_reach
        sub.extract_radius = extract_radius
        return sub

    @classmethod
    def _resolve_centers(
        cls, parent: Atomistic, touched: Iterable[Atom | int]
    ) -> list[Atom]:
        """Normalise ``touched`` (atoms or handles) to deduplicated parent atoms.

        A handle that no longer names a live atom is a contract violation by the
        edit that produced it — a deleted atom must never be reported as touched.
        """
        seen: set[int] = set()
        centers: list[Atom] = []
        for item in touched:
            handle = item.handle if isinstance(item, Atom) else int(item)
            # A dead handle still interns to an (empty) Atom view, so probe the
            # graph itself: a live atom is always at distance 0 from itself.
            if not list(parent.topo_distances(handle, max_hops=0)):
                raise ValueError(
                    f"touched handle {handle} is not a live atom of the parent "
                    "graph (a deleted atom must not be reported as touched)"
                )
            if handle in seen:
                continue
            seen.add(handle)
            view = item if isinstance(item, Atom) else parent._intern_node(handle)
            if not isinstance(view, Atom):
                raise ValueError(f"handle {handle} did not intern to an Atom")
            centers.append(view)
        if not centers:
            raise ValueError("touched is empty: an edit must report its seed atoms")
        return centers

    def canonical_interior(self) -> tuple[int, ...]:
        """The write-back set as positions in this region's canonical order.

        A cached snapshot is replayed by canonical position, so this tuple — not
        the parent handles, which differ between two isomorphic junctions — is
        what says *which* atoms a snapshot speaks for.
        """
        position = {
            handle: index for index, handle in enumerate(self.canonical_order())
        }
        return tuple(sorted(position[atom.handle] for atom in self.interior))

    def __hash__(self) -> int:
        # Isomorphism-invariant molrs Weisfeiler–Lehman hash, rooted at the
        # write-back set — the dedup key.
        return hash((self.structural_hash(), self.canonical_interior()))

    def __eq__(self, other: object) -> bool:
        # Graph equality (resolves the rare hash collision); only regions of the
        # same kind compare — a region never equals a plain Atomistic.
        #
        # The interior is part of the identity, not an attribute of it. One
        # graph can be extracted twice around different edits — two crosslinks
        # onto the same pair of chains close a macrocycle, and from then on
        # every junction on it extracts the *whole* ring. Those regions are
        # isomorphic and must still not share a snapshot: the snapshot holds
        # one interior's types, and replaying it on the other writes the first
        # edit's atoms again while the second's stay untyped.
        return (
            isinstance(other, AffectedRegion)
            and self.canonical_interior() == other.canonical_interior()
            and self.is_isomorphic(other)
        )
