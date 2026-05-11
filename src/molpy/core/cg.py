"""Coarse-grained molecular structure.

Mirrors :mod:`molpy.core.atomistic` in shape and contract; see
:doc:`../specs/cg-atomistic-mapping-redesign` for the design rationale.

Conventional dict keys (none enforced):

* ``bead["atoms"]`` — ``tuple[Atom, ...]`` of atom references this bead
  represents (when projected from an :class:`Atomistic`). Required only by
  :meth:`CoarseGrain.beads_of`.
* ``bead["x"]``, ``bead["y"]``, ``bead["z"]`` — primary position; consumed by
  :class:`SpatialMixin` for ``move`` / ``rotate`` / ``scale`` / ``align``.
* ``bead["type"]``, ``bead["mass"]``, ``bead["charge"]`` — as needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .entity import (
    ConnectivityMixin,
    Entities,
    Entity,
    Link,
    MembershipMixin,
    SpatialMixin,
    Struct,
)

if TYPE_CHECKING:
    from .atomistic import Atom
    from .frame import Frame


class Bead(Entity):
    """Coarse-grained bead.

    Structurally identical to :class:`~molpy.core.atomistic.Atom`: a dict-like
    :class:`Entity` with no mandatory fields. All bead state lives in the
    underlying dict.

    See module docstring for the optional convention keys recognised by
    :class:`CoarseGrain` and the spatial mixin.
    """

    def __repr__(self) -> str:
        identifier: str
        if "type" in self.data:
            identifier = str(self.data["type"])
        elif "name" in self.data:
            identifier = str(self.data["name"])
        else:
            identifier = str(id(self))
        return f"<Bead: {identifier}>"


class CGBond(Link):
    """Coarse-grained bond between two beads.

    Parallel to :class:`~molpy.core.atomistic.Bond`: a :class:`Link` carrying
    two :class:`Bead` endpoints plus arbitrary attributes.
    """

    def __init__(self, a: Bead, b: Bead, /, **attrs: Any) -> None:
        """Initialise a CG bond between two beads.

        Args:
            a: First bead endpoint.
            b: Second bead endpoint.
            **attrs: Additional bond attributes.

        Raises:
            AssertionError: If ``a`` or ``b`` is not a :class:`Bead` instance.
        """
        assert isinstance(a, Bead), f"a must be Bead, got {type(a)}"
        assert isinstance(b, Bead), f"b must be Bead, got {type(b)}"
        super().__init__([a, b], **attrs)

    def __repr__(self) -> str:
        return f"<CGBond: {self.ibead} - {self.jbead}>"

    @property
    def ibead(self) -> Bead:
        """First bead endpoint."""
        return self.endpoints[0]

    @property
    def jbead(self) -> Bead:
        """Second bead endpoint."""
        return self.endpoints[1]


class CoarseGrain(Struct, MembershipMixin, SpatialMixin, ConnectivityMixin):
    """Coarse-grained molecular structure.

    Public surface mirrors :class:`~molpy.core.atomistic.Atomistic` 1:1
    except for the single extra method :meth:`beads_of`. The class makes no
    assumption about the existence, source, or definition of bead positions,
    masses, or atom-level provenance — those are dict-key conventions
    documented at module level.
    """

    def __init__(self, **props: Any) -> None:
        """Initialise an empty coarse-grained structure.

        Registers buckets for :class:`Bead` and :class:`CGBond`. If the
        concrete subclass defines a ``__post_init__`` method, it is called
        with the same keyword arguments (template pattern, parallel to
        :class:`Atomistic`).

        Args:
            **props: Arbitrary properties stored on the structure (e.g.,
                ``name="POPC"``, ``model="martini3"``).
        """
        super().__init__(**props)
        if hasattr(self, "__post_init__"):
            for klass in type(self).__mro__:
                if klass is CoarseGrain:
                    break
                if "__post_init__" in klass.__dict__:
                    klass.__dict__["__post_init__"](self, **props)
                    break
        self.entities.register_type(Bead)
        self.links.register_type(CGBond)

    @property
    def beads(self) -> Entities[Bead]:
        """All beads registered in this structure."""
        return self.entities[Bead]

    @property
    def cgbonds(self) -> Entities[CGBond]:  # type: ignore[type-var]
        """All CG bonds registered in this structure."""
        return self.links[CGBond]  # type: ignore[return-value]

    def __repr__(self) -> str:
        from collections import Counter

        types = Counter(b.get("type", "?") for b in self.beads)
        if len(types) <= 5:
            comp = " ".join(f"{t}:{n}" for t, n in sorted(types.items()))
        else:
            comp = f"{len(types)} types"
        return (
            f"<CoarseGrain, {len(self.beads)} beads ({comp}), "
            f"{len(self.cgbonds)} bonds>"
        )

    def __len__(self) -> int:
        """Return the number of beads."""
        return len(self.beads)

    # ========== Factory Methods (def_*: create + register) ==========

    def def_bead(self, /, **attrs: Any) -> Bead:
        """Create a new :class:`Bead` and register it.

        Args:
            **attrs: Bead attributes (any keys; see module docstring for
                conventional keys).

        Returns:
            The newly registered Bead.
        """
        bead = Bead(**attrs)
        self.entities.add(bead)
        return bead

    def def_cgbond(self, a: Bead, b: Bead, /, **attrs: Any) -> CGBond:
        """Create a new :class:`CGBond` between two beads and register it.

        Args:
            a: First bead endpoint.
            b: Second bead endpoint.
            **attrs: Additional bond attributes (e.g., ``type="A-B"``).

        Returns:
            The newly registered CG bond.
        """
        bond = CGBond(a, b, **attrs)
        self.links.add(bond)
        return bond

    def add_bead(self, bead: Bead, /) -> Bead:
        """Register an existing :class:`Bead`.

        Args:
            bead: The bead to register.

        Returns:
            The same bead, after registration.
        """
        self.entities.add(bead)
        return bead

    def add_cgbond(self, bond: CGBond, /) -> CGBond:
        """Register an existing :class:`CGBond`.

        Args:
            bond: The CG bond to register.

        Returns:
            The same bond, after registration.
        """
        self.links.add(bond)
        return bond

    def del_bead(self, *beads: Bead) -> None:
        """Remove beads (and their incident CG bonds).

        Args:
            *beads: One or more :class:`Bead` instances to remove. Each
                bead is unregistered along with any :class:`CGBond` that
                references it as an endpoint.
        """
        self.remove_entity(*beads)

    def del_cgbond(self, *bonds: CGBond) -> None:
        """Remove CG bonds.

        Args:
            *bonds: One or more :class:`CGBond` instances to remove.
        """
        self.remove_link(*bonds)

    def def_beads(self, beads_data: list[dict[str, Any]], /) -> list[Bead]:
        """Create multiple beads from a list of attribute dicts.

        Args:
            beads_data: List of attribute dicts; each dict is forwarded as
                ``**attrs`` to :meth:`def_bead`.

        Returns:
            List of newly registered beads, one per input dict.
        """
        return [self.def_bead(**a) for a in beads_data]

    def def_cgbonds(
        self,
        bonds_data: list[tuple[Bead, Bead] | tuple[Bead, Bead, dict[str, Any]]],
        /,
    ) -> list[CGBond]:
        """Create multiple CG bonds from ``(a, b)`` or ``(a, b, attrs)`` tuples.

        Args:
            bonds_data: Iterable of bond specifications. Each item is either
                a 2-tuple ``(a, b)`` of bead endpoints or a 3-tuple
                ``(a, b, attrs)`` where ``attrs`` is a dict forwarded as
                ``**attrs`` to :meth:`def_cgbond`.

        Returns:
            List of newly registered CG bonds, one per input tuple.
        """
        out: list[CGBond] = []
        for spec in bonds_data:
            if len(spec) == 2:
                a, b = spec  # type: ignore[misc]
                attrs: dict[str, Any] = {}
            else:
                a, b, attrs = spec  # type: ignore[misc]
            out.append(self.def_cgbond(a, b, **attrs))
        return out

    def add_beads(self, beads: list[Bead], /) -> list[Bead]:
        """Register multiple existing beads.

        Args:
            beads: List of :class:`Bead` instances to register.

        Returns:
            The same list, after registration.
        """
        for bead in beads:
            self.entities.add(bead)
        return beads

    def add_cgbonds(self, bonds: list[CGBond], /) -> list[CGBond]:
        """Register multiple existing CG bonds.

        Args:
            bonds: List of :class:`CGBond` instances to register.

        Returns:
            The same list, after registration.
        """
        for bond in bonds:
            self.links.add(bond)
        return bonds

    # ========== Reverse Lookup (the one extra core method) ==========

    def beads_of(self, atom: "Atom") -> tuple[Bead, ...]:
        """Return all beads whose ``bead["atoms"]`` contains ``atom``.

        Justified as a core method by the same standard as :meth:`move`:
        operates on a single conventional key (``"atoms"``) and has no
        second reasonable implementation. Beads without an ``"atoms"``
        key are skipped silently.

        Args:
            atom: The atom to look up.

        Returns:
            Tuple of beads referencing ``atom``. Empty if none; multiple
            if the mapping has overlap.

        Note:
            O(N_beads × ⟨|atoms|⟩); no caching. Callers needing speed
            should build their own ``id(atom) → list[Bead]`` index.
        """
        return tuple(b for b in self.beads if atom in b.get("atoms", ()))

    # ========== Property / Type / Selection Editing ==========

    def rename_type(self, old: str, new: str, *, kind: type = Bead) -> int:
        """Rename all beads/bonds of ``kind`` whose ``type`` equals ``old``.

        Returns:
            Number of items renamed.
        """
        if issubclass(kind, Link):
            items = self.links.bucket(kind)
        else:
            items = self.entities.bucket(kind)
        count = 0
        for item in items:
            if item.get("type") == old:
                item["type"] = new
                count += 1
        return count

    def set_property(
        self,
        selector: Callable[[Bead], bool],
        key: str,
        value: Any,
        *,
        kind: type = Bead,
    ) -> int:
        """Set a property on every bead (or link) matching ``selector``.

        Args:
            selector: Callable ``(item) -> bool`` returning ``True`` for
                items whose ``key`` should be set.
            key: Attribute key to assign on matching items.
            value: Value to assign.
            kind: Item class to iterate over (default :class:`Bead`); pass a
                :class:`Link` subclass to operate on bonds instead.

        Returns:
            Number of items modified.

        Raises:
            TypeError: If ``selector`` is not callable.
        """
        if not callable(selector):
            raise TypeError(
                "selector must be a callable (bead) -> bool; "
                "SMARTS-string selectors are not yet supported"
            )
        if issubclass(kind, Link):
            items = self.links.bucket(kind)
        else:
            items = self.entities.bucket(kind)
        count = 0
        for item in items:
            if selector(item):
                item[key] = value
                count += 1
        return count

    def select(self, predicate: Callable[[Bead], bool]) -> "CoarseGrain":
        """Return a new CoarseGrain containing beads matching ``predicate``.

        Bonds whose endpoints are both inside the selection are carried over.

        Args:
            predicate: Callable ``(bead) -> bool`` selecting beads to keep.

        Returns:
            A new :class:`CoarseGrain` containing the selected beads and the
            bonds whose endpoints are entirely within the selection.

        Raises:
            TypeError: If ``predicate`` is not callable.
        """
        if not callable(predicate):
            raise TypeError(
                "predicate must be a callable (bead) -> bool; "
                "SMARTS-string predicates are not yet supported"
            )
        selected = [b for b in self.beads if predicate(b)]
        sub, _ = self.extract_subgraph(
            selected, radius=0, entity_type=Bead, link_type=Link
        )
        return sub  # type: ignore[return-value]

    # ========== Spatial Operations (return self for chaining) ==========

    def move(
        self, delta: list[float], *, entity_type: type[Entity] = Bead
    ) -> "CoarseGrain":
        """Translate every bead by ``delta``.

        Args:
            delta: Translation vector ``[dx, dy, dz]`` in Angstroms.
            entity_type: Entity subclass to translate (default :class:`Bead`).

        Returns:
            ``self`` for method chaining.
        """
        super().move(delta, entity_type=entity_type)
        return self

    def rotate(
        self,
        axis: list[float],
        angle: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity] = Bead,
    ) -> "CoarseGrain":
        """Rotate beads around ``axis`` by ``angle``.

        Args:
            axis: Rotation axis ``[ax, ay, az]`` (need not be unit length).
            angle: Rotation angle in radians.
            about: Optional pivot point ``[x, y, z]`` in Angstroms; defaults
                to the origin.
            entity_type: Entity subclass to rotate (default :class:`Bead`).

        Returns:
            ``self`` for method chaining.
        """
        super().rotate(axis, angle, about=about, entity_type=entity_type)
        return self

    def scale(
        self,
        factor: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity] = Bead,
    ) -> "CoarseGrain":
        """Scale bead positions by ``factor``.

        Args:
            factor: Uniform scaling factor (dimensionless).
            about: Optional pivot point ``[x, y, z]`` in Angstroms; defaults
                to the origin.
            entity_type: Entity subclass to scale (default :class:`Bead`).

        Returns:
            ``self`` for method chaining.
        """
        super().scale(factor, about=about, entity_type=entity_type)
        return self

    def align(
        self,
        a: Entity,
        b: Entity,
        *,
        a_dir: list[float] | None = None,
        b_dir: list[float] | None = None,
        flip: bool = False,
        entity_type: type[Entity] = Bead,
    ) -> "CoarseGrain":
        """Align beads via a vector pair.

        Args:
            a: Source entity defining the start of the alignment vector.
            b: Target entity defining the end of the alignment vector.
            a_dir: Optional source direction ``[x, y, z]``; if omitted, the
                vector from ``a`` to ``b`` is used.
            b_dir: Optional target direction ``[x, y, z]``.
            flip: If ``True``, reverse the alignment direction.
            entity_type: Entity subclass to transform (default :class:`Bead`).

        Returns:
            ``self`` for method chaining.
        """
        super().align(
            a, b, a_dir=a_dir, b_dir=b_dir, flip=flip, entity_type=entity_type
        )
        return self

    # ========== System Composition ==========

    def __iadd__(self, other: "CoarseGrain") -> "CoarseGrain":
        """Merge ``other`` into this structure in-place.

        Args:
            other: Source structure whose entities and links are transferred
                into ``self``. After the call, ``other`` should not be used.

        Returns:
            ``self`` for method chaining (in-place merge).
        """
        self.merge(other)
        return self

    def __add__(self, other: "CoarseGrain") -> "CoarseGrain":
        """Return a new structure that is the union of ``self`` and ``other``.

        Args:
            other: Right-hand operand whose contents are merged into a copy
                of ``self``. Neither operand is modified.

        Returns:
            A new :class:`CoarseGrain` containing copies of all entities and
            links from both structures.
        """
        result = self.copy()
        result.merge(other)
        return result

    def replicate(
        self,
        n: int,
        transform: Callable[["CoarseGrain", int], None] | None = None,
    ) -> "CoarseGrain":
        """Replicate ``self`` ``n`` times into a new structure.

        Args:
            n: Number of copies.
            transform: Optional callable ``(copy, index) -> None`` applied to
                each replica before merging.

        Returns:
            A new :class:`CoarseGrain` containing ``n`` merged replicas.
        """
        result = type(self)()
        for i in range(n):
            replica = self.copy()
            if transform is not None:
                transform(replica, i)
            result.merge(replica)
        return result

    # ========== Tabular Conversion ==========

    def to_frame(self, bead_fields: list[str] | None = None) -> "Frame":
        """Convert this CoarseGrain into a :class:`Frame`.

        Mirrors :meth:`molpy.core.atomistic.Atomistic.to_frame`. Beads are
        flattened into a ``"beads"`` :class:`Block` (struct-of-arrays);
        CG bonds, when present, become a ``"cgbonds"`` block carrying
        integer endpoint indices ``ibead`` / ``jbead`` plus any remaining
        bond attributes.

        Args:
            bead_fields: Optional whitelist of bead dict keys to extract.
                If ``None``, every key encountered on any bead is included.
                Note: a bead may carry a ``"atoms"`` convention key whose
                value is a tuple of :class:`Atom` references; including
                that key produces an object-dtype column. Pass an explicit
                whitelist to skip it when writing to numerical formats.

        Returns:
            A :class:`Frame` with a ``"beads"`` block and (if any CG bonds
            exist) a ``"cgbonds"`` block. The frame carries no
            :class:`Box`; callers attach one as needed.

        Raises:
            ValueError: If a CG bond references a bead that is not
                registered in this structure.
        """
        import numpy as np

        from .frame import Block, Frame

        frame = Frame()
        beads = list(self.beads)
        cgbonds = list(self.cgbonds)

        # ---- beads block ----
        if bead_fields is None:
            keys: set[str] = set()
            for bead in beads:
                keys.update(bead.keys())
        else:
            keys = set(bead_fields)

        bead_dict: dict[str, list[Any]] = {k: [] for k in keys}
        bead_index: dict[int, int] = {}
        for bead in beads:
            bead_index[id(bead)] = len(bead_index)
            for key in keys:
                bead_dict[key].append(bead.get(key, None))

        bead_arrays = {
            k: np.array(v, dtype=object) if k == "atoms" else np.array(v)
            for k, v in bead_dict.items()
        }
        frame["beads"] = Block.from_dict(bead_arrays)

        # ---- cgbonds block ----
        if cgbonds:
            cgbond_dict: dict[str, list[Any]] = {"ibead": [], "jbead": []}
            all_bond_keys: set[str] = set()
            for bond in cgbonds:
                all_bond_keys.update(bond.keys())
            for key in all_bond_keys:
                if key not in ("ibead", "jbead"):
                    cgbond_dict[key] = []

            for k, bond in enumerate(cgbonds):
                if id(bond.ibead) not in bead_index:
                    raise ValueError(
                        f"CGBond {k + 1}: ibead (id={id(bond.ibead)}) is not "
                        f"registered in this CoarseGrain."
                    )
                if id(bond.jbead) not in bead_index:
                    raise ValueError(
                        f"CGBond {k + 1}: jbead (id={id(bond.jbead)}) is not "
                        f"registered in this CoarseGrain."
                    )
                cgbond_dict["ibead"].append(bead_index[id(bond.ibead)])
                cgbond_dict["jbead"].append(bead_index[id(bond.jbead)])
                for key in all_bond_keys:
                    if key not in ("ibead", "jbead"):
                        cgbond_dict[key].append(bond.get(key, None))

            cgbond_arrays = {k: np.array(v) for k, v in cgbond_dict.items()}
            frame["cgbonds"] = Block.from_dict(cgbond_arrays)

        return frame
