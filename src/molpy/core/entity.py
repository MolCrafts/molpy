from collections import UserDict
from collections.abc import Iterable, Iterator
from copy import deepcopy
from typing import Any, Protocol, Self, TypeVar, cast, overload

import molrs

from molpy.core.ops.geometry import (
    _cross,
    _dot,
    _norm,
    _rodrigues_rotate,
    _unit,
    _vec_add,
    _vec_scale,
    _vec_sub,
)
from molpy.core.utils import get_nearest_type


def _snapshot_data(obj: Any) -> dict[str, Any]:
    """Return a plain ``dict`` of an entity/link's attributes.

    Works for both detached handles (``self.data`` is a real dict) and bound
    handles (``self.data`` is a live graph proxy). Crucially this never returns
    the proxy itself, so the backing molrs graph is never deepcopied/pickled.
    """
    data = getattr(obj, "data", None)
    if data is None:
        return {}
    return dict(data)


class EntityLike(Protocol):
    """Protocol for objects that can act as Entities (Entity or subclass)."""

    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, item: Any) -> None: ...
    def __hash__(self) -> int: ...
    def get(self, key: str, default: Any = None) -> Any: ...


class Entity(UserDict):
    """Dictionary-like base object for all structure elements.

    Dual-mode (spec ``atomistic-cg-on-molrs-molgraph``, P1):

    * **Detached** — created standalone (e.g. ``Atom(element="C")``). Behaves
      exactly as a plain :class:`collections.UserDict`; ``self.data`` is an
      ordinary dict it owns.
    * **Bound** — once a :class:`Struct` adds it, :meth:`bind` swaps
      ``self.data`` for a live :class:`~molpy.core._handle._AtomPropProxy` so
      every ``UserDict`` operation proxies to the backing molrs graph node.

    Identity is preserved across the transition: the same handle object stays
    in the struct's bucket, so ``bond.itom is atom`` and ``id(atom)`` keyed
    bookkeeping survive the migration.
    """

    #: backing graph once bound, else ``None``.
    _graph: Any = None
    #: node index in the backing graph once bound (compacts on removal).
    _index: int = -1

    @property
    def is_bound(self) -> bool:
        """True once this entity is backed by a graph node."""
        return self._graph is not None

    def bind(self, graph: Any, index: int) -> None:
        """Attach this entity to ``graph`` node ``index``, migrating ``data``.

        The current ``self.data`` contents are written into the graph node's
        property bag (D5: non int/float/str values stay in the proxy's Python
        fallback), then ``self.data`` is replaced by the live proxy.
        """
        from molpy.core._handle import _AtomPropProxy

        proxy = _AtomPropProxy(graph, self)
        self._graph = graph
        self._index = index
        detached = dict(self.data)
        # ``add_atom`` always seeds an ``element`` prop on the molrs node; drop
        # it if this entity never carried one so reads stay faithful (e.g.
        # crystal atoms keyed on ``symbol`` rather than ``element``).
        if "element" not in detached and "element" in proxy._g_keys():
            proxy._g_del("element")
        # migrate existing detached attributes into the proxy
        for k, v in detached.items():
            proxy[k] = v
        self.data = proxy

    def __deepcopy__(self, memo: dict) -> "Entity":
        """Deep-copy into a *detached* entity.

        A bound entity's ``data`` is a live graph proxy holding the (unpicklable)
        molrs graph. Deep-copying snapshots the visible attributes into a plain
        dict and returns a fresh detached handle, so ``copy.deepcopy`` works and
        the clone is independent of any graph.
        """
        clone = self.__class__()
        memo[id(self)] = clone
        clone.data = deepcopy(dict(self.data), memo)
        return clone

    # Keep identity-based hashing/equality from UserDict's object semantics
    def __hash__(self) -> int:  # pragma: no cover - trivial identity
        return id(self)


class LinkLike(Protocol):
    """Protocol for objects that can act as Links (Link or subclass)."""

    data: dict[str, Any]
    endpoints: tuple[EntityLike, ...]

    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, item: Any) -> None: ...
    def __hash__(self) -> int: ...
    def get(self, key: str, default: Any = None) -> Any: ...


class Link[T: Entity](UserDict):
    """Connectivity object holding direct references to endpoint entities.

    Dual-mode (spec ``atomistic-cg-on-molrs-molgraph``, P1), parallel to
    :class:`Entity`:

    * **Detached** — a plain :class:`collections.UserDict` owning ``self.data``.
    * **Bound** — :meth:`bind` records a molrs link id (``_index``) and swaps
      ``self.data`` for a live :class:`~molpy.core._handle._LinkPropProxy`
      routing attribute reads/writes to the graph link's property bag.

    ``endpoints`` always holds the (bound) endpoint :class:`Entity` handles, so
    ``bond.itom``, ``angle.ktom``, :meth:`replace_endpoint`, and identity
    semantics behave the same whether or not the link is graph-backed.

    Attributes
    ----------
    endpoints: tuple[Entity]
        The ordered tuple of endpoint entity references.
    """

    endpoints: tuple[T, ...]

    #: molrs link-kind tag — overridden per concrete subclass.
    _link_kind: str = "bond"

    _graph: Any = None
    _index: int = -1

    def __init__(self, endpoints: Iterable[T], /, **attrs: Any):
        super().__init__()
        self.endpoints = tuple(endpoints)
        # store remaining attributes in the mapping
        for k, v in attrs.items():
            self.data[k] = v

    @property
    def is_bound(self) -> bool:
        """True once this link is backed by a graph link."""
        return self._graph is not None

    def bind(self, graph: Any, index: int) -> None:
        """Attach this link to ``graph`` link ``index``, migrating ``data``."""
        from molpy.core._handle import _LinkPropProxy

        proxy = _LinkPropProxy(graph, self)
        self._graph = graph
        self._index = index
        for k, v in list(self.data.items()):
            proxy[k] = v
        self.data = proxy

    # ----- molrs link-kind dispatch (used by _LinkPropProxy) -----
    def _kind_keys(self, graph: Any, idx: int) -> list[str]:
        return list(getattr(graph, f"{self._link_kind}_keys")(idx))

    def _kind_get(self, graph: Any, idx: int, key: str) -> Any:
        return getattr(graph, f"get_{self._link_kind}_prop")(idx, key)

    def _kind_set(self, graph: Any, idx: int, key: str, value: Any) -> None:
        getattr(graph, f"set_{self._link_kind}_prop")(idx, key, value)

    def _kind_del(self, graph: Any, idx: int, key: str) -> None:
        getattr(graph, f"del_{self._link_kind}_prop")(idx, key)

    def replace_endpoint(self, old: T, new: T) -> None:
        """Replace one endpoint reference with another in-place."""
        self.endpoints = tuple(new if e is old else e for e in self.endpoints)

    def __deepcopy__(self, memo: dict) -> "Link":
        """Deep-copy into a *detached* link (see :meth:`Entity.__deepcopy__`)."""
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        clone._graph = None
        clone._index = -1
        clone.endpoints = tuple(deepcopy(ep, memo) for ep in self.endpoints)
        UserDict.__init__(clone)
        clone.data = deepcopy(dict(self.data), memo)
        return clone

    def __hash__(self) -> int:  # pragma: no cover - trivial identity
        return id(self)


# Note: EntityLike is already defined above (line 18), removing duplicate definition
# The first definition is the canonical one with full protocol methods

E = TypeVar("E", bound=Entity)
U = TypeVar("U", bound=Entity)


# ---------- Column-friendly list ----------
class Entities[E: Entity](list[E]):
    """A list of Entity-like objects supporting column-style access via a string key.

    When accessing with a string key, returns a numpy array if numpy is available,
    otherwise returns a list.
    """

    @overload
    def __getitem__(self, key: int) -> E: ...  # type: ignore[override]
    @overload
    def __getitem__(self, key: slice) -> list[E]: ...  # type: ignore[override]
    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: int | slice | str) -> E | list[E] | Any:  # type: ignore[override]
        if isinstance(key, str):
            # Column access - returns numpy array if available
            values = [ent.get(key) for ent in self]
            try:
                import numpy as np

                return np.array(values)
            except ImportError:
                return values
        return super().__getitem__(key)


# ---------- Helper: choose bucket key (override if needed) ----------
# Note: get_nearest_type is now imported from utils


# ---------- Entity-specific TypeBucket (returns Entities type) ----------
class TypeBucket[E: Entity]:
    """
    Entity-specific TypeBucket that groups and stores objects by their concrete type.
    Uses Entities as container to support column-style access.

    Bucket objects by (concrete) type using dict[type, Entities].
    - Key:  type[U], where U <: E
    - Item: Entities[U], paired with the same U as in the key

    Query methods return Entities[...] so you can do column-style access directly.
    """

    def __init__(self, binder: Any = None, unbinder: Any = None) -> None:
        # Internal store uses Any for flexibility across entity types
        self._items: dict[type[Any], Entities[Any]] = {}
        # Mirror the list contents as an identity-set for O(1) dedup.
        self._ids: dict[type[Any], set[int]] = {}
        # Optional graph-binding hooks (set by an owning Struct). When present,
        # ``add`` binds an item into the backing graph the first time it is
        # registered, and ``remove`` detaches it. Standalone buckets (no owner)
        # leave both ``None`` and behave as a plain identity-deduped container.
        self._binder = binder
        self._unbinder = unbinder

    # ----- mutate -----
    def add(self, item: E) -> None:
        """Add one object to the bucket for its nearest type.

        Dedup is by object identity (``id(item)``) so the same instance
        is never stored twice. Both insertion and the identity check
        are O(1).
        """
        cls = get_nearest_type(item)  # type: ignore[arg-type]
        bucket = self._items.setdefault(cls, Entities())
        idset = self._ids.setdefault(cls, set())
        iid = id(item)
        if iid in idset:
            return
        if self._binder is not None:
            self._binder(item)
        bucket.append(item)
        idset.add(iid)

    def add_many(self, items: Iterable[E]) -> None:
        """Add multiple objects."""
        for it in items:
            self.add(it)

    def remove(self, item: E) -> bool:
        """Remove an object from its bucket; returns True if removed."""
        cls = get_nearest_type(item)  # type: ignore[arg-type]
        bucket = self._items.get(cls)
        if not bucket:
            return False
        idset = self._ids.get(cls)
        if idset is None or id(item) not in idset:
            return False
        for i, obj in enumerate(bucket):
            if obj is item:
                if self._unbinder is not None:
                    self._unbinder(item)
                bucket.pop(i)
                idset.discard(id(item))
                if not bucket:
                    self._items.pop(cls, None)
                    self._ids.pop(cls, None)
                return True
        return False

    def register_type(self, cls: type[Any]) -> None:
        """Ensure a bucket exists for the given class."""
        self._items.setdefault(cls, Entities())

    # ----- queries (return Entities) -----
    def all(self) -> Entities[E]:
        """All items across all buckets (returns a new Entities)."""
        out: Entities[E] = Entities()
        for b in self._items.values():
            out.extend(cast(Entities[E], b))
        return out

    def exact_bucket(self, cls: type[U]) -> Entities[U]:
        """Items whose concrete class is exactly 'cls' (no subclasses)."""
        b = self._items.get(cls)
        return Entities(cast(Entities[U], b)) if b else Entities()

    def bucket(self, cls: type[U]) -> Entities[U]:
        """
        Items whose concrete class is 'cls' or any subclass of 'cls'.
        Returns a new Entities[U].
        """
        out: Entities[U] = Entities()
        if cls in self._items:
            out.extend(cast(Entities[U], self._items[cls]))
        for k, b in self._items.items():
            try:
                if k is not cls and isinstance(k, type) and issubclass(k, cls):  # type: ignore[arg-type]
                    out.extend(cast(Entities[U], b))
            except TypeError:
                # Skip if k is not a proper class
                pass
        return out

    def classes(self) -> Iterator[type[E]]:
        """Concrete classes that currently have buckets."""
        return cast(Iterator[type[E]], iter(self._items.keys()))

    def __len__(self) -> int:
        """Total number of stored objects across all buckets."""
        return sum(len(b) for b in self._items.values())

    def __getitem__(self, cls: type[U]) -> Entities[U]:
        """Get bucket for class (includes subclasses)."""
        return self.bucket(cls)

    def __setitem__(self, cls: type[U], items: Iterable[U]) -> None:
        """Set the bucket for a given class."""
        self._items[cls] = Entities(items)


class StructLike(Protocol):
    """Protocol for objects that can act as Structs (Struct or subclass).

    Defines the interface for structural containers that hold entities and links.
    """

    entities: TypeBucket[Any]
    links: TypeBucket[Any]


T = TypeVar("T", bound="Struct")


class Struct:
    """Container holding entities and links via typed buckets.

    This is the root class for all molecular structure types in MolPy.
    Supports entity/link management and serves as the base for wrappers.

    A Struct is a typed container that organizes entities (e.g., atoms, residues)
    and links (e.g., bonds, angles) into type-specific buckets for efficient
    access and manipulation.

    Backing (P1, spec ``atomistic-cg-on-molrs-molgraph``): the concrete
    containers (``Atomistic`` / ``CoarseGrain``) root on the molrs general graph
    (:class:`molrs.Graph`) by listing it as their LAST base, so molpy's mixin
    methods (e.g. ``SpatialMixin.rotate``) win in the MRO over the molrs base's
    same-named methods. ``Struct`` itself stays mixin-friendly.
    """

    def __init__(self, **props: Any) -> None:
        """Initialize a new Struct.

        Args:
            **props: Additional properties to store in the struct
        """
        # P1 (spec atomistic-cg-on-molrs-molgraph): the concrete subclass IS a
        # molrs graph (it lists ``molrs.Graph`` as its last base). The buckets
        # below are wired with binders so that registering an entity / link
        # migrates its storage into ``self`` (the graph) and hands the handle a
        # live property proxy. Identity / ordering still live in the buckets.
        self.entities: TypeBucket[Any] = TypeBucket(
            binder=self._bind_entity, unbinder=self._unbind_entity
        )
        self.links: TypeBucket[Any] = TypeBucket(
            binder=self._bind_link, unbinder=self._unbind_link
        )
        self._props: dict[str, Any] = dict(props)
        # Insertion-ordered registries mirroring the molrs graph's compacting
        # index spaces. The handle at position ``k`` here has molrs index ``k``.
        # Keeping these explicit (rather than deriving from type-grouped
        # buckets) keeps indices correct even with mixed Entity subclasses.
        self._ordered_nodes: list[Entity] = []
        self._ordered_links: dict[str, list[Link]] = {}

    # ---------- graph binding hooks (P1) ----------
    def _bind_entity(self, ent: "Entity") -> None:
        """Add ``ent`` as a graph node and bind it to the backing graph.

        No-op when ``self`` is not a molrs graph (defensive: a bare ``Struct``
        instantiated directly has no graph backend, so handles stay detached
        and the bucket degrades to a plain container).
        """
        if not isinstance(self, molrs.Graph):
            return
        if ent.is_bound:
            return
        # molrs ``add_atom`` requires a symbol; pass the real element when the
        # entity has one, else a placeholder that ``Entity.bind`` strips back
        # out. molpy overrides ``add_atom`` with an (atom)->atom helper, so
        # reach the molrs node-creating method through the base class.
        symbol = str(ent.data.get("element") or "X")
        index = molrs.Graph.add_atom(self, symbol)
        ent.bind(self, index)
        self._ordered_nodes.append(ent)

    def _unbind_entity(self, ent: "Entity") -> None:
        if not getattr(ent, "is_bound", False):
            return
        graph = ent._graph
        idx = ent._index
        # snapshot BEFORE clearing the index (the proxy reads ``_index``)
        snapshot = dict(ent.data)
        ent._graph = None
        ent._index = -1
        ent.data = snapshot
        molrs.Graph.remove_atom(graph, idx)
        # molrs compacts node ids on removal: drop this handle and shift the
        # index of every node that sat after it.
        self._ordered_nodes.pop(idx)
        for k in range(idx, len(self._ordered_nodes)):
            self._ordered_nodes[k]._index = k

    def _bind_link(self, link: "Link") -> None:
        if not isinstance(self, molrs.Graph):
            return
        if link.is_bound:
            return
        # Orphan link: an endpoint is not registered in *this* struct's graph.
        # Leave the link detached (it still lives in the bucket) so legacy
        # orphan-link semantics survive — e.g. ``to_frame`` raising later.
        if any(getattr(ep, "_graph", None) is not self for ep in link.endpoints):
            return
        idxs = [ep._index for ep in link.endpoints]
        kind = link._link_kind
        # reach molrs link-creating methods through the base class (molpy
        # shadows the same names with (object)->object helpers).
        if kind == "bond":
            molrs.Graph.add_bond(self, idxs[0], idxs[1])
            index = self.n_bonds - 1
        elif kind == "angle":
            index = molrs.Graph.add_angle(self, idxs[0], idxs[1], idxs[2])
        elif kind == "dihedral":
            index = molrs.Graph.add_dihedral(self, idxs[0], idxs[1], idxs[2], idxs[3])
        elif kind == "improper":
            index = molrs.Graph.add_improper(self, idxs[0], idxs[1], idxs[2], idxs[3])
        else:  # pragma: no cover - defensive
            return
        link.bind(self, index)
        self._ordered_links.setdefault(kind, []).append(link)

    def _unbind_link(self, link: "Link") -> None:
        if not getattr(link, "is_bound", False):
            return
        graph = link._graph
        idx = link._index
        kind = link._link_kind
        # snapshot BEFORE clearing the index (the proxy reads ``_index``)
        snapshot = dict(link.data)
        link._graph = None
        link._index = -1
        link.data = snapshot
        getattr(molrs.Graph, f"remove_{kind}")(graph, idx)
        # per-kind link id space also compacts on removal.
        order = self._ordered_links.get(kind, [])
        if 0 <= idx < len(order):
            order.pop(idx)
            for k in range(idx, len(order)):
                order[k]._index = k

    # ---------- dict-like access to props ----------
    def __getitem__(self, key: str) -> Any:
        """Get property by key.

        Args:
            key: Property key

        Returns:
            Property value

        Raises:
            KeyError: If key doesn't exist
        """
        return self._props[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set property by key.

        Args:
            key: Property key
            value: Property value
        """
        self._props[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in props.

        Args:
            key: Property key to check

        Returns:
            True if key exists, False otherwise
        """
        return key in self._props

    def get(self, key: str, default: Any = None) -> Any:
        """Get property with default.

        Args:
            key: Property key
            default: Default value if key not found

        Returns:
            Property value or default
        """
        return self._props.get(key, default)

    # ---------- helpers ----------
    def _iter_all_entities(self) -> Iterable[Entity]:
        for cls in self.entities.classes():
            yield from self.entities.bucket(cls)

    def _iter_all_links(self) -> Iterable[Link]:
        for cls in self.links.classes():
            yield from self.links.bucket(cls)

    # ---------- built-ins ----------
    def copy(self) -> Self:
        """Create a deep copy of this struct.

        All entities and links are deep-copied. Entity references
        within links are remapped to the new copies.

        Returns:
            New Struct with independent copies of all entities and links.

        Related:
            - merge
        """
        new = type(self)()
        # deep-copy entities (snapshot graph-backed data to plain dicts so the
        # backing molrs graph is never deepcopied/pickled)
        emap: dict[Entity, Entity] = {}
        for ent in self._iter_all_entities():
            cloned = ent.__class__(deepcopy(_snapshot_data(ent)))
            emap[ent] = cloned
            new.entities.add(cloned)

        # deep-copy links (remap endpoints)
        for link in self._iter_all_links():
            # Ensure all endpoints are in emap (defensive check)
            for ep in link.endpoints:
                if ep not in emap:
                    # Edge case: endpoint not in entities bucket
                    # This shouldn't happen in well-formed assemblies
                    cloned_ep = ep.__class__(deepcopy(_snapshot_data(ep)))
                    emap[ep] = cloned_ep
                    new.entities.add(cloned_ep)

            mapped_eps = [emap[ep] for ep in link.endpoints]
            attrs = deepcopy(_snapshot_data(link))
            lcls: type[Link] = type(link)
            try:
                new_link = lcls(*mapped_eps, **attrs)  # Endpoints as positional args
            except TypeError:
                new_link = lcls(mapped_eps, **attrs)  # Or as list
            new.links.add(new_link)

        return new

    def merge(self, other: "Struct") -> Self:
        """
        Transfer all entities and links from another struct into self.

        **NO deep copy** - entities and links are directly transferred.
        After merge, `other` should not be used (its entities now belong to self).

        Args:
            other: Struct to merge into self

        Returns:
            Self for method chaining

        Raises:
            ValueError: If struct contains orphan links (endpoints not in entities)

        Example:
            >>> struct1.merge(struct2)  # Transfers struct2 into struct1
            >>> # struct2 should not be used after this!
        """
        # Snapshot membership before mutating either side.
        other_entity_list = list(other._iter_all_entities())
        other_entities = set(other_entity_list)
        other_links = list(other._iter_all_links())

        # Validate links up front (orphan detection) before any transfer.
        for link in other_links:
            if any(ep not in other_entities for ep in link.endpoints):
                raise ValueError(
                    "Found link with endpoints not in entities bucket. "
                    "This indicates orphan links in the struct."
                )

        # Transfer entities: detach each handle from ``other``'s graph and
        # rebind it into ``self``. The handle object is reused (so external
        # references and link endpoints stay valid); ``other`` must not be
        # used afterwards.
        for ent in other_entity_list:
            self._adopt_entity(ent, other)

        # Transfer links similarly, rebinding onto the (now self-bound)
        # endpoints.
        for link in other_links:
            self._adopt_link(link, other)

        return self

    # ---------- merge helpers (P1) ----------
    def _adopt_entity(self, ent: "Entity", other: "Struct") -> None:
        """Detach ``ent`` from ``other`` and register it in ``self``."""
        if getattr(ent, "is_bound", False):
            snapshot = _snapshot_data(ent)
            ent._graph = None
            ent._index = -1
            ent.data = snapshot
        self.entities.add(ent)

    def _adopt_link(self, link: "Link", other: "Struct") -> None:
        if getattr(link, "is_bound", False):
            snapshot = _snapshot_data(link)
            link._graph = None
            link._index = -1
            link.data = snapshot
        self.links.add(link)


class SpatialMixin:
    """Geometry operations on entities with a "xyz" key only."""

    entities: TypeBucket[Any]
    links: TypeBucket[Any]

    def move(self, delta: list[float], *, entity_type: type[Entity]) -> Self:
        """Translate all entities of the given type by a displacement vector.

        Args:
            delta: Translation vector [dx, dy, dz] in Angstrom.
            entity_type: Entity subclass to translate.

        Returns:
            Self for method chaining.
        """
        for e in self.entities.bucket(entity_type):
            e["x"] = e["x"] + delta[0]
            e["y"] = e["y"] + delta[1]
            e["z"] = e["z"] + delta[2]
        return self

    def rotate(
        self,
        axis: list[float],
        angle: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity],
    ) -> Self:
        """Rotate all entities of the given type around an axis.

        Uses Rodrigues' rotation formula.

        Args:
            axis: Rotation axis [ax, ay, az] (will be normalized).
            angle: Rotation angle in radians.
            about: Center of rotation [x, y, z] in Angstrom. Defaults to origin.
            entity_type: Entity subclass to rotate.

        Returns:
            Self for method chaining.
        """
        k = _unit(axis)
        o = [0.0, 0.0, 0.0] if about is None else about
        for e in self.entities.bucket(entity_type):
            xyz = [e["x"], e["y"], e["z"]]
            xyz = _rodrigues_rotate(xyz, k, angle, o)
            e["x"] = xyz[0]
            e["y"] = xyz[1]
            e["z"] = xyz[2]
        return self

    def scale(
        self,
        factor: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity],
    ) -> Self:
        """Scale positions of all entities of the given type.

        Args:
            factor: Scale factor (1.0 = no change).
            about: Center of scaling [x, y, z] in Angstrom. Defaults to origin.
            entity_type: Entity subclass to scale.

        Returns:
            Self for method chaining.
        """
        o = [0.0, 0.0, 0.0] if about is None else about
        for e in self.entities.bucket(entity_type):
            xyz = [e["x"], e["y"], e["z"]]
            xyz = _vec_sub(xyz, o)
            xyz = _vec_add(o, _vec_scale(xyz, factor))
            e["x"] = xyz[0]
            e["y"] = xyz[1]
            e["z"] = xyz[2]
        return self

    def align(
        self,
        a: Entity,
        b: Entity,
        *,
        a_dir: list[float] | None = None,
        b_dir: list[float] | None = None,
        flip: bool = False,
        entity_type: type[Entity],
    ) -> Self:
        """Align the struct so that entity a moves to entity b's position.

        Optionally rotates so that direction a_dir aligns with b_dir.

        Args:
            a: Source entity (must have x, y, z keys).
            b: Target entity (must have x, y, z keys).
            a_dir: Direction vector at source.
            b_dir: Direction vector at target.
            flip: Negate b_dir before aligning.
            entity_type: Entity subclass to transform.

        Returns:
            Self for method chaining.
        """
        pa = [a["x"], a["y"], a["z"]]
        pb = [b["x"], b["y"], b["z"]]
        if not (
            isinstance(pa, list)
            and isinstance(pb, list)
            and len(pa) == 3
            and len(pb) == 3
        ):
            return self  # silently skip if missing positions

        ents = self.entities.bucket(entity_type)

        # rotate if directions provided
        if a_dir is not None and b_dir is not None:
            va = _unit(a_dir)
            vb = _unit(b_dir)
            if flip:
                vb = _vec_scale(vb, -1.0)
            # axis = va x vb; angle = atan2(|axis|, dot)
            axis = _cross(va, vb)
            na = _norm(axis)
            if na > 0:
                # angle via sin/cos components
                from math import atan2

                angle = atan2(na, _dot(va, vb))
                for e in ents:
                    xyz = [e["x"], e["y"], e["z"]]
                    xyz = _rodrigues_rotate(xyz, _vec_scale(axis, 1.0 / na), angle, pa)
                    e["x"] = xyz[0]
                    e["y"] = xyz[1]
                    e["z"] = xyz[2]
        # translate so that a -> b
        delta = _vec_sub(pb, pa)
        self.move(delta, entity_type=entity_type)

        return self


class MembershipMixin:
    """CRUD operations for entities and links within a StructLike."""

    entities: TypeBucket[Any]
    links: TypeBucket[Any]

    def register_type(self, cls: type[Any]) -> None:
        """Register an entity type so its bucket exists (even if empty).

        Args:
            cls: Entity subclass to register.
        """
        self.entities._items.setdefault(cls, Entities())

    # Entities -------------------------------------------------------------
    def add_entity(self, *ents: Entity) -> None:
        """Add one or more entities to this struct.

        Args:
            ents: Entity instances to add.
        """
        for e in ents:
            self.entities.add(e)

    def remove_entity(self, *ents: Entity, drop_incident_links: bool = True) -> None:
        """Remove entities from this struct.

        Args:
            ents: Entity instances to remove.
            drop_incident_links: If True, also remove links that reference
                the removed entities.
        """
        to_remove = set(ents)
        # optionally drop incident links
        if drop_incident_links:
            # Convert to list to avoid RuntimeError: dictionary changed size during iteration
            for lcls in list(self.links.classes()):
                bucket = self.links.bucket(lcls)
                doomed: list[Link] = []
                for l in bucket:
                    if any(ep in to_remove for ep in l.endpoints):
                        doomed.append(l)
                if doomed:
                    self.remove_link(*doomed)
        # finally discard entities
        for e in ents:
            self.entities.remove(e)

    # Links ----------------------------------------------------------------
    def add_link(self, *links: Link, include_endpoints: bool = True) -> None:
        """Add one or more links to this struct.

        Args:
            links: Link instances to add.
            include_endpoints: If True, also add the link's endpoint
                entities if not already present.
        """
        for l in links:
            self.links.add(l)
            if include_endpoints:
                for ep in l.endpoints:
                    self.entities.add(ep)

    def remove_link(self, *links: Link) -> None:
        """Remove one or more links from this struct.

        Args:
            links: Link instances to remove.
        """
        for l in links:
            self.links.remove(l)

    # Normalize ------------------------------------------------------------
    def normalize(self, include_missing_endpoints: bool = False) -> None:
        """Clean up orphan links whose endpoints are not in this struct.

        Args:
            include_missing_endpoints: If True, add missing endpoints
                instead of removing the orphan links.
        """
        present: set[Entity] = set()
        for ecls in self.entities.classes():
            present.update(self.entities.bucket(ecls))
        for lcls in self.links.classes():
            bucket = self.links.bucket(lcls)
            doomed: list[Link] = []
            for l in bucket:
                missing = [ep for ep in l.endpoints if ep not in present]
                if missing:
                    if include_missing_endpoints:
                        for ep in missing:
                            self.entities.add(ep)
                            present.add(ep)
                    else:
                        doomed.append(l)
            if doomed:
                self.remove_link(*doomed)


class ConnectivityMixin:
    """Graph traversal operations for structs with entities and links."""

    entities: TypeBucket[Any]
    links: TypeBucket[Any]

    def get_neighbors(
        self, entity: Entity, link_type: type[Link] = Link
    ) -> list[Entity]:  # type: ignore[assignment]
        """Find all entities connected to the given entity via links.

        Args:
            entity: Entity to find neighbors of.
            link_type: Only consider links of this type.

        Returns:
            List of neighboring entities.
        """
        neighbors: list[Entity] = []
        try:
            bucket = self.links.bucket(link_type)  # type: ignore[arg-type]
        except KeyError:
            return neighbors
        for link in bucket:
            # Use identity check (is) not equality check (==)
            if any(ep is entity for ep in link.endpoints):
                for ep in link.endpoints:
                    if ep is not entity:
                        neighbors.append(ep)  # type: ignore[arg-type]
        return neighbors

    def get_topo(
        self,
        entity_type: type[Entity] = Entity,
        link_type: type[Link] = Link,
    ) -> "Topology":
        """Export structure as a topology graph.

        Args:
            entity_type: Entity type to include in the topology graph
            link_type: Link type used to build connections

        Returns:
            Topology: igraph.Graph object containing entity-to-vertex mapping information

        Note:
            The vertex order in the returned Topology graph is consistent with entities[entity_type].
            Use `entity_to_idx` and `idx_to_entity` attributes to access the mapping.
        """
        from molpy.core.topology import Topology

        # Collect all relevant entities
        entities_list = list(self.entities.bucket(entity_type))
        entity_to_idx: dict[Entity, int] = {
            ent: i for i, ent in enumerate(entities_list)
        }
        entity_set = set(entities_list)

        # Build edge list (only consider links connecting two endpoints)
        edges: list[tuple[int, int]] = []
        for link in self.links.bucket(link_type):  # type: ignore[arg-type]
            endpoints = link.endpoints
            # Only process links with two endpoints that are both in the entity set
            if len(endpoints) >= 2:
                ep1, ep2 = endpoints[0], endpoints[1]
                if ep1 in entity_set and ep2 in entity_set:
                    idx1 = entity_to_idx[ep1]
                    idx2 = entity_to_idx[ep2]
                    if idx1 != idx2:  # Avoid self-loops
                        edges.append((idx1, idx2))

        # Create topology graph, using formal members to store mapping
        topo = Topology(
            n=len(entities_list),
            edges=edges,
            directed=False,
            entity_to_idx=entity_to_idx,
            idx_to_entity=entities_list,
        )

        return topo

    def get_topo_neighbors(
        self,
        entity: Entity,
        radius: int = 1,
        entity_type: type[Entity] = Entity,
        link_type: type[Link] = Link,
    ) -> list[Entity]:
        """Get all neighbors of a specified entity within a given topological radius.

        Args:
            entity: Center entity
            radius: Topological radius (number of hops)
            entity_type: Entity type to consider
            link_type: Link type used for topological connections

        Returns:
            list[Entity]: List of all neighbor entities within radius (including self if radius>=0)
        """
        topo = self.get_topo(entity_type=entity_type, link_type=link_type)

        # Get entity index in graph
        entity_to_idx: dict[Entity, int] = topo.entity_to_idx
        if entity not in entity_to_idx:
            return []

        center_idx = entity_to_idx[entity]

        # Get distances
        distances = topo.distances(source=[center_idx])[0]

        # Collect all entities within radius
        neighbors: list[Entity] = []
        idx_to_entity: list[Entity] = topo.idx_to_entity
        for i, dist in enumerate(distances):
            if dist <= radius and dist < float("inf"):
                neighbors.append(idx_to_entity[i])

        return neighbors

    def get_topo_distances(
        self,
        source: Entity,
        entity_type: type[Entity] = Entity,
        link_type: type[Link] = Link,
    ) -> dict[Entity, int]:
        """Get topological distances from source entity to all other entities.

        Args:
            source: Source entity
            entity_type: Entity type to consider
            link_type: Link type used for topological connections

        Returns:
            dict[Entity, int]: Dictionary of topological distances from source to each entity.
                If an entity is unreachable, the distance is infinity (float('inf')).
        """
        topo = self.get_topo(entity_type=entity_type, link_type=link_type)

        # Get source entity index in graph
        entity_to_idx: dict[Entity, int] = topo.entity_to_idx
        if source not in entity_to_idx:
            return {}

        source_idx = entity_to_idx[source]

        # Calculate distances
        distances = topo.distances(source=[source_idx])[0]

        # Build distance dictionary
        idx_to_entity: list[Entity] = topo.idx_to_entity
        result: dict[Entity, int] = {}
        for i, dist in enumerate(distances):
            if dist < float("inf"):
                result[idx_to_entity[i]] = int(dist)

        return result

    def extract_subgraph(
        self,
        center_entities: Iterable[Entity],
        radius: int,
        entity_type: type[Entity] = Entity,
        link_type: type[Link] = Link,
    ) -> tuple["Struct", list[Entity]]:
        """Extract subgraph within specified topological radius.

        Args:
            center_entities: Set of center entities
            radius: Topological radius (number of hops)
            entity_type: Entity type to consider
            link_type: Link type used for topological connections

        Returns:
            tuple[Struct, list[Entity]]: Contains:
                - subgraph: Extracted subgraph (new Struct instance)
                - edge_entities: List of boundary entities (entities with neighbors in original graph but not in subgraph)

        Note:
            The subgraph contains all entities within radius and links between them.
            Boundary entities are those that have neighbors in the original graph not included in the subgraph.
        """
        center_entities_list = list(center_entities)
        topo = self.get_topo(entity_type=entity_type, link_type=link_type)

        entity_to_idx: dict[Entity, int] = topo.entity_to_idx
        idx_to_entity: list[Entity] = topo.idx_to_entity

        # Get indices of center entities
        center_indices: list[int] = []
        for cent in center_entities_list:
            if cent in entity_to_idx:
                center_indices.append(entity_to_idx[cent])

        if not center_indices:
            # If no valid center entities, return empty subgraph
            from copy import deepcopy

            new_struct = type(self)()
            return new_struct, []

        # Collect all entity indices within radius
        selected_indices: set[int] = set()
        for c in center_indices:
            distances = topo.distances(source=[c])[0]
            for i, d in enumerate(distances):
                if d <= radius and d < float("inf"):
                    selected_indices.add(i)

        selected_indices_list = sorted(selected_indices)
        selected_entities = [idx_to_entity[i] for i in selected_indices_list]
        selected_entities_set = set(selected_entities)

        # Find boundary entities (entities in subgraph with neighbors not in subgraph)
        selected_indices_set = set(selected_indices)
        edge_indices: set[int] = set()
        for i in selected_indices:
            for j in topo.neighbors(i):
                if j not in selected_indices_set:
                    edge_indices.add(i)
                    break

        edge_entities = [idx_to_entity[i] for i in sorted(edge_indices)]

        # Create new Struct instance
        from copy import deepcopy

        new_struct = type(self)()
        new_struct._props = deepcopy(self._props)

        # Add selected entities and create entity mapping (original entity -> cloned entity)
        entity_map: dict[Entity, Entity] = {}
        cloned_entities_list: list[Entity] = []
        for ent in selected_entities:
            # Deep copy entity
            cloned_ent = ent.__class__(deepcopy(_snapshot_data(ent)))
            new_struct.entities.add(cloned_ent)
            entity_map[ent] = cloned_ent
            cloned_entities_list.append(cloned_ent)

        # Add links in subgraph
        for link in self.links.bucket(link_type):  # type: ignore[arg-type]
            endpoints = link.endpoints
            if len(endpoints) >= 2:
                # Check if all link endpoints are in subgraph
                if all(ep in selected_entities_set for ep in endpoints):
                    # Create cloned link, mapping endpoints to new entities
                    cloned_eps = [entity_map[ep] for ep in endpoints]
                    attrs = deepcopy(_snapshot_data(link))
                    lcls: type[Link] = type(link)
                    try:
                        new_link = lcls(*cloned_eps, **attrs)
                    except TypeError:
                        new_link = lcls(cloned_eps, **attrs)
                    new_struct.links.add(new_link)

        # Update boundary entity list to cloned entities
        cloned_edge_entities = [
            entity_map[ep] for ep in edge_entities if ep in entity_map
        ]

        return new_struct, cloned_edge_entities
