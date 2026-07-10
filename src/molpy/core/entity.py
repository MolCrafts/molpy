"""Handle-view structure layer over a molrs ECS world.

molpy's structure containers (:class:`~molpy.core.atomistic.Atomistic`,
:class:`~molpy.core.cg.CoarseGrain`) *are* molrs leaves — they subclass
``molrs.Atomistic`` / ``molrs.CoarseGrain`` and the molrs world is the single
source of truth. There is no mirror, no index shifting, no ``Struct`` container.

* **Entity / Link are handle views** holding ``(world, handle)``. Property reads
  and writes route straight through the molrs component columns via the
  :class:`_NodeProxy` / :class:`_LinkProxy` live mappings below, decomposing
  ``"xyz"`` to the ``x`` / ``y`` / ``z`` columns. There is no Python-side storage:
  a value molrs cannot represent (not bool/int/float/str) raises at the boundary.
* **Identity is preserved by interning**: a :class:`weakref.WeakValueDictionary`
  keyed by handle returns the same view object for the same handle, so
  ``bond.itom is atom``, ``atom in s.atoms`` and ``hash(atom)`` hold while a
  reference is alive. A handle whose node is despawned is evicted lazily.
* **Removal is stable**: ``world.despawn(handle)`` plus intern eviction; no
  reindex, surviving handles and relations stay valid.

The shared base :class:`_GraphViews` provides interning, the ``def_*`` skeleton
and the ``.atoms`` / ``.bonds`` view surface. The atomistic / CG leaves add their
domain view types.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableMapping
from copy import deepcopy
from typing import Any, Protocol, Self, overload
from weakref import WeakValueDictionary

import molrs
import numpy as np

from molpy.core import fields
from molrs.fields import FieldSpec


# ===================================================================
#                       Entity / Link views
# ===================================================================


class EntityLike(Protocol):
    """Protocol for objects that can act as Entities (Entity or subclass)."""

    data: Any

    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, item: Any) -> None: ...
    def __hash__(self) -> int: ...
    def get(self, key: str, default: Any = None) -> Any: ...


class _DictView:
    """Common dict-style surface shared by :class:`Entity` and :class:`Link`.

    ``self.data`` is either a plain ``dict`` (pending, no world yet) or a live
    :class:`_NodeProxy` / :class:`_LinkProxy` (bound). Every dict operation
    funnels through ``self.data`` so the same surface works in both states.
    Identity-based hashing/equality is kept (views are interned).
    """

    data: Any

    @staticmethod
    def _key(key: Any) -> Any:
        """Accept a canonical :class:`~molrs.fields.FieldSpec` wherever a column
        name is expected, so call sites name fields through the registry
        (``atom[fields.SITE]``) instead of repeating a string literal.
        """
        return key.key if isinstance(key, FieldSpec) else key

    def __getitem__(self, key: Any) -> Any:
        return self.data[self._key(key)]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[self._key(key)] = value

    def __delitem__(self, key: Any) -> None:
        del self.data[self._key(key)]

    def __contains__(self, key: object) -> bool:
        return self._key(key) in self.data

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def keys(self) -> Any:
        return self.data.keys()

    def values(self) -> Any:
        return self.data.values()

    def items(self) -> Any:
        return self.data.items()

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self.data[self._key(key)]
        except KeyError:
            return default

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.data.update(*args, **kwargs)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self.data:
            return self.data[key]
        self.data[key] = default
        return default

    def pop(self, key: str, *default: Any) -> Any:
        return self.data.pop(key, *default)

    def __hash__(self) -> int:  # identity-based
        return id(self)

    def __eq__(self, other: object) -> bool:  # identity-based
        return self is other

    def __ne__(self, other: object) -> bool:
        return self is not other


# ===================================================================
#            Live handle proxies (Entity/Link .data backing)
# ===================================================================

# Component keys that compose the decomposed coordinate vector.
_XYZ_KEYS = (fields.POS_X.key, fields.POS_Y.key, fields.POS_Z.key)
_XYZ = fields.XYZ.key


class _NodeProxy(MutableMapping):
    """Live ``MutableMapping`` over one molrs node's component columns.

    Holds ``(world, handle)`` and routes every read/write straight onto the
    molrs world by stable handle. No Python-side storage and no field-dtype
    policy: a value molrs cannot represent (not bool/int/float/str) raises at the
    boundary, canonical fields are coerced to their dtype by molrs itself, and
    key enumeration comes from molrs (``world.node_keys``), not a shadow set.
    """

    __slots__ = ("_world", "_handle")

    def __init__(self, world: Any, handle: int) -> None:
        self._world = world
        self._handle = handle

    # ----- molrs column accessors (overridden by _LinkProxy) -----
    def _col_keys(self) -> list[str]:
        return list(self._world.node_keys(self._handle))

    def _col_has(self, key: str) -> bool:
        return self._world.has(self._handle, key)

    def _col_get(self, key: str) -> Any:
        return self._world.get(self._handle, key)

    def _col_set(self, key: str, value: Any) -> None:
        self._world.set(self._handle, key, value)

    def _col_del(self, key: str) -> None:
        self._world.delete(self._handle, key)

    # ----- MutableMapping protocol -----
    def __getitem__(self, key: str) -> Any:
        if key == _XYZ:
            return self._get_xyz()
        if self._col_has(key):
            return self._col_get(key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == _XYZ:
            self._set_xyz(value)
            return
        if value is None:
            # ``None`` is the absence marker in a sparse component store, not a
            # storable value: setting it clears the key (no Python-side stash).
            if self._col_has(key):
                self._col_del(key)
            return
        # molrs coerces canonical fields to their dtype and raises if ``value``
        # is not numpy-representable.
        self._col_set(key, value)

    def __delitem__(self, key: str) -> None:
        if not self._col_has(key):
            raise KeyError(key)
        self._col_del(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._col_keys())

    def __len__(self) -> int:
        return len(self._col_keys())

    def __contains__(self, key: object) -> bool:
        if key == _XYZ:
            return all(self._col_has(k) for k in _XYZ_KEYS)
        return isinstance(key, str) and self._col_has(key)

    # ----- derived "xyz" vector over the x/y/z columns -----
    def _get_xyz(self) -> list[float]:
        return [self._col_get(k) for k in _XYZ_KEYS]

    def _set_xyz(self, value: Any) -> None:
        if len(value) != 3:
            raise ValueError(f"xyz must be a 3-vector, got {value!r}")
        for k, v in zip(_XYZ_KEYS, value):
            self._col_set(k, float(v))

    def __repr__(self) -> str:
        return repr(dict(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (MutableMapping, dict)):
            return dict(self) == dict(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    __hash__ = None  # type: ignore[assignment]

    def copy(self) -> dict[str, Any]:
        return dict(self)

    def __deepcopy__(self, memo: dict) -> dict[str, Any]:
        """Snapshot to a plain ``dict`` (never deep-copy the molrs world)."""
        import copy as _copy

        return {
            _copy.deepcopy(k, memo): _copy.deepcopy(v, memo) for k, v in self.items()
        }


class _LinkProxy(_NodeProxy):
    """Live ``MutableMapping`` over one molrs relation's property columns."""

    __slots__ = ("_kind",)

    def __init__(self, world: Any, kind: str, handle: int) -> None:
        super().__init__(world, handle)
        self._kind = kind

    def _col_keys(self) -> list[str]:
        return list(self._world.relation_keys(self._kind, self._handle))

    def _col_has(self, key: str) -> bool:
        return self._world.get_relation_prop(self._kind, self._handle, key) is not None

    def _col_get(self, key: str) -> Any:
        return self._world.get_relation_prop(self._kind, self._handle, key)

    def _col_set(self, key: str, value: Any) -> None:
        self._world.set_relation_prop(self._kind, self._handle, key, value)

    def _col_del(self, key: str) -> None:
        self._world.delete_relation_prop(self._kind, self._handle, key)


class Entity(_DictView):
    """Dict-like view of one node in a molrs world (or a pending node).

    Constructed standalone it is *pending*: ``self.data`` is a plain dict it owns
    and ``self._world`` is ``None``. Once a container's ``def_*`` / ``add_*``
    spawns a node for it, :meth:`_attach` flushes the dict into the world and
    swaps ``self.data`` for a live column proxy — the same object becomes a bound
    view, so identity (``bond.itom is atom``) survives.

    Accepts both forms::

        Atom(symbol="C", xyz=[0, 0, 0])   # kwargs
        Atom({"symbol": "C", "xyz": ...}) # positional mapping
    """

    def __init__(self, mapping: Any = None, /, **attrs: Any) -> None:
        self._world: Any = None
        self._handle: int = -1
        data: dict[str, Any] = {}
        if mapping is not None:
            data.update(mapping)
        data.update(attrs)
        self.data = data

    # ----- binding (called by the owning container) -----
    @property
    def is_bound(self) -> bool:
        return self._world is not None

    @property
    def handle(self) -> int:
        return self._handle

    def _attach(self, world: Any, handle: int) -> None:
        """Bind this view to ``world`` node ``handle``, flushing pending data."""
        pending = dict(self.data)
        self._world = world
        self._handle = handle
        proxy = _NodeProxy(world, handle)
        self.data = proxy
        for k, v in pending.items():
            proxy[k] = v

    def _detach(self) -> None:
        """Snapshot live data back to a plain dict and unbind (for copy/merge)."""
        if self.is_bound:
            snapshot = dict(self.data)
            self._world = None
            self._handle = -1
            self.data = snapshot

    def __deepcopy__(self, memo: dict) -> "Entity":
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        clone._world = None
        clone._handle = -1
        clone.data = deepcopy(dict(self.data), memo)
        return clone

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {dict(self.data)}>"


class LinkLike(Protocol):
    """Protocol for objects that can act as Links (Link or subclass)."""

    data: Any
    endpoints: tuple[EntityLike, ...]

    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, item: Any) -> None: ...
    def __hash__(self) -> int: ...
    def get(self, key: str, default: Any = None) -> Any: ...


class Link[T: Entity](_DictView):
    """Dict-like view of one relation in a molrs world (or a pending relation).

    Holds ordered endpoint :class:`Entity` views in ``self.endpoints``; when
    bound, endpoints are resolved from ``world.relation_nodes`` so they are the
    interned endpoint views.
    """

    #: canonical molrs relation kind name — overridden per concrete subclass and
    #: asserted to exist in the world's registered kinds when bound.
    _kind: str = "bonds"

    def __init__(self, endpoints: Iterable[T], /, **attrs: Any) -> None:
        self._world: Any = None
        self._handle: int = -1
        self.endpoints = tuple(endpoints)
        self.data = dict(attrs)

    @property
    def is_bound(self) -> bool:
        return self._world is not None

    @property
    def handle(self) -> int:
        return self._handle

    def _attach(self, world: Any, handle: int, endpoints: tuple[T, ...]) -> None:
        pending = dict(self.data)
        self._world = world
        self._handle = handle
        self.endpoints = endpoints
        proxy = _LinkProxy(world, self._kind, handle)
        self.data = proxy
        for k, v in pending.items():
            proxy[k] = v

    def _detach(self) -> None:
        if self.is_bound:
            snapshot = dict(self.data)
            self._world = None
            self._handle = -1
            self.data = snapshot

    def replace_endpoint(self, old: T, new: T) -> None:
        """Replace one endpoint reference with another in-place."""
        self.endpoints = tuple(new if e is old else e for e in self.endpoints)

    def __deepcopy__(self, memo: dict) -> "Link":
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        clone._world = None
        clone._handle = -1
        clone.endpoints = tuple(deepcopy(ep, memo) for ep in self.endpoints)
        clone.data = deepcopy(dict(self.data), memo)
        return clone

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.endpoints}>"


# ===================================================================
#               Entities — column-accessible view sequence
# ===================================================================


class Entities[E: Entity](list[E]):
    """A list of entity/link views supporting column access by string key.

    ``ents["symbol"]`` returns a numpy array (one value per item, ``None`` where
    unset); ``ents["xyz"]`` returns a numpy array of 3-vectors. Integer / slice
    indexing behave as a normal list.
    """

    @overload
    def __getitem__(self, key: int) -> E: ...  # type: ignore[override]
    @overload
    def __getitem__(self, key: slice) -> list[E]: ...  # type: ignore[override]
    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: int | slice | str) -> Any:  # type: ignore[override]
        if isinstance(key, str):
            values = [ent.get(key) for ent in self]
            # Let numpy infer the dtype (float/int/str) so downstream math works;
            # fall back to object dtype only for ragged / None-bearing columns
            # (missing fields, vector "xyz" values).
            try:
                return np.array(values)
            except (ValueError, TypeError):
                return np.array(values, dtype=object)
        return super().__getitem__(key)


# ===================================================================
#               _GraphViews — shared thin base
# ===================================================================


class _GraphViews:
    """Shared base for molpy structure leaves over a molrs world.

    Subclassed alongside a molrs leaf (``Atomistic(_GraphViews, molrs.Atomistic)``)
    so ``self`` *is* the molrs world. Owns the per-world interning tables and the
    Python overflow store; provides the ``def_*`` skeleton and the column-key
    bookkeeping the proxies need.
    """

    # set by concrete leaves: the Entity view class and per-kind Link classes
    _entity_cls: type[Entity]
    _link_classes: dict[str, type[Link]]

    def __init__(self, **props: Any) -> None:
        # molrs leaf __init__ takes no required args; the base PyGraph __new__
        # already built the world. Initialise molpy-side bookkeeping.
        self._props: dict[str, Any] = dict(props)
        self._atom_intern: WeakValueDictionary[int, Entity] = WeakValueDictionary()
        self._link_intern: dict[str, WeakValueDictionary[int, Link]] = {}

    # ---------- props (struct-level dict surface) ----------
    def __getitem__(self, key: str) -> Any:
        return self._props[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._props[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._props

    def get(self, key: str, default: Any = None) -> Any:
        return self._props.get(key, default)

    # ---------- interning ----------
    def _intern_atom(self, handle: int) -> Entity:
        view = self._atom_intern.get(handle)
        if view is None:
            view = self._entity_cls.__new__(self._entity_cls)
            view._world = self
            view._handle = handle
            view.data = _NodeProxy(self, handle)
            self._atom_intern[handle] = view
        return view

    def _intern_link(self, kind: str, rh: int) -> Link:
        table = self._link_intern.setdefault(kind, WeakValueDictionary())
        view = table.get(rh)
        if view is None:
            cls = self._link_classes[kind]
            view = cls.__new__(cls)
            view._world = self
            view._handle = rh
            view._kind = kind
            nodes = self.relation_nodes(kind, rh)
            view.endpoints = tuple(self._intern_atom(h) for h in nodes)
            view.data = _LinkProxy(self, kind, rh)
            table[rh] = view
        return view

    # ---------- entity spawning ----------
    def _spawn_entity(self, view: Entity) -> Entity:
        """Spawn a node for a pending ``view`` and bind/intern it.

        ``_attach`` replays pending props through the proxy, which routes each to
        a molrs column (noting the key) or the overflow store.
        """
        if view.is_bound:
            return view
        handle = self.spawn()
        view._attach(self, handle)
        self._atom_intern[handle] = view
        return view

    def _spawn_link(self, kind: str, view: Link) -> Link:
        if view.is_bound:
            return view
        handles = [ep.handle for ep in view.endpoints]
        rh = self.add_relation(kind, handles)
        view._attach(self, rh, view.endpoints)
        self._link_intern.setdefault(kind, WeakValueDictionary())[rh] = view
        return view

    # ---------- iteration views ----------
    def _node_handles(self) -> list[int]:
        """Live node handles via the leaf's own molrs ``entities()`` method.

        ``molrs.Graph.entities`` reads the dead base ``PyGraph`` storage, so the
        *leaf* type's generated method (operating on the leaf's own world) must
        be used. ``entities`` is a pyo3 method, not shadowed by Python here.
        """
        return self.entities()  # leaf molrs method

    def _atom_views(self) -> Entities[Entity]:
        return Entities(self._intern_atom(h) for h in self._node_handles())

    def _link_views(self, kind: str) -> Entities[Link]:
        # molrs is the single source of truth for live relation handles
        # (enumerated in row order). Unregistered kinds simply have no links.
        if kind not in self.kinds():
            return Entities()
        return Entities(self._intern_link(kind, rh) for rh in self.relation_ids(kind))

    def _all_link_views(self) -> Entities[Link]:
        out: Entities[Link] = Entities()
        for kind in self._link_classes:
            out.extend(self._link_views(kind))
        return out

    # ---------- compat .links bucket view ----------
    # ``.entities`` would collide with the pyo3 ``entities()`` method, so the
    # node bucket compat is not exposed; downstream uses ``.atoms`` / ``.beads``.
    # ``.links`` has no pyo3 collision and stays as a compat surface used by the
    # typifier / assembler.
    @property
    def links(self) -> "_LinkBucketView":
        return _LinkBucketView(self)

    # ---------- removal ----------
    def _remove_atom(self, view: Entity) -> None:
        h = view.handle
        # drop incident relations first so their interned views clear
        # (snapshot the handles since _remove_link mutates the molrs relations)
        for kind in self.kinds():
            for rh in list(self.relation_ids(kind)):
                if h in self.relation_nodes(kind, rh):
                    self._remove_link(self._intern_link(kind, rh))
        # snapshot the view's data to a plain dict so a caller holding a
        # reference to a removed atom can still read its (frozen) attributes.
        view._detach()
        self.despawn(h)
        self._atom_intern.pop(h, None)

    def _remove_link(self, view: Link) -> None:
        kind = view._kind
        rh = view.handle
        view._detach()
        self.remove_relation(kind, rh)
        table = self._link_intern.get(kind)
        if table is not None:
            table.pop(rh, None)


# ===================================================================
#       Compat .links bucket view over the world's relation handles
# ===================================================================


class _LinkBucketView:
    """Backward-compatible ``.links`` surface over a :class:`_GraphViews`."""

    def __init__(self, world: _GraphViews) -> None:
        self._world = world

    def _all(self) -> Entities[Any]:
        return self._world._all_link_views()

    def _kind_for(self, cls: type) -> str | None:
        for kind, link_cls in self._world._link_classes.items():
            if issubclass(link_cls, cls) or issubclass(cls, link_cls):
                return kind
        return None

    def add(self, item: Link) -> None:
        self._world._spawn_link(item._kind, item)

    def add_many(self, items: Iterable[Link]) -> None:
        for it in items:
            self._world._spawn_link(it._kind, it)

    def remove(self, *items: Link) -> None:
        for item in items:
            if getattr(item, "is_bound", False) and item._world is self._world:
                self._world._remove_link(item)

    def register_type(self, cls: type) -> None:  # no-op
        pass

    def all(self) -> Entities[Any]:
        return self._all()

    def bucket(self, cls: type) -> Entities[Any]:
        return Entities(link for link in self._all() if isinstance(link, cls))

    def exact_bucket(self, cls: type) -> Entities[Any]:
        return Entities(link for link in self._all() if type(link) is cls)

    def classes(self) -> Iterator[type]:
        seen: dict[type, None] = {}
        for link in self._all():
            seen.setdefault(type(link), None)
        return iter(seen)

    def __getitem__(self, cls: type) -> Entities[Any]:
        return self.bucket(cls)

    def __len__(self) -> int:
        return len(self._all())

    def __iter__(self) -> Iterator[Any]:
        return iter(self._all())
