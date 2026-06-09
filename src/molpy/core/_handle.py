"""Live property views backing handle-based :class:`~molpy.core.entity.Entity`
and :class:`~molpy.core.entity.Link` views over a molrs ECS world.

A bound entity/link no longer owns its own ``dict``: its ``.data`` is one of the
:class:`MutableMapping` proxies below, which route every read/write onto the
molrs world by **stable handle** (no index shifting). Because ``Entity`` /
``Link`` funnel their dict surface (``[]``, ``get``, ``keys``, ``update``, ``in``
…) through ``self.data``, swapping ``.data`` for a proxy transparently makes the
whole public surface graph-backed.

Storage policy
--------------
* **scalar** ``int`` / ``float`` / ``str`` (not ``bool``): stored as a molrs
  component column via ``world.set(handle, key, value)`` — the zero-copy hot
  path. ``bool`` would be coerced to a float by molrs, so it is kept Python-side
  to preserve observable type.
* **``"xyz"``**: decomposed into the molrs ``x`` / ``y`` / ``z`` columns. Reading
  ``"xyz"`` recomposes a 3-vector; writing a 3-sequence fans out to the columns.
* **everything else** (lists, tuples, objects, ``None``, ``bool``): kept in a
  per-handle Python *overflow* dict owned by the container, so arbitrary props
  round-trip. Reads merge columns with overflow, overflow shadowing on collision.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Any

from molpy.core import fields

# Component keys that make up the decomposed coordinate vector.
_XYZ_KEYS = (fields.POS_X.key, fields.POS_Y.key, fields.POS_Z.key)
_XYZ = fields.XYZ.key

# Canonical f64 columns: writes are coerced to ``float`` so an int literal does
# not type the molrs column ``i32`` and reject a later float write (coordinates
# from move/rotate/scale; aromatic bond orders 1.5 after integer order 1; etc).
_FLOAT_KEYS = frozenset(
    {
        fields.POS_X.key,
        fields.POS_Y.key,
        fields.POS_Z.key,
        fields.ORDER.key,
        fields.CHARGE.key,
        fields.MASS.key,
    }
)


def _graph_storable(value: Any) -> bool:
    """Return ``True`` when ``value`` may live in a molrs component column.

    ``bool`` is a subclass of ``int`` but molrs would coerce ``True`` to ``1.0``;
    keep it Python-side so the observed type is preserved.
    """
    return isinstance(value, (int, float, str)) and not isinstance(value, bool)


class _NodeProxy(MutableMapping):
    """MutableMapping view over one molrs node's columns + Python overflow.

    Holds ``(world, handle)`` and reads/writes the per-handle overflow dict from
    the owning container's ``_overflow`` store.
    """

    __slots__ = ("_world", "_handle")

    def __init__(self, world: Any, handle: int) -> None:
        self._world = world
        self._handle = handle

    # ----- overflow store (owned by the container) -----
    @property
    def _overflow(self) -> dict[str, Any]:
        return self._world._overflow_for(self._handle)

    # ----- molrs column accessors (overridden by _LinkProxy) -----
    def _col_keys(self) -> list[str]:
        return list(self._world._node_keys(self._handle))

    def _col_has(self, key: str) -> bool:
        return self._world.has(self._handle, key)

    def _col_get(self, key: str) -> Any:
        return self._world.get(self._handle, key)

    def _col_set(self, key: str, value: Any) -> None:
        self._world.set(self._handle, key, value)
        self._world._note_column_key(key)

    def _col_del(self, key: str) -> None:
        self._world.delete(self._handle, key)

    # ----- MutableMapping protocol -----
    def __getitem__(self, key: str) -> Any:
        if key == _XYZ:
            return self._get_xyz()
        overflow = self._overflow
        if key in overflow:
            return overflow[key]
        if self._col_has(key):
            return self._col_get(key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == _XYZ:
            self._set_xyz(value)
            return
        overflow = self._overflow
        if (
            key in _FLOAT_KEYS
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            # canonical f64 column: coerce so a later float write is accepted.
            overflow.pop(key, None)
            self._col_set(key, float(value))
        elif _graph_storable(value):
            overflow.pop(key, None)
            self._col_set(key, value)
        else:
            if self._col_has(key):
                self._col_del(key)
            overflow[key] = value

    def __delitem__(self, key: str) -> None:
        existed = False
        overflow = self._overflow
        if key in overflow:
            del overflow[key]
            existed = True
        if self._col_has(key):
            self._col_del(key)
            existed = True
        if not existed:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for k in self._overflow:
            seen.add(k)
            yield k
        for k in self._col_keys():
            if k not in seen:
                yield k

    def __len__(self) -> int:
        return len(set(self._overflow) | set(self._col_keys()))

    def __contains__(self, key: object) -> bool:
        if key == _XYZ:
            return all(self._col_has(k) for k in _XYZ_KEYS)
        return key in self._overflow or (isinstance(key, str) and self._col_has(key))

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
    """MutableMapping view over one molrs relation's props + Python overflow."""

    __slots__ = ("_kind",)

    def __init__(self, world: Any, kind: str, handle: int) -> None:
        super().__init__(world, handle)
        self._kind = kind

    @property
    def _overflow(self) -> dict[str, Any]:
        return self._world._overflow_for_relation(self._kind, self._handle)

    def _col_keys(self) -> list[str]:
        # molrs relation props expose no key listing; the overflow is the only
        # Python-visible namespace, and column props are tracked there too via
        # a shadow set kept by the world. We re-read tracked keys.
        return list(self._world._relation_keys(self._kind, self._handle))

    def _col_has(self, key: str) -> bool:
        return self._world.get_relation_prop(self._kind, self._handle, key) is not None

    def _col_get(self, key: str) -> Any:
        return self._world.get_relation_prop(self._kind, self._handle, key)

    def _col_set(self, key: str, value: Any) -> None:
        self._world.set_relation_prop(self._kind, self._handle, key, value)
        self._world._track_relation_key(self._kind, self._handle, key)

    def _col_del(self, key: str) -> None:
        # molrs has no relation-prop delete; drop from the tracked set so it
        # stops surfacing (the stale value is harmless once untracked).
        self._world._untrack_relation_key(self._kind, self._handle, key)

    def __contains__(self, key: object) -> bool:
        return key in self._overflow or (
            isinstance(key, str) and key in self._col_keys()
        )
