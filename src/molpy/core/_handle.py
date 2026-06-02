"""Live property proxies that back :class:`~molpy.core.entity.Entity` and
:class:`~molpy.core.entity.Link` handles with a molrs graph.

Part of the P1 migration (spec ``atomistic-cg-on-molrs-molgraph``): once an
:class:`Entity` / :class:`Link` is *bound* to a :class:`molrs.Graph`, its
``.data`` mapping is swapped for one of the proxies below.  Because
``Entity`` / ``Link`` are :class:`collections.UserDict` subclasses and every
``UserDict`` method funnels through ``self.data``, swapping ``.data`` for a
:class:`~collections.abc.MutableMapping` proxy transparently routes the whole
dict-style public surface (``[]``, ``get``, ``keys``, ``update``,
``in``, ``==`` …) onto the graph with no per-method overrides.

D5 attribute policy
-------------------
molrs ``PropValue`` only accepts ``int`` / ``float`` / ``str``.  Anything else
(``None``, ``bool``, ``tuple``, ``list``, objects, …) is kept in a per-handle
Python *fallback* dict.  ``bool`` is deliberately excluded from the graph even
though molrs would coerce it to ``int`` — round-tripping ``True`` as ``1``
would change observable behaviour.  Reads merge graph props with the fallback,
the fallback shadowing the graph on key collision.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Any


def _graph_storable(value: Any) -> bool:
    """Return True when ``value`` may live in a molrs property bag (D5)."""
    # bool is a subclass of int but molrs would coerce True -> 1; keep it Python-side.
    return isinstance(value, (int, float, str)) and not isinstance(value, bool)


class _AtomPropProxy(MutableMapping):
    """MutableMapping view over one graph atom's property bag + Python fallback."""

    __slots__ = ("_graph", "_handle", "_fallback")

    def __init__(self, graph: Any, handle: Any) -> None:
        self._graph = graph
        self._handle = handle
        self._fallback: dict[str, Any] = {}

    # ----- index resolution (indices compact on removal) -----
    @property
    def _idx(self) -> int:
        return self._handle._index

    # ----- graph backend accessors (overridden per entity kind) -----
    def _g_keys(self) -> list[str]:
        return list(self._graph.atom_keys(self._idx))

    def _g_get(self, key: str) -> Any:
        return self._graph.get_atom_prop(self._idx, key)

    def _g_set(self, key: str, value: Any) -> None:
        self._graph.set_atom_prop(self._idx, key, value)

    def _g_del(self, key: str) -> None:
        self._graph.del_atom_prop(self._idx, key)

    # ----- MutableMapping protocol -----
    def __getitem__(self, key: str) -> Any:
        if key in self._fallback:
            return self._fallback[key]
        if key in self._g_keys():
            return self._g_get(key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if _graph_storable(value):
            # value goes to graph; make sure no stale fallback shadows it.
            if key in self._fallback:
                del self._fallback[key]
            self._g_set(key, value)
        else:
            # value stays Python-side; drop any graph copy so reads see fallback.
            if key in self._g_keys():
                self._g_del(key)
            self._fallback[key] = value

    def __delitem__(self, key: str) -> None:
        existed = False
        if key in self._fallback:
            del self._fallback[key]
            existed = True
        if key in self._g_keys():
            self._g_del(key)
            existed = True
        if not existed:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        seen = set()
        for k in self._fallback:
            seen.add(k)
            yield k
        for k in self._g_keys():
            if k not in seen:
                yield k

    def __len__(self) -> int:
        return len(set(self._fallback) | set(self._g_keys()))

    def __contains__(self, key: object) -> bool:
        return key in self._fallback or key in self._g_keys()

    def __repr__(self) -> str:
        return repr(dict(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MutableMapping) or isinstance(other, dict):
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
        """Deep-copy to a plain ``dict`` snapshot.

        The proxy holds a reference to the (unpicklable) molrs graph, so a naive
        deepcopy would try to copy the whole graph. Callers that ``deepcopy`` an
        entity's ``.data`` want an independent attribute dict, not a live proxy —
        return exactly that.
        """
        import copy as _copy

        return {
            _copy.deepcopy(k, memo): _copy.deepcopy(v, memo) for k, v in self.items()
        }


class _LinkPropProxy(_AtomPropProxy):
    """MutableMapping view over one graph link (bond/angle/dihedral/improper)."""

    __slots__ = ()

    def _g_keys(self) -> list[str]:
        return list(self._handle._kind_keys(self._graph, self._idx))

    def _g_get(self, key: str) -> Any:
        return self._handle._kind_get(self._graph, self._idx, key)

    def _g_set(self, key: str, value: Any) -> None:
        self._handle._kind_set(self._graph, self._idx, key, value)

    def _g_del(self, key: str) -> None:
        self._handle._kind_del(self._graph, self._idx, key)
