"""Coarse-grained molecular structure as a handle-view over a molrs ``CoarseGrain``.

Mirrors :mod:`molpy.core.atomistic`: ``CoarseGrain(_GraphViews, molrs.CoarseGrain)``
IS a molrs world; :class:`Bead` / :class:`CGBond` are interned handle views.

Dict keys:

* ``bead["atoms"]`` — ``tuple[Atom, ...]`` of atom views this bead groups. This
  is the bead's **membership**, owned by the molrs ``CoarseGrain`` as opaque atom
  handles (not a scalar component) and resolved back to views through the source
  all-atom world. Drives :meth:`CoarseGrain.beads_of`.
* ``bead["x"]`` / ``bead["y"]`` / ``bead["z"]`` — position (molrs columns).
* ``bead["type"]`` / ``bead["mass"]`` / ``bead["charge"]`` — molrs columns.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self

import numpy as np

import molrs

from molrs.views import Bead, CGBond, Entities, Entity, Link, _GraphViews

if TYPE_CHECKING:
    from molrs import Frame

    from .atomistic import Atom


class CoarseGrain(molrs.CoarseGrain, _GraphViews):
    """Coarse-grained molecular structure backed by a molrs ``CoarseGrain``.

    Base order: the pyo3 native ``molrs.CoarseGrain`` must be first (see the note
    on :class:`molpy.core.atomistic.Atomistic`).
    """

    _node_cls = Bead
    _relation_classes = {"bonds": CGBond}
    _hidden_native_builders = frozenset({"add_bead", "add_bond"})

    def __getattribute__(self, name: str) -> Any:
        if name in object.__getattribute__(self, "_hidden_native_builders"):
            target = "def_bead" if name == "add_bead" else "def_cgbond"
            raise AttributeError(f"{name} is not public; use {target} instead")
        return super().__getattribute__(name)

    def __init__(self, **props: Any) -> None:
        _GraphViews.__init__(self, **props)
        # Source all-atom world that bead membership handles resolve against
        # (set on the first ``def_bead(atoms=...)``); membership itself lives in
        # the molrs ``CoarseGrain`` as opaque handles.
        self._member_world: Any = None
        if hasattr(self, "__post_init__"):
            for klass in type(self).__mro__:
                if klass is CoarseGrain:
                    break
                if "__post_init__" in klass.__dict__:
                    klass.__dict__["__post_init__"](self, **props)
                    break

    # ---------- collection views ----------
    @property
    def beads(self) -> Entities[Bead]:
        return self._node_views()  # type: ignore[return-value]

    @property
    def cgbonds(self) -> Entities[CGBond]:
        return self._relation_views("bonds")  # type: ignore[return-value]

    def __repr__(self) -> str:
        from collections import Counter

        types = Counter(b.get("type") or "?" for b in self.beads)
        if len(types) <= 5:
            comp = " ".join(f"{t}:{n}" for t, n in sorted(types.items()))
        else:
            comp = f"{len(types)} types"
        return f"<CoarseGrain, {len(self.beads)} beads ({comp}), {len(self.cgbonds)} bonds>"

    def __len__(self) -> int:
        return len(self.beads)

    # ---------- factory / add ----------
    def def_bead(self, mapping: Any = None, /, **attrs: Any) -> Bead:
        # "atoms" is membership (a tuple of atom views), not a scalar component:
        # split it out before the bead is spawned so it never hits a column.
        atoms = attrs.pop("atoms", None)
        if mapping is not None and "atoms" in mapping:
            mapping = dict(mapping)
            atoms = mapping.pop("atoms")
        bead = self._create_node(mapping, cls=Bead, **attrs)
        if atoms is not None:
            self._set_bead_atoms(bead, tuple(atoms))
        return bead  # type: ignore[return-value]

    # ---------- bead → atom membership (molrs-owned handle store) ----------
    def _set_bead_atoms(self, bead: Bead, atoms: tuple["Atom", ...]) -> None:
        """Record a bead's atom membership in the molrs world (by handle)."""
        handles: list[int] = []
        for a in atoms:
            world = a.world
            if self._member_world is None:
                self._member_world = world
            elif world is not self._member_world:
                raise ValueError(
                    "bead membership atoms must all come from the same source world"
                )
            handles.append(a.handle)
        molrs.CoarseGrain.set_bead_members(self, bead.handle, handles)

    def _resolve_bead_atoms(self, bead_handle: int) -> tuple["Atom", ...]:
        """Resolve a bead's stored membership handles back to atom views."""
        handles = molrs.CoarseGrain.bead_members(self, bead_handle)
        if not handles or self._member_world is None:
            return ()
        return tuple(self._member_world._intern_node(h) for h in handles)

    def def_cgbond(self, a: Bead, b: Bead, /, **attrs: Any) -> CGBond:
        return self._create_relation("bonds", (a, b), cls=CGBond, **attrs)  # type: ignore[return-value]

    def del_bead(self, *beads: Bead) -> None:
        for bead in beads:
            self._remove_node(bead)

    def remove_entity(self, *beads: Bead, drop_incident_links: bool = True) -> None:
        for bead in beads:
            self._remove_node(bead)

    def del_cgbond(self, *bonds: CGBond) -> None:
        for bond in bonds:
            self._remove_relation(bond)

    def remove_link(self, *links: Link) -> None:
        for link in links:
            self._remove_relation(link)

    def def_beads(self, beads_data: list[dict[str, Any]], /) -> list[Bead]:
        return [self.def_bead(**a) for a in beads_data]

    def def_cgbonds(self, bonds_data: list[Any], /) -> list[CGBond]:
        out: list[CGBond] = []
        for spec in bonds_data:
            if len(spec) == 2:
                a, b = spec
                attrs: dict[str, Any] = {}
            else:
                a, b, attrs = spec
            out.append(self.def_cgbond(a, b, **attrs))
        return out

    # ---------- reverse lookup ----------
    def beads_of(self, atom: "Atom") -> tuple[Bead, ...]:
        """Beads whose membership includes ``atom`` (molrs reverse lookup)."""
        handles = molrs.CoarseGrain.beads_of_atom(self, atom.handle)
        return tuple(self._intern_node(h) for h in handles)  # type: ignore[misc]

    # ---------- property / type / selection editing ----------
    def rename_type(self, old: str, new: str, *, kind: type = Bead) -> int:
        items = self._items_of_kind(kind)
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
        if not callable(selector):
            raise TypeError(
                "selector must be a callable (bead) -> bool; "
                "SMARTS-string selectors are not yet supported"
            )
        count = 0
        for item in self._items_of_kind(kind):
            if selector(item):
                item[key] = value
                count += 1
        return count

    def select(self, predicate: Callable[[Bead], bool]) -> "CoarseGrain":
        if not callable(predicate):
            raise TypeError(
                "predicate must be a callable (bead) -> bool; "
                "SMARTS-string predicates are not yet supported"
            )
        selected = [b for b in self.beads if predicate(b)]
        return self._subset(selected)

    def _items_of_kind(self, kind: type) -> Entities[Any]:
        if isinstance(kind, type) and issubclass(kind, Link):
            return self._relation_views("bonds")
        return self.beads  # type: ignore[return-value]

    def _subset(self, selected: list[Bead]) -> "CoarseGrain":
        selected_set = set(selected)
        new = type(self)()
        new._props = dict(self._props)
        bead_map: dict[Bead, Bead] = {}
        for bead in selected:
            bead_map[bead] = new.def_bead(dict(bead.data))
        for bond in self.cgbonds:
            if all(ep in selected_set for ep in bond.endpoints):
                mapped = [bead_map[ep] for ep in bond.endpoints]
                new.def_cgbond(*mapped, **dict(bond.data))
        return new

    # ---------- copy / merge ----------
    def copy(self) -> Self:
        """Independent deep copy. **Handles are preserved** (molrs clone)."""
        bare = molrs.CoarseGrain.copy(self)
        new = type(self)()
        molrs.CoarseGrain.adopt(new, bare)
        new._props = dict(self._props)
        new._member_world = self._member_world
        return new

    def merge(self, other: "CoarseGrain") -> Self:
        """Structural merge of ``other`` into ``self`` (molrs).

        Handles are remapped; ``other`` is emptied. View identity is not preserved.
        """
        molrs.CoarseGrain.merge(self, other)
        if self._member_world is None:
            self._member_world = other._member_world
        other._node_refs.clear()
        other._relation_refs.clear()
        other._props.clear()
        other._member_world = None
        return self

    @staticmethod
    def adopt(graph: molrs.CoarseGrain) -> "CoarseGrain":
        """Zero-copy take ownership of a molrs-produced ``CoarseGrain`` graph."""
        struct = CoarseGrain()
        molrs.CoarseGrain.adopt(struct, graph)
        return struct

    # ---------- spatial ----------
    def move(
        self, delta: list[float], *, entity_type: type[Entity] = Bead
    ) -> "CoarseGrain":
        molrs.translate(self, [float(d) for d in delta])
        return self

    def rotate(
        self,
        axis: list[float],
        angle: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity] = Bead,
    ) -> "CoarseGrain":
        o = [0.0, 0.0, 0.0] if about is None else list(about)
        molrs.rotate(self, axis, float(angle), o)
        return self

    def scale(
        self,
        factor: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity] = Bead,
    ) -> "CoarseGrain":
        o = [0.0, 0.0, 0.0] if about is None else list(about)
        molrs.scale(self, [factor, factor, factor], o)
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
        pa = [a["x"], a["y"], a["z"]]
        pb = [b["x"], b["y"], b["z"]]
        molrs.align_direction(self, pa, pb, a_dir, b_dir, flip)
        return self

    # ---------- composition ----------
    def __iadd__(self, other: "CoarseGrain") -> "CoarseGrain":
        self.merge(other)
        return self

    def __add__(self, other: "CoarseGrain") -> "CoarseGrain":
        result = self.copy()
        result.merge(other.copy())  # merge empties its argument
        return result

    def replicate(
        self, n: int, transform: Callable[["CoarseGrain", int], None] | None = None
    ) -> "CoarseGrain":
        if transform is None:
            return type(self).adopt(molrs.CoarseGrain.replicate(self, n))
        result = type(self)()
        for i in range(n):
            replica = self.copy()
            if transform is not None:
                transform(replica, i)
            result.merge(replica)
        return result

    # ---------- tabular conversion ----------
    def to_frame(self, bead_fields: list[str] | None = None) -> "Frame":
        """Export to a tabular :class:`Frame` (``beads`` + ``cgbonds`` blocks).

        Delegates straight to the molrs world's native ``to_frame``: the Rust
        column store yields dense numpy columns and applies the CG-domain block /
        column labels (``beads`` / ``cgbonds`` / ``ibead`` / ``jbead``) itself, so
        there is zero Python-side conversion. ``bead_fields`` optionally restricts
        the beads block columns.
        """
        from molrs import Frame

        # Upgrade the bare pyo3 frame to the rich ``Frame`` callers expect.
        frame = Frame.from_dict(molrs.CoarseGrain.to_frame(self))
        if bead_fields is not None and "beads" in frame:
            keep = set(bead_fields)
            beads = frame["beads"]
            for col in [k for k in beads.keys() if k not in keep]:
                del beads[col]
        return frame
