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

from .entity import Entities, Entity, Link, _GraphViews

if TYPE_CHECKING:
    from .atomistic import Atom
    from .frame import Frame


class Bead(Entity):
    """Coarse-grained bead view: one node in a molrs ``CoarseGrain`` world.

    The ``"atoms"`` key is not a scalar component: it is the bead's **membership**
    (the underlying atom views it groups), owned by the molrs ``CoarseGrain`` as
    opaque atom handles and resolved back to views through the source world. It
    is intercepted here so ``bead["atoms"]`` / ``bead.get("atoms")`` keep working
    over the membership store rather than the component columns.
    """

    __slots__ = ()

    def __getitem__(self, key: str) -> Any:
        if key == "atoms" and self.is_bound:
            return self._world._resolve_bead_atoms(self._handle)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "atoms" and self.is_bound:
            return self._world._resolve_bead_atoms(self._handle)
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        if key == "atoms" and self.is_bound:
            return bool(self._world._resolve_bead_atoms(self._handle))
        return super().__contains__(key)

    def __repr__(self) -> str:
        ident = self.get("type") or self.get("name")
        return f"<Bead: {ident if ident is not None else id(self)}>"


class CGBond(Link["Bead"]):
    """Coarse-grained bond between two beads (molrs relation kind ``bonds``)."""

    __slots__ = ()
    _kind = "bonds"

    def __init__(self, a: Bead, b: Bead, /, **attrs: Any) -> None:
        assert isinstance(a, Bead), f"a must be Bead, got {type(a)}"
        assert isinstance(b, Bead), f"b must be Bead, got {type(b)}"
        super().__init__([a, b], **attrs)

    def __repr__(self) -> str:
        return f"<CGBond: {self.ibead} - {self.jbead}>"

    @property
    def ibead(self) -> Bead:
        return self.endpoints[0]

    @property
    def jbead(self) -> Bead:
        return self.endpoints[1]


class CoarseGrain(molrs.CoarseGrain, _GraphViews):
    """Coarse-grained molecular structure backed by a molrs ``CoarseGrain``.

    Base order: the pyo3 native ``molrs.CoarseGrain`` must be first (see the note
    on :class:`molpy.core.atomistic.Atomistic`).
    """

    _entity_cls = Bead
    _link_classes = {"bonds": CGBond}

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
        return self._atom_views()  # type: ignore[return-value]

    @property
    def cgbonds(self) -> Entities[CGBond]:
        return self._link_views("bonds")  # type: ignore[return-value]

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
        bead = self._spawn_entity(Bead(mapping, **attrs))
        if atoms is not None:
            self._set_bead_atoms(bead, tuple(atoms))
        return bead  # type: ignore[return-value]

    # ---------- bead → atom membership (molrs-owned handle store) ----------
    def _set_bead_atoms(self, bead: Bead, atoms: tuple["Atom", ...]) -> None:
        """Record a bead's atom membership in the molrs world (by handle)."""
        handles: list[int] = []
        for a in atoms:
            world = getattr(a, "_world", None)
            if world is None:
                raise ValueError(
                    "bead membership requires bound atoms (each must belong to a "
                    "source Atomistic world); got an unbound atom"
                )
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
        return tuple(self._member_world._intern_atom(h) for h in handles)

    def def_cgbond(self, a: Bead, b: Bead, /, **attrs: Any) -> CGBond:
        bond = CGBond(a, b, **attrs)
        return self._spawn_link("bonds", bond)  # type: ignore[return-value]

    def add_bead(self, bead: Bead, /) -> Bead:
        return self._spawn_entity(bead)  # type: ignore[return-value]

    def add_cgbond(self, bond: CGBond, /) -> CGBond:
        return self._spawn_link("bonds", bond)  # type: ignore[return-value]

    def add_entity(self, *beads: Bead) -> None:
        for bead in beads:
            self._spawn_entity(bead)

    def del_bead(self, *beads: Bead) -> None:
        for bead in beads:
            self._remove_atom(bead)

    def remove_entity(self, *beads: Bead, drop_incident_links: bool = True) -> None:
        for bead in beads:
            self._remove_atom(bead)

    def del_cgbond(self, *bonds: CGBond) -> None:
        for bond in bonds:
            self._remove_link(bond)

    def remove_link(self, *links: Link) -> None:
        for link in links:
            self._remove_link(link)

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

    def add_beads(self, beads: list[Bead], /) -> list[Bead]:
        for bead in beads:
            self._spawn_entity(bead)
        return beads

    def add_cgbonds(self, bonds: list[CGBond], /) -> list[CGBond]:
        for bond in bonds:
            self._spawn_link("bonds", bond)
        return bonds

    # ---------- reverse lookup ----------
    def beads_of(self, atom: "Atom") -> tuple[Bead, ...]:
        """Beads whose membership includes ``atom`` (molrs reverse lookup)."""
        handles = molrs.CoarseGrain.beads_of_atom(self, atom.handle)
        return tuple(self._intern_atom(h) for h in handles)  # type: ignore[misc]

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
            return self._link_views("bonds")
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
        new = type(self)()
        new._props = dict(self._props)
        bead_map: dict[Bead, Bead] = {}
        for bead in self.beads:
            bead_map[bead] = new.def_bead(dict(bead.data))
        for bond in self.cgbonds:
            mapped = [bead_map[ep] for ep in bond.endpoints]
            new.def_cgbond(*mapped, **dict(bond.data))
        return new

    def merge(self, other: "CoarseGrain") -> Self:
        """Transfer ``other``'s beads and bonds into ``self`` in place.

        Identity-preserving move (see :meth:`molpy.core.atomistic.Atomistic.merge`):
        the same view objects are reused; ``other`` is emptied.
        """
        beads = list(other.beads)
        bonds = list(other.cgbonds)
        for bead in beads:
            bead._detach()
            self._spawn_entity(bead)
        for bond in bonds:
            if all(ep._world is self for ep in bond.endpoints):
                bond._detach()
                self._spawn_link("bonds", bond)
        return self

    @staticmethod
    def adopt(graph: molrs.CoarseGrain) -> "CoarseGrain":
        """Zero-copy take ownership of a molrs-produced ``CoarseGrain`` graph."""
        struct = CoarseGrain()
        molrs.CoarseGrain.adopt(struct, graph)
        # Link views enumerate live relations from molrs via relation_ids().
        return struct

    # ---------- spatial ----------
    def move(
        self, delta: list[float], *, entity_type: type[Entity] = Bead
    ) -> "CoarseGrain":
        # Delegate to the molrs Rust kernel (vectorized over the dense
        # coordinate columns) instead of a per-bead Python loop.
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
        molrs.rotate(self, _unit(axis), float(angle), o)
        return self

    def scale(
        self,
        factor: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity] = Bead,
    ) -> "CoarseGrain":
        o = [0.0, 0.0, 0.0] if about is None else about
        for b in self.beads:
            xyz = _vec_add(o, _vec_scale(_vec_sub([b["x"], b["y"], b["z"]], o), factor))
            b["x"], b["y"], b["z"] = xyz
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
        if a_dir is not None and b_dir is not None:
            va = _unit(a_dir)
            vb = _unit(b_dir)
            if flip:
                vb = _vec_scale(vb, -1.0)
            axis = _cross(va, vb)
            na = _norm(axis)
            if na > 0:
                from math import atan2

                angle = atan2(na, _dot(va, vb))
                for e in self.beads:
                    xyz = _rodrigues_rotate(
                        [e["x"], e["y"], e["z"]], _vec_scale(axis, 1.0 / na), angle, pa
                    )
                    e["x"], e["y"], e["z"] = xyz
        self.move(_vec_sub(pb, pa))
        return self

    # ---------- composition ----------
    def __iadd__(self, other: "CoarseGrain") -> "CoarseGrain":
        self.merge(other)
        return self

    def __add__(self, other: "CoarseGrain") -> "CoarseGrain":
        result = self.copy()
        result.merge(other)
        return result

    def replicate(
        self, n: int, transform: Callable[["CoarseGrain", int], None] | None = None
    ) -> "CoarseGrain":
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
        from .frame import Frame

        # Upgrade the bare pyo3 frame to the rich ``Frame`` callers expect.
        frame = Frame.from_dict(molrs.CoarseGrain.to_frame(self))
        if bead_fields is not None and "beads" in frame:
            keep = set(bead_fields)
            beads = frame["beads"]
            for col in [k for k in beads.keys() if k not in keep]:
                del beads[col]
        return frame
