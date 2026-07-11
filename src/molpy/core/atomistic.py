"""All-atom molecular structure as a handle-view over a molrs ``Atomistic``.

``Atomistic(_GraphViews, molrs.Atomistic)`` IS a molrs world — it is accepted
directly by every ``molrs.*`` system free function (no ``.to_molrs()`` bridge).
:class:`Atom` / :class:`Bond` / :class:`Angle` / :class:`Dihedral` /
:class:`Improper` are handle views interned per stable handle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Self

import numpy as np

import molrs

from molpy.core.ops.geometry import (
    _cross,
    _dot,
    _norm,
    _unit,
    _vec_scale,
    _vec_sub,
)

from .atomistic_types import (
    Angle,
    Atom,
    Bond,
    Dihedral,
    DrudeParticle,
    Improper,
    MasslessSite,
    VirtualSite,
)
from .entity import Entities, Entity, Link, _GraphViews

if TYPE_CHECKING:
    from .frame import Frame

__all__ = [
    "Angle",
    "Atom",
    "Atomistic",
    "Bond",
    "Dihedral",
    "DrudeParticle",
    "Improper",
    "MasslessSite",
    "VirtualSite",
]


class Atomistic(molrs.Atomistic, _GraphViews):
    """All-atom molecular structure backed by a molrs ``Atomistic`` world.

    Note on base order: the pyo3 native ``molrs.Atomistic`` must be the first
    base so the extension instance layout is initialised correctly (a pyo3
    ``extends`` class cannot sit behind a plain-Python base). ``_GraphViews``
    contributes only non-conflicting helpers, and the leaf's own methods win in
    the MRO regardless of order, so this preserves the spec's intent.
    """

    _entity_cls = Atom
    _link_classes = {
        "bonds": Bond,
        "angles": Angle,
        "dihedrals": Dihedral,
        "impropers": Improper,
    }

    def __init__(self, **props: Any) -> None:
        _GraphViews.__init__(self, **props)
        if hasattr(self, "__post_init__"):
            for klass in type(self).__mro__:
                if klass is Atomistic:
                    break
                if "__post_init__" in klass.__dict__:
                    klass.__dict__["__post_init__"](self, **props)
                    break

    # ---------- collection views ----------
    @property
    def atoms(self) -> Entities[Atom]:
        return self._atom_views()  # type: ignore[return-value]

    @property
    def bonds(self) -> Entities[Bond]:
        return self._link_views("bonds")  # type: ignore[return-value]

    @property
    def angles(self) -> Entities[Angle]:
        return self._link_views("angles")  # type: ignore[return-value]

    @property
    def dihedrals(self) -> Entities[Dihedral]:
        return self._link_views("dihedrals")  # type: ignore[return-value]

    @property
    def impropers(self) -> Entities[Improper]:
        return self._link_views("impropers")  # type: ignore[return-value]

    @property
    def symbols(self) -> list[str]:
        """Element symbols for every atom (canonical :data:`~molpy.core.fields.ELEMENT`)."""
        from molpy.core import fields

        return [str(a.get(fields.ELEMENT) or "") for a in self.atoms]

    @property
    def xyz(self) -> np.ndarray:
        # Read the dense molrs columns directly rather than a per-atom Python
        # comprehension (~100x faster; column order matches self.atoms).
        x = np.asarray(self.column("x"))
        if x.size == 0:
            return np.zeros((0, 3), dtype=float)
        return np.stack(
            [x, np.asarray(self.column("y")), np.asarray(self.column("z"))], axis=1
        )

    @property
    def positions(self) -> np.ndarray:
        return self.xyz

    def __repr__(self) -> str:
        from collections import Counter

        atoms = list(self.atoms)
        from molpy.core import fields

        comp = Counter(a.get(fields.ELEMENT) or "?" for a in atoms)
        if len(comp) <= 5:
            composition = " ".join(f"{s}:{n}" for s, n in sorted(comp.items()))
        else:
            composition = f"{len(comp)} types"
        return (
            f"<Atomistic, {len(atoms)} atoms ({composition}), {len(self.bonds)} bonds>"
        )

    def __len__(self) -> int:
        return len(self.atoms)

    # ---------- factory methods (def_*: create + register) ----------
    def def_atom(self, mapping: Any = None, /, **attrs: Any) -> Atom:
        atom = Atom(mapping, **attrs)
        return self._spawn_entity(atom)  # type: ignore[return-value]

    def def_bond(self, a: Atom, b: Atom, /, **attrs: Any) -> Bond:
        bond = Bond(a, b, **attrs)
        return self._spawn_link("bonds", bond)  # type: ignore[return-value]

    def def_angle(self, a: Atom, b: Atom, c: Atom, /, **attrs: Any) -> Angle:
        angle = Angle(a, b, c, **attrs)
        return self._spawn_link("angles", angle)  # type: ignore[return-value]

    def def_dihedral(
        self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any
    ) -> Dihedral:
        dihedral = Dihedral(a, b, c, d, **attrs)
        return self._spawn_link("dihedrals", dihedral)  # type: ignore[return-value]

    def def_improper(
        self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any
    ) -> Improper:
        improper = Improper(a, b, c, d, **attrs)
        return self._spawn_link("impropers", improper)  # type: ignore[return-value]

    # ---------- add methods (add_*: register existing views) ----------
    def add_atom(self, atom: Atom, /) -> Atom:
        return self._spawn_entity(atom)  # type: ignore[return-value]

    def add_bond(self, bond: Bond, /) -> Bond:
        return self._spawn_link("bonds", bond)  # type: ignore[return-value]

    def add_angle(self, angle: Angle, /) -> Angle:
        return self._spawn_link("angles", angle)  # type: ignore[return-value]

    def add_dihedral(self, dihedral: Dihedral, /) -> Dihedral:
        return self._spawn_link("dihedrals", dihedral)  # type: ignore[return-value]

    def add_improper(self, improper: Improper, /) -> Improper:
        return self._spawn_link("impropers", improper)  # type: ignore[return-value]

    def add_entity(self, *atoms: Atom) -> None:
        for atom in atoms:
            self._spawn_entity(atom)

    def add_link(self, *links: Link, include_endpoints: bool = True) -> None:
        for link in links:
            if include_endpoints:
                for ep in link.endpoints:
                    if not ep.is_bound:
                        self._spawn_entity(ep)
            self._spawn_link(link._kind, link)

    # ---------- batch factory / add ----------
    def def_atoms(self, atoms_data: list[dict[str, Any]], /) -> list[Atom]:
        return [self.def_atom(**a) for a in atoms_data]

    def def_bonds(self, bonds_data: list[Any], /) -> list[Bond]:
        out: list[Bond] = []
        for spec in bonds_data:
            if len(spec) == 2:
                a, b = spec
                attrs: dict[str, Any] = {}
            else:
                a, b, attrs = spec
            out.append(self.def_bond(a, b, **attrs))
        return out

    def def_angles(self, angles_data: list[Any], /) -> list[Angle]:
        out: list[Angle] = []
        for spec in angles_data:
            if len(spec) == 3:
                a, b, c = spec
                attrs: dict[str, Any] = {}
            else:
                a, b, c, attrs = spec
            out.append(self.def_angle(a, b, c, **attrs))
        return out

    def def_dihedrals(self, dihedrals_data: list[Any], /) -> list[Dihedral]:
        out: list[Dihedral] = []
        for spec in dihedrals_data:
            if len(spec) == 4:
                a, b, c, d = spec
                attrs: dict[str, Any] = {}
            else:
                a, b, c, d, attrs = spec
            out.append(self.def_dihedral(a, b, c, d, **attrs))
        return out

    def add_atoms(self, atoms: list[Atom], /) -> list[Atom]:
        for atom in atoms:
            self._spawn_entity(atom)
        return atoms

    def add_bonds(self, bonds: list[Bond], /) -> list[Bond]:
        for bond in bonds:
            self._spawn_link("bonds", bond)
        return bonds

    def add_angles(self, angles: list[Angle], /) -> list[Angle]:
        for angle in angles:
            self._spawn_link("angles", angle)
        return angles

    def add_dihedrals(self, dihedrals: list[Dihedral], /) -> list[Dihedral]:
        for dihedral in dihedrals:
            self._spawn_link("dihedrals", dihedral)
        return dihedrals

    # ---------- delete ----------
    def del_atom(self, *atoms: Atom) -> None:
        for atom in atoms:
            self._remove_atom(atom)

    def remove_entity(self, *atoms: Atom, drop_incident_links: bool = True) -> None:
        for atom in atoms:
            self._remove_atom(atom)

    def del_bond(self, *bonds: Bond) -> None:
        for bond in bonds:
            self._remove_link(bond)

    def del_angle(self, *angles: Angle) -> None:
        for angle in angles:
            self._remove_link(angle)

    def del_dihedral(self, *dihedrals: Dihedral) -> None:
        for dihedral in dihedrals:
            self._remove_link(dihedral)

    def del_improper(self, *impropers: Improper) -> None:
        for improper in impropers:
            self._remove_link(improper)

    def remove_link(self, *links: Link) -> None:
        for link in links:
            self._remove_link(link)

    # ---------- property / type / selection editing ----------
    def rename_type(self, old: str, new: str, *, kind: type = Atom) -> int:
        items = self._items_of_kind(kind)
        count = 0
        for item in items:
            if item.get("type") == old:
                item["type"] = new
                count += 1
        return count

    def set_property(
        self, selector: Any, key: str, value: Any, *, kind: type = Atom
    ) -> int:
        if not callable(selector):
            raise TypeError(
                "selector must be a callable (a, ...) -> bool; "
                "SMARTS-string selectors are not yet supported"
            )
        count = 0
        for item in self._items_of_kind(kind):
            if selector(item):
                item[key] = value
                count += 1
        return count

    def select(self, predicate: Any) -> "Atomistic":
        if not callable(predicate):
            raise TypeError(
                "predicate must be a callable (atom) -> bool; "
                "SMARTS-string predicates are not yet supported"
            )
        selected = [a for a in self.atoms if predicate(a)]
        sub, _ = self.extract_subgraph(selected, radius=0)
        return sub

    def _items_of_kind(self, kind: type) -> Entities[Any]:
        if isinstance(kind, type) and issubclass(kind, Link):
            return self._link_views(kind._kind)
        return self.atoms  # type: ignore[return-value]

    # ---------- connectivity / topology ----------
    def get_neighbors(self, atom: Atom, link_type: type[Link] = Bond) -> list[Atom]:
        # Bonds (arity-2) live in the molrs adjacency index, so resolve
        # neighbours in O(degree) instead of scanning every link. Self-loops are
        # excluded to match the scan's `ep is not atom` semantics.
        if link_type is Bond:
            h = atom.handle
            return [
                self._intern_atom(other)
                for _, other in self.incident_relations(h, link_type._kind)
                if other != h
            ]
        # Higher-arity link types are not in the adjacency index — fall back.
        out: list[Atom] = []
        for link in self._link_views(link_type._kind):
            if any(ep is atom for ep in link.endpoints):
                out.extend(ep for ep in link.endpoints if ep is not atom)
        return out

    def get_topo(
        self,
        entity_type: type[Atom] = Atom,
        link_type: type[Link] = Bond,
        gen_angle: bool = False,
        gen_dihe: bool = False,
        clear_existing: bool = False,
    ) -> "Atomistic":
        """Return a copy with angle/dihedral relations perceived from the bonds.

        Angle/dihedral perception (2-edge / 3-edge paths over the bond graph) is
        a molrs-native graph operation; this delegates to that Rust kernel on a
        copy. With no ``gen_*`` flags it is a plain copy. Always returns an
        :class:`Atomistic` (never a bare topology graph).
        """
        new_struct = self.copy()
        new_struct.generate_topology(
            gen_angle=gen_angle,
            gen_dihedral=gen_dihe,
            clear_existing=clear_existing,
        )
        return new_struct

    def complete_valence(self) -> "Atomistic":
        """Return a copy with every dangling valence filled by hydrogen.

        Under-coordinated atoms (bond degree below their element valence) are
        capped with hydrogens, turning a sliced subgraph into a chemically valid
        molecule an external tool can parameterise. Original atoms keep their
        order (caps appended); ``self`` is untouched. See
        :func:`molpy.core.capping.complete_valence`.
        """
        from molpy.core.capping import complete_valence

        return complete_valence(self)

    def get_topo_neighbors(
        self,
        entity: Atom,
        radius: int = 1,
        entity_type: type[Atom] = Atom,
        link_type: type[Link] = Bond,
    ) -> list[Atom]:
        # BFS over the bond graph via the molrs Rust kernel (single source);
        # unreachable atoms are already excluded. Matches the prior semantics
        # (the source itself, at distance 0, is within any radius >= 0).
        return [
            self._intern_atom(h)
            for h, _ in self.topo_distances(entity.handle, max_hops=radius)
        ]

    def get_topo_distances(
        self,
        source: Atom,
        entity_type: type[Atom] = Atom,
        link_type: type[Link] = Bond,
    ) -> dict[Atom, int]:
        return {
            self._intern_atom(h): int(d) for h, d in self.topo_distances(source.handle)
        }

    def extract_subgraph(
        self,
        center_entities: Iterable[Atom],
        radius: int,
        entity_type: type[Atom] = Atom,
        link_type: type[Link] = Bond,
    ) -> tuple["Atomistic", list[Atom]]:
        sub, boundary, _, _ = self._extract_mapped(
            list(center_entities), radius, type(self)
        )
        return sub, boundary

    def _extract_mapped[G: "Atomistic"](
        self,
        centers: list[Atom],
        radius: int,
        out_cls: type[G],
        *,
        regenerate_topology: bool = False,
    ) -> tuple[G, list[Atom], dict[Atom, Atom], dict[int, int]]:
        """Induced radius-``radius`` ball plus a region-atom → parent-atom map.

        Delegates BFS + materialisation to
        :meth:`molrs.Atomistic.extract_subgraph`. Returns
        ``(subgraph, boundary_atoms, {region_atom: parent_atom},
        {region_atom_handle: hops_from_nearest_center})``.
        """
        new = out_cls()
        new._props = dict(self._props)
        if not centers:
            return new, [], {}, {}

        res = molrs.Atomistic.extract_subgraph(
            self,
            [c.handle for c in centers],
            int(radius),
            regenerate_topology=regenerate_topology,
        )
        molrs.Atomistic.adopt(new, res.graph)

        parent_by_old: dict[int, Atom] = {
            old: self._intern_atom(old) for old in res.node_map
        }
        region_by_old: dict[int, Atom] = {
            old: new._intern_atom(new_h)  # type: ignore[attr-defined]
            for old, new_h in res.node_map.items()
        }
        boundary = [region_by_old[h] for h in res.boundary if h in region_by_old]
        region_to_parent = {
            region_by_old[old]: parent_by_old[old] for old in res.node_map
        }
        hops = {
            res.node_map[old]: int(d)
            for old, d in res.hops.items()
            if old in res.node_map
        }
        return new, boundary, region_to_parent, hops

    # ---------- copy / merge / adopt ----------
    def copy(self) -> Self:
        """Independent deep copy. **Handles are preserved** (molrs clone)."""
        bare = molrs.Atomistic.copy(self)
        new = type(self)()
        molrs.Atomistic.adopt(new, bare)
        new._props = dict(self._props)
        return new

    def merge(self, other: "Atomistic") -> Self:
        """Structural merge of ``other`` into ``self`` (molrs).

        Every node of ``other`` is remapped to a fresh handle in ``self``.
        ``other`` is emptied and must not be used afterwards. Cross-graph
        identity is handle-based — Python view objects are **not** rebound.
        """
        molrs.Atomistic.merge(self, other)
        other._atom_intern.clear()
        other._link_intern.clear()
        other._props.clear()
        return self

    @staticmethod
    def adopt(graph: molrs.Atomistic) -> "Atomistic":
        """Zero-copy take ownership of a molrs-produced ``Atomistic`` graph.

        Uses the molrs zero-copy ``adopt`` to move ``graph``'s storage into a
        fresh molpy ``Atomistic`` (``graph`` is left empty). Views over the
        adopted nodes/relations are interned lazily on access.
        """
        struct = Atomistic()
        molrs.Atomistic.adopt(struct, graph)
        return struct

    @classmethod
    def from_frame(cls, frame: "Frame") -> "Atomistic":
        """Build a molpy ``Atomistic`` from a :class:`~molpy.core.frame.Frame`.

        The inverse of :meth:`to_frame`. molrs' inherited ``from_frame`` returns a
        bare molrs graph; this override adopts it so the result is a molpy
        ``Atomistic`` — the call site never needs a second ``adopt``.
        """
        return cls.adopt(molrs.Atomistic.from_frame(frame))

    # ---------- spatial operations ----------
    def move(
        self, delta: list[float], *, entity_type: type[Entity] = Atom
    ) -> "Atomistic":
        # Delegate to the molrs Rust kernel (vectorized over the dense
        # coordinate columns) instead of a per-atom Python loop.
        molrs.translate(self, [float(d) for d in delta])
        return self

    def rotate(
        self,
        axis: list[float],
        angle: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity] = Atom,
    ) -> "Atomistic":
        o = [0.0, 0.0, 0.0] if about is None else list(about)
        molrs.rotate(self, _unit(axis), float(angle), o)
        return self

    def scale(
        self,
        factor: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity] = Atom,
    ) -> "Atomistic":
        o = [0.0, 0.0, 0.0] if about is None else list(about)
        molrs.scale(self, [factor, factor, factor], o)
        return self

    def align(
        self,
        a: Atom,
        b: Atom,
        *,
        a_dir: list[float] | None = None,
        b_dir: list[float] | None = None,
        flip: bool = False,
        entity_type: type[Entity] = Atom,
    ) -> "Atomistic":
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
                # Rotate about pa via the molrs Rust kernel (was a per-atom
                # Rodrigues loop reimplementing molrs.rotate).
                molrs.rotate(self, _vec_scale(axis, 1.0 / na), angle, pa)
        self.move(_vec_sub(pb, pa))
        return self

    # ---------- composition ----------
    def __iadd__(self, other: "Atomistic") -> "Atomistic":
        self.merge(other)
        return self

    def __add__(self, other: "Atomistic") -> "Atomistic":
        result = self.copy()
        result.merge(other.copy())  # merge empties its argument
        return result

    def replicate(self, n: int, transform: Any = None) -> "Atomistic":
        result = type(self)()
        for i in range(n):
            replica = self.copy()
            if transform is not None:
                transform(replica, i)
            result.merge(replica)
        return result

    # ---------- tabular conversion ----------
    def to_frame(self, atom_fields: list[str] | None = None) -> "Frame":
        """Export to a tabular :class:`Frame` (atoms + bonds/angles/dihedrals/
        impropers blocks).

        Delegates straight to the molrs world's native ``to_frame``: the Rust
        column store already holds every component as a dense, row-aligned
        column, so each block is materialized as numpy with zero Python-side
        conversion. ``atom_fields`` optionally restricts the atoms block columns.
        """
        from .frame import Frame

        # ``molrs.Atomistic.to_frame`` yields the bare pyo3 frame; upgrade it to
        # the rich ``Frame`` (metadata, box, rich Blocks) callers expect.
        frame = Frame.from_dict(molrs.Atomistic.to_frame(self))
        if atom_fields is not None and "atoms" in frame:
            keep = set(atom_fields)
            atoms = frame["atoms"]
            for col in [k for k in atoms.keys() if k not in keep]:
                del atoms[col]
        return frame
