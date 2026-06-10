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
    _rodrigues_rotate,
    _unit,
    _vec_add,
    _vec_scale,
    _vec_sub,
)

from .entity import Entities, Entity, Link, _GraphViews

if TYPE_CHECKING:
    from .frame import Frame
    from .topology import Topology


# ===================================================================
#                       Handle view types
# ===================================================================


class Atom(Entity):
    """Atom view: one node in a molrs ``Atomistic`` world (or pending)."""

    __slots__ = ()

    def __repr__(self) -> str:
        ident = self.get("element") or self.get("symbol") or self.get("type")
        return f"<Atom: {ident if ident is not None else id(self)}>"

    @property
    def is_virtual(self) -> bool:
        """True if this node carries a virtual-site marker (``vsite`` field).

        The marker is a stored data field, not the Python class: molrs
        re-materialises nodes as plain :class:`Atom`, so identity must be read
        from the persisted ``vsite`` field rather than ``isinstance``.
        """
        return self.get("vsite") is not None


class VirtualSite(Atom):
    """A massless / rule-placed auxiliary particle. Data only — no energy.

    Carries a persistent ``vsite`` marker field (set on construction) plus the
    usual atom data. Subclasses set the marker value via ``_vsite_kind``.
    Identity after a molrs round-trip is read from the ``vsite`` field
    (:attr:`Atom.is_virtual`), since the Python subclass is not preserved.
    """

    __slots__ = ()
    _vsite_kind = "virtual"

    def __init__(self, mapping: Any = None, /, **attrs: Any) -> None:
        attrs.setdefault("vsite", self._vsite_kind)
        super().__init__(mapping, **attrs)


class DrudeParticle(VirtualSite):
    """Polarizable Drude shell: co-located with its core, spring-bound."""

    __slots__ = ()
    _vsite_kind = "drude"


class MasslessSite(VirtualSite):
    """Rigid geometric site (e.g. TIP4P M-site, lone pair); no spring."""

    __slots__ = ()
    _vsite_kind = "massless"


class Bond(Link[Atom]):
    """Covalent bond between two atoms (molrs relation kind ``bonds``)."""

    __slots__ = ()
    _kind = "bonds"

    def __init__(self, a: Atom, b: Atom, /, **attrs: Any) -> None:
        assert isinstance(a, Atom), f"atom a must be an Atom instance, got {type(a)}"
        assert isinstance(b, Atom), f"atom b must be an Atom instance, got {type(b)}"
        super().__init__([a, b], **attrs)

    def __repr__(self) -> str:
        return f"<Bond: {self.itom} - {self.jtom}>"

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]


class Angle(Link[Atom]):
    """Valence angle over three atoms i--j--k (kind ``angles``)."""

    __slots__ = ()
    _kind = "angles"

    def __init__(self, a: Atom, b: Atom, c: Atom, /, **attrs: Any) -> None:
        for x in (a, b, c):
            assert isinstance(x, Atom), f"endpoint must be an Atom, got {type(x)}"
        super().__init__([a, b, c], **attrs)

    def __repr__(self) -> str:
        return f"<Angle: {self.itom} - {self.jtom} - {self.ktom}>"

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]

    @property
    def ktom(self) -> Atom:
        return self.endpoints[2]


class Dihedral(Link[Atom]):
    """Proper dihedral (torsion) over four atoms (kind ``dihedrals``)."""

    __slots__ = ()
    _kind = "dihedrals"

    def __init__(self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any) -> None:
        for x in (a, b, c, d):
            assert isinstance(x, Atom), f"endpoint must be an Atom, got {type(x)}"
        super().__init__([a, b, c, d], **attrs)

    def __repr__(self) -> str:
        return f"<Dihedral: {self.itom} - {self.jtom} - {self.ktom} - {self.ltom}>"

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]

    @property
    def ktom(self) -> Atom:
        return self.endpoints[2]

    @property
    def ltom(self) -> Atom:
        return self.endpoints[3]


class Improper(Link[Atom]):
    """Improper torsion over four atoms, ``i`` central (kind ``impropers``)."""

    __slots__ = ()
    _kind = "impropers"

    def __init__(self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any) -> None:
        for x in (a, b, c, d):
            assert isinstance(x, Atom), f"endpoint must be an Atom, got {type(x)}"
        super().__init__([a, b, c, d], **attrs)

    def __repr__(self) -> str:
        return f"<Improper: {self.itom} - {self.jtom} - {self.ktom} - {self.ltom}>"

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]

    @property
    def ktom(self) -> Atom:
        return self.endpoints[2]

    @property
    def ltom(self) -> Atom:
        return self.endpoints[3]


# ===================================================================
#                          Atomistic leaf
# ===================================================================


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
        return [str(a.get("element") or a.get("symbol") or "") for a in self.atoms]

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
        comp = Counter(a.get("element") or a.get("symbol") or "?" for a in atoms)
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
    ) -> "Atomistic | Topology":
        if not gen_angle and not gen_dihe:
            return self._build_topology(link_type)

        new_struct = self.copy()
        topo = new_struct._build_topology(link_type)
        atoms = topo.idx_to_entity

        if gen_angle:
            if clear_existing:
                new_struct.del_angle(*list(new_struct.angles))
            existing = {(ang.itom, ang.jtom, ang.ktom) for ang in new_struct.angles}
            for angle in topo.angles:
                i, j, k = angle.tolist()
                triple = (atoms[i], atoms[j], atoms[k])
                if triple not in existing:
                    new_struct.def_angle(*triple)
                    existing.add(triple)

        if gen_dihe:
            if clear_existing:
                new_struct.del_dihedral(*list(new_struct.dihedrals))
            existing_d = {
                (d.itom, d.jtom, d.ktom, d.ltom) for d in new_struct.dihedrals
            }
            for dihe in topo.dihedrals:
                i, j, k, l = dihe.tolist()
                quad = (atoms[i], atoms[j], atoms[k], atoms[l])
                if quad not in existing_d:
                    new_struct.def_dihedral(*quad)
                    existing_d.add(quad)

        return new_struct

    def _build_topology(self, link_type: type[Link] = Bond) -> "Topology":
        from molpy.core.topology import Topology

        atoms = list(self.atoms)
        entity_to_idx = {a: i for i, a in enumerate(atoms)}
        entity_set = set(atoms)
        edges: list[tuple[int, int]] = []
        for link in self._link_views(link_type._kind):
            eps = link.endpoints
            if len(eps) >= 2 and eps[0] in entity_set and eps[1] in entity_set:
                i, j = entity_to_idx[eps[0]], entity_to_idx[eps[1]]
                if i != j:
                    edges.append((i, j))
        return Topology(
            n=len(atoms),
            edges=edges,
            directed=False,
            entity_to_idx=entity_to_idx,
            idx_to_entity=atoms,
        )

    def get_topo_neighbors(
        self,
        entity: Atom,
        radius: int = 1,
        entity_type: type[Atom] = Atom,
        link_type: type[Link] = Bond,
    ) -> list[Atom]:
        topo = self._build_topology(link_type)
        if entity not in topo.entity_to_idx:
            return []
        center = topo.entity_to_idx[entity]
        distances = topo.distances(source=[center])[0]
        idx_to_entity = topo.idx_to_entity
        return [
            idx_to_entity[i]
            for i, d in enumerate(distances)
            if d <= radius and d < float("inf")
        ]

    def get_topo_distances(
        self,
        source: Atom,
        entity_type: type[Atom] = Atom,
        link_type: type[Link] = Bond,
    ) -> dict[Atom, int]:
        topo = self._build_topology(link_type)
        if source not in topo.entity_to_idx:
            return {}
        s = topo.entity_to_idx[source]
        distances = topo.distances(source=[s])[0]
        idx_to_entity = topo.idx_to_entity
        return {
            idx_to_entity[i]: int(d)
            for i, d in enumerate(distances)
            if d < float("inf")
        }

    def extract_subgraph(
        self,
        center_entities: Iterable[Atom],
        radius: int,
        entity_type: type[Atom] = Atom,
        link_type: type[Link] = Bond,
    ) -> tuple["Atomistic", list[Atom]]:
        centers = list(center_entities)
        topo = self._build_topology(link_type)
        entity_to_idx = topo.entity_to_idx
        idx_to_entity = topo.idx_to_entity

        center_idx = [entity_to_idx[c] for c in centers if c in entity_to_idx]
        new_struct = type(self)()
        new_struct._props = dict(self._props)
        if not center_idx:
            return new_struct, []

        selected: set[int] = set()
        for c in center_idx:
            distances = topo.distances(source=[c])[0]
            for i, d in enumerate(distances):
                if d <= radius and d < float("inf"):
                    selected.add(i)

        selected_entities = [idx_to_entity[i] for i in sorted(selected)]
        selected_set = set(selected_entities)

        edge_idx: set[int] = set()
        for i in selected:
            for j in topo.neighbors(i):
                if j not in selected:
                    edge_idx.add(i)
                    break
        edge_entities = [idx_to_entity[i] for i in sorted(edge_idx)]

        # clone selected atoms + induced topology into the new struct
        entity_map: dict[Atom, Atom] = {}
        for atom in selected_entities:
            clone = new_struct.def_atom(dict(atom.data))
            entity_map[atom] = clone

        def _clone_links(views: Entities[Any], adder: Any) -> None:
            for link in views:
                eps = link.endpoints
                if all(ep in selected_set for ep in eps):
                    mapped = [entity_map[ep] for ep in eps]
                    adder(*mapped, **dict(link.data))

        _clone_links(self.bonds, new_struct.def_bond)
        _clone_links(self.angles, new_struct.def_angle)
        _clone_links(self.dihedrals, new_struct.def_dihedral)
        _clone_links(self.impropers, new_struct.def_improper)

        cloned_edges = [entity_map[e] for e in edge_entities if e in entity_map]
        return new_struct, cloned_edges

    # ---------- copy / merge / adopt ----------
    def copy(self) -> Self:
        new = type(self)()
        new._props = dict(self._props)
        emap: dict[Atom, Atom] = {}
        for atom in self.atoms:
            emap[atom] = new.def_atom(dict(atom.data))

        def _copy_links(views: Entities[Any], adder: Any) -> None:
            for link in views:
                mapped = [emap[ep] for ep in link.endpoints]
                adder(*mapped, **dict(link.data))

        _copy_links(self.bonds, new.def_bond)
        _copy_links(self.angles, new.def_angle)
        _copy_links(self.dihedrals, new.def_dihedral)
        _copy_links(self.impropers, new.def_improper)
        return new

    def merge(self, other: "Atomistic") -> Self:
        """Transfer ``other``'s atoms and topology into ``self`` in place.

        The **same** view objects are reused (identity preserved): each of
        ``other``'s atoms/links is detached and re-spawned onto ``self`` with a
        fresh handle, so external references (e.g. a reacter's leaving-group
        list) stay valid. ``other`` is emptied and must not be used afterwards.
        """
        atoms = list(other.atoms)
        bonds = list(other.bonds)
        angles = list(other.angles)
        dihedrals = list(other.dihedrals)
        impropers = list(other.impropers)

        for atom in atoms:
            atom._detach()
            self._spawn_entity(atom)

        def _transfer(links: list[Any], kind: str) -> None:
            for link in links:
                if all(ep._world is self for ep in link.endpoints):
                    link._detach()
                    self._spawn_link(kind, link)

        _transfer(bonds, "bonds")
        _transfer(angles, "angles")
        _transfer(dihedrals, "dihedrals")
        _transfer(impropers, "impropers")
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
        # surface every relation handle so the link views enumerate. molrs
        # relation handles are 1-based dense per kind after a fresh adopt.
        for kind in struct.kinds():
            n = struct.n_relations(kind)
            struct._rel_handles[kind] = _relation_handles(struct, kind, n)
        return struct

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
        o = [0.0, 0.0, 0.0] if about is None else about
        for a in self.atoms:
            xyz = _vec_add(o, _vec_scale(_vec_sub([a["x"], a["y"], a["z"]], o), factor))
            a["x"], a["y"], a["z"] = xyz
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
                for e in self.atoms:
                    xyz = _rodrigues_rotate(
                        [e["x"], e["y"], e["z"]], _vec_scale(axis, 1.0 / na), angle, pa
                    )
                    e["x"], e["y"], e["z"] = xyz
        self.move(_vec_sub(pb, pa))
        return self

    # ---------- composition ----------
    def __iadd__(self, other: "Atomistic") -> "Atomistic":
        self.merge(other)
        return self

    def __add__(self, other: "Atomistic") -> "Atomistic":
        result = self.copy()
        result.merge(other)
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
        from ._columns import to_numpy_column
        from .frame import Block, Frame

        frame = Frame()
        atoms_data = list(self.atoms)
        atom_index = {id(a): i for i, a in enumerate(atoms_data)}

        if atom_fields is None:
            keys: set[str] = set()
            for atom in atoms_data:
                keys.update(atom.keys())
        else:
            keys = set(atom_fields)
        # numpy-only Store: build a representable column per key, skipping any
        # that cannot be expressed without an object dtype (ragged / all-None).
        atom_cols: dict[str, np.ndarray] = {}
        for k in keys:
            col = to_numpy_column([atom.get(k, None) for atom in atoms_data])
            if col is not None:
                atom_cols[k] = col
        frame["atoms"] = Block.from_dict(atom_cols)

        self._links_to_block(frame, "bonds", self.bonds, ("atomi", "atomj"), atom_index)
        self._links_to_block(
            frame, "angles", self.angles, ("atomi", "atomj", "atomk"), atom_index
        )
        self._links_to_block(
            frame,
            "dihedrals",
            self.dihedrals,
            ("atomi", "atomj", "atomk", "atoml"),
            atom_index,
        )
        self._links_to_block(
            frame,
            "impropers",
            self.impropers,
            ("atomi", "atomj", "atomk", "atoml"),
            atom_index,
        )
        return frame

    @staticmethod
    def _links_to_block(
        frame: Any,
        name: str,
        links: Entities[Any],
        index_labels: tuple[str, ...],
        atom_index: dict[int, int],
    ) -> None:
        links = list(links)  # type: ignore[assignment]
        if not links:
            return
        from ._columns import to_numpy_column
        from .frame import Block

        block: dict[str, list[Any]] = {lbl: [] for lbl in index_labels}
        all_keys: set[str] = set()
        for link in links:
            all_keys.update(link.keys())
        attr_keys = [k for k in all_keys if k not in index_labels]
        for k in attr_keys:
            block[k] = []
        for n, link in enumerate(links):
            for lbl, ep in zip(index_labels, link.endpoints):
                if id(ep) not in atom_index:
                    raise ValueError(
                        f"{name} {n + 1}: {lbl} references an atom not in the "
                        f"atoms list (removed or invalid)."
                    )
                block[lbl].append(atom_index[id(ep)])
            for k in attr_keys:
                block[k].append(link.get(k, None))
        # numpy-only Store: index columns are dense ints; skip attr columns that
        # are not numpy-representable (ragged / all-None) rather than overflow.
        cols: dict[str, np.ndarray] = {}
        for k, v in block.items():
            col = to_numpy_column(v)
            if col is not None:
                cols[k] = col
        frame[name] = Block.from_dict(cols)


def _relation_handles(struct: molrs.Atomistic, kind: str, n: int) -> list[int]:
    """Discover the live relation handles of ``kind`` after a fresh adopt.

    molrs relation handles are opaque ``u64`` encoded as ``(generation<<32)|index``
    with no enumeration API. A graph produced by SMILES / Conformer has never had
    a relation removed, so its ``kind`` handles occupy the dense generation-1
    range ``base+1 .. base+n``; probe that range and keep the resolvable handles.
    """
    handles: list[int] = []
    base = 1 << 32
    idx = 1
    # probe a generous window past n to tolerate any sparse gaps
    while len(handles) < n and idx <= n + 1024:
        rh = base + idx
        try:
            struct.relation_nodes(kind, rh)
            handles.append(rh)
        except Exception:
            pass
        idx += 1
    return handles
