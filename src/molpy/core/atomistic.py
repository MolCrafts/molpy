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

from .entity import Entities, Entity, Link, _GraphViews

if TYPE_CHECKING:
    from .frame import Frame


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
            for h, d in self.topo_distances(entity.handle)
            if d <= radius
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
        sub, boundary, _ = self._extract_mapped(
            list(center_entities), radius, type(self)
        )
        return sub, boundary

    def _extract_mapped[G: "Atomistic"](
        self,
        centers: list[Atom],
        radius: int,
        out_cls: type[G],
    ) -> tuple[G, list[Atom], dict[Atom, Atom]]:
        """Induced radius-``radius`` ball plus a region-atom → parent-atom map.

        Backs :meth:`extract_subgraph` (which drops the map) and
        :class:`~molpy.core.affected_region.AffectedRegion` (which keeps it).
        ``out_cls`` selects the produced graph type — a plain :class:`Atomistic`
        or an :class:`AffectedRegion` — so the ball is materialised straight into
        a region subclass with no second copy. Returns
        ``(subgraph, boundary_atoms, {region_atom: parent_atom})``.
        """
        new = out_cls()
        new._props = dict(self._props)
        if not centers:
            return new, [], {}

        # BFS radius ball around every center over the bond graph (molrs kernel).
        # A center not in this structure contributes nothing (empty distances).
        selected_handles: set[int] = set()
        for c in centers:
            selected_handles.update(
                h for h, d in self.topo_distances(c.handle) if d <= radius
            )
        if not selected_handles:
            return new, [], {}

        # Deterministic (row) order over the selection.
        selected_entities = [a for a in self.atoms if a.handle in selected_handles]
        selected_set = set(selected_entities)

        # Boundary atoms: a selected atom with a bond-neighbor outside the ball.
        edge_entities = [
            a
            for a in selected_entities
            if any(nb not in selected_set for nb in self.get_neighbors(a))
        ]

        # clone selected atoms + induced topology into the new struct
        parent_to_clone: dict[Atom, Atom] = {}
        for atom in selected_entities:
            parent_to_clone[atom] = new.def_atom(dict(atom.data))

        def _clone_links(views: Entities[Any], adder: Any) -> None:
            for link in views:
                eps = link.endpoints
                if all(ep in selected_set for ep in eps):
                    mapped = [parent_to_clone[ep] for ep in eps]
                    adder(*mapped, **dict(link.data))

        _clone_links(self.bonds, new.def_bond)
        _clone_links(self.angles, new.def_angle)
        _clone_links(self.dihedrals, new.def_dihedral)
        _clone_links(self.impropers, new.def_improper)

        boundary = [parent_to_clone[e] for e in edge_entities if e in parent_to_clone]
        region_to_parent = {clone: parent for parent, clone in parent_to_clone.items()}
        return new, boundary, region_to_parent

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
        # Link views enumerate live relations straight from molrs via
        # relation_ids(); no Python-side handle shadow to populate.
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
