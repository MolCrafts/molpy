"""All-atom molecular structure as a handle-view over a native ``Atomistic``.

``Atomistic`` mixes ``_GraphViews`` into the native ``Atomistic``, so it IS a
native world — accepted directly by every native ``system`` free function (no
conversion bridge).
:class:`Atom` / :class:`Bond` / :class:`Angle` / :class:`Dihedral` /
:class:`Improper` are handle views interned per stable handle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Self

import numpy as np

import molrs

from molrs.views import (
    Angle,
    Atom,
    Bond,
    Dihedral,
    DrudeParticle,
    Improper,
    MasslessSite,
    VirtualSite,
    _GraphViews,
)

from molpy.core.entity import Entities, Link, NotPublic
from molpy.core import fields

if TYPE_CHECKING:
    from molrs import Frame

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
    """All-atom molecular structure backed by a native ``Atomistic`` world.

    Note on base order: the pyo3 native ``Atomistic`` must be the first
    base so the extension instance layout is initialised correctly (a pyo3
    ``extends`` class cannot sit behind a plain-Python base). ``_GraphViews``
    contributes only non-conflicting helpers, and the leaf's own methods win in
    the MRO regardless of order, so this preserves the spec's intent.
    """

    _node_cls = Atom
    _relation_classes = {
        "bonds": Bond,
        "angles": Angle,
        "dihedrals": Dihedral,
        "impropers": Improper,
    }
    add_atom = NotPublic("def_atom")
    add_bond = NotPublic("def_bond")
    add_angle = NotPublic("def_angle")
    add_dihedral = NotPublic("def_dihedral")
    add_improper = NotPublic("def_improper")

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
        return self._node_views()  # type: ignore[return-value]

    @property
    def bonds(self) -> Entities[Bond]:
        return self._relation_views("bonds")  # type: ignore[return-value]

    @property
    def angles(self) -> Entities[Angle]:
        return self._relation_views("angles")  # type: ignore[return-value]

    @property
    def dihedrals(self) -> Entities[Dihedral]:
        return self._relation_views("dihedrals")  # type: ignore[return-value]

    @property
    def impropers(self) -> Entities[Improper]:
        return self._relation_views("impropers")  # type: ignore[return-value]

    @property
    def symbols(self) -> list[str]:
        """Element symbols for every atom (canonical :data:`~molpy.core.fields.ELEMENT`).

        Reads the world component store by handle — does **not** intern ``Atom``
        views (see also :meth:`column` for dense numeric fields).
        """
        key = fields.ELEMENT
        if key not in self.columns():
            return [""] * len(self.entities())
        if self.validity(key).all():
            return [str(s) for s in self.column(key).tolist()]
        return [str(self.get(h, key) or "") for h in self.entities()]

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

        # View-free: symbols/len via handles + n_relations, not list(self.atoms).
        syms = self.symbols
        n_atoms = len(syms)
        comp = Counter(s or "?" for s in syms)
        if len(comp) <= 5:
            composition = " ".join(f"{s}:{n}" for s, n in sorted(comp.items()))
        else:
            composition = f"{len(comp)} types"
        n_bonds = self.n_relations("bonds") if "bonds" in self.kinds() else 0
        return f"<Atomistic, {n_atoms} atoms ({composition}), {n_bonds} bonds>"

    def __len__(self) -> int:
        return self.n_nodes

    # ---------- factory methods (def_*: create + register) ----------
    def def_atom(self, mapping: Any = None, /, **attrs: Any) -> Atom:
        return self._create_node(mapping, cls=Atom, **attrs)  # type: ignore[return-value]

    def def_virtual_site(
        self,
        mapping: Any = None,
        /,
        *,
        kind: type[VirtualSite] = VirtualSite,
        **attrs: Any,
    ) -> VirtualSite:
        attrs.setdefault("vsite", kind._vsite_kind)
        return self._create_node(mapping, cls=kind, **attrs)  # type: ignore[return-value]

    def def_bond(self, a: Atom, b: Atom, /, **attrs: Any) -> Bond:
        return self._create_relation("bonds", (a, b), cls=Bond, **attrs)  # type: ignore[return-value]

    def def_angle(self, a: Atom, b: Atom, c: Atom, /, **attrs: Any) -> Angle:
        return self._create_relation("angles", (a, b, c), cls=Angle, **attrs)  # type: ignore[return-value]

    def def_dihedral(
        self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any
    ) -> Dihedral:
        return self._create_relation("dihedrals", (a, b, c, d), cls=Dihedral, **attrs)  # type: ignore[return-value]

    def def_improper(
        self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any
    ) -> Improper:
        return self._create_relation("impropers", (a, b, c, d), cls=Improper, **attrs)  # type: ignore[return-value]

    # ---------- batch factories ----------
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

    # ---------- delete ----------
    def del_atom(self, *atoms: Atom) -> None:
        for atom in atoms:
            self._remove_node(atom)

    def remove_entity(self, *atoms: Atom, drop_incident_links: bool = True) -> None:
        for atom in atoms:
            self._remove_node(atom)

    def del_bond(self, *bonds: Bond) -> None:
        for bond in bonds:
            self._remove_relation(bond)

    def del_angle(self, *angles: Angle) -> None:
        for angle in angles:
            self._remove_relation(angle)

    def del_dihedral(self, *dihedrals: Dihedral) -> None:
        for dihedral in dihedrals:
            self._remove_relation(dihedral)

    def del_improper(self, *impropers: Improper) -> None:
        for improper in impropers:
            self._remove_relation(improper)

    def remove_link(self, *links: Link) -> None:
        for link in links:
            self._remove_relation(link)

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
            return self._relation_views(kind._kind)
        return self.atoms  # type: ignore[return-value]

    # ---------- connectivity / topology ----------
    def get_neighbors(self, atom: Atom, link_type: type[Link] = Bond) -> list[Atom]:
        # Bonds (arity-2) live in the molrs adjacency index, so resolve
        # neighbours in O(degree) instead of scanning every link. Self-loops are
        # excluded to match the scan's `ep is not atom` semantics.
        if link_type is Bond:
            h = atom.handle
            return [
                self._intern_node(other)
                for _, other in self.incident_relations(h, link_type._kind)
                if other != h
            ]
        # Higher-arity link types are not in the adjacency index — fall back.
        out: list[Atom] = []
        for link in self._relation_views(link_type._kind):
            if any(ep is atom for ep in link.endpoints):
                out.extend(ep for ep in link.endpoints if ep is not atom)
        return out

    def get_topo(
        self,
        *,
        gen_angle: bool = False,
        gen_dihe: bool = False,
        gen_improper: bool = False,
        clear_existing: bool = False,
    ) -> "Atomistic":
        """Perceive angles/dihedrals/impropers from the bond graph **in place**.

        All three run in the native Rust kernel via :meth:`generate_topology`;
        nothing is enumerated on the Python side. Mutates ``self`` and returns
        it for chaining — matching the core mutation contract (``.copy()`` is
        the explicit opt-in for an independent graph).

        Angles are 2-edge paths and proper dihedrals 3-edge paths over the bond
        graph. Impropers are the **molecular-mechanics** reading: one
        ``[centre, i, j, k]`` quartet per atom with exactly three neighbours,
        centre first and peripherals sorted — not every 3-combination at every
        centre of degree >= 3, which would hand an sp3 carbon four quartets
        where a force field wants none. Whether a trivalent centre is planar
        enough to deserve the term is force-field data (GAFF reads PARMCHK's
        ``improper_flag``), not a graph property, so every trivalent centre is
        emitted and the selection belongs to the layer holding the table.

        With no ``gen_*`` flags this is a no-op that returns ``self``. Pass
        ``clear_existing=True`` to drop previously generated relations of the
        requested kinds before re-perceiving.

        Args:
            gen_angle: When True, generate angle relations.
            gen_dihe: When True, generate dihedral relations.
            gen_improper: When True, generate improper relations.
            clear_existing: When True, clear existing relations of the
                requested kinds first.

        Returns:
            ``self``, with any requested topology written in place.
        """
        if gen_angle or gen_dihe or gen_improper or clear_existing:
            self.generate_topology(
                gen_angle=gen_angle,
                gen_dihedral=gen_dihe,
                gen_improper=gen_improper,
                clear_existing=clear_existing,
            )
        return self

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
            self._intern_node(h)
            for h, _ in self.topo_distances(entity.handle, max_hops=radius)
        ]

    def get_topo_distances(
        self,
        source: Atom,
        entity_type: type[Atom] = Atom,
        link_type: type[Link] = Bond,
    ) -> dict[Atom, int]:
        return {
            self._intern_node(h): int(d) for h, d in self.topo_distances(source.handle)
        }

    def extract_subgraph(
        self,
        center_entities: Iterable[Atom],
        radius: int,
        entity_type: type[Atom] = Atom,
        link_type: type[Link] = Bond,
        *,
        max_ring_size: int | None = None,
    ) -> tuple["Atomistic", list[Atom]]:
        """Induced radius-``radius`` ball around ``center_entities``.

        With ``max_ring_size``, a ring of at most that many atoms which the
        radius only partly reaches is pulled in whole (together with anything
        fused or bridged to it), so the slice never contains a cut small ring.
        Larger rings are cut like any other path — see
        :meth:`Atomistic.extract_subgraph` for why the bound is part of
        the claim rather than a tuning knob.
        """
        sub, boundary, _, _ = self._extract_mapped(
            list(center_entities), radius, type(self), max_ring_size=max_ring_size
        )
        return sub, boundary

    def _extract_mapped[G: "Atomistic"](
        self,
        centers: list[Atom],
        radius: int,
        out_cls: type[G],
        *,
        regenerate_topology: bool = False,
        max_ring_size: int | None = None,
    ) -> tuple[G, list[Atom], dict[Atom, Atom], dict[int, int]]:
        """Induced radius-``radius`` ball plus a region-atom → parent-atom map.

        Delegates BFS + materialisation to
        :meth:`Atomistic.extract_subgraph`. Returns
        ``(subgraph, boundary_atoms, {region_atom: parent_atom},
        {region_atom_handle: hops_from_nearest_center})``.

        ``max_ring_size`` closes the ball on ring systems built from rings no
        larger than that; the atoms it adds sit beyond ``radius``, so their hops
        exceed it.
        """
        new = out_cls()
        new.props = dict(self.props)
        if not centers:
            return new, [], {}, {}

        res = molrs.Atomistic.extract_subgraph(
            self,
            [c.handle for c in centers],
            int(radius),
            regenerate_topology=regenerate_topology,
            max_ring_size=max_ring_size,
        )
        molrs.Atomistic.adopt(new, res.graph)

        parent_by_old: dict[int, Atom] = {
            old: self._intern_node(old) for old in res.node_map
        }
        region_by_old: dict[int, Atom] = {
            old: new._intern_node(new_h)  # type: ignore[attr-defined]
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
        """Independent deep copy. **Handles are preserved** (native clone)."""
        bare = molrs.Atomistic.copy(self)
        new = type(self)()
        molrs.Atomistic.adopt(new, bare)
        # Whole-graph annotations live on ``props`` (see ``_GraphViews``).
        new.props = dict(self.props)
        return new

    def merge(self, other: "Atomistic") -> Self:
        """Structural merge of ``other`` into ``self`` (native).

        Every node of ``other`` is remapped to a fresh handle in ``self``.
        ``other`` is emptied and must not be used afterwards. Cross-graph
        identity is handle-based — Python view objects are **not** rebound.
        """
        self._merge_map(other)
        return self

    def _merge_map(self, other: "Atomistic") -> dict[int, int]:
        """Merge ``other`` and return its old-handle to new-handle mapping."""
        mapping = molrs.Atomistic.merge(self, other)
        other._node_refs.clear()
        other._relation_refs.clear()
        other.props.clear()
        return mapping

    @staticmethod
    def adopt(graph: molrs.Atomistic) -> "Atomistic":
        """Zero-copy take ownership of a native-produced ``Atomistic`` graph.

        Uses the native zero-copy ``adopt`` to move ``graph``'s storage into a
        fresh molpy ``Atomistic`` (``graph`` is left empty). Views over the
        adopted nodes/relations are interned lazily on access.
        """
        struct = Atomistic()
        molrs.Atomistic.adopt(struct, graph)
        return struct

    @classmethod
    def from_frame(cls, frame: "Frame") -> "Atomistic":
        """Build a molpy ``Atomistic`` from a :class:`~molpy.Frame`.

        The inverse of :meth:`to_frame`. The inherited native ``from_frame`` returns a
        bare native graph; this override adopts it so the result is a molpy
        ``Atomistic`` — the call site never needs a second ``adopt``.
        """
        return cls.adopt(molrs.Atomistic.from_frame(frame))

    # ---------- composition ----------
    def __iadd__(self, other: "Atomistic") -> "Atomistic":
        self.merge(other)
        return self

    def __add__(self, other: "Atomistic") -> "Atomistic":
        result = self.copy()
        result.merge(other.copy())  # merge empties its argument
        return result

    def replicate(self, n: int, transform: Any = None) -> "Atomistic":
        if transform is None:
            return type(self).adopt(molrs.Atomistic.replicate(self, n))
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

        Delegates straight to the native world's native ``to_frame``: the Rust
        column store already holds every component as a dense, row-aligned
        column, so each block is materialized as numpy with zero Python-side
        conversion. ``atom_fields`` optionally restricts the atoms block columns.
        """
        from molrs import Frame

        # ``molrs.Atomistic.to_frame`` yields the bare pyo3 frame; upgrade it to
        # the rich ``Frame`` (metadata, box, rich Blocks) callers expect.
        frame = Frame(molrs.Atomistic.to_frame(self))
        if atom_fields is not None and "atoms" in frame:
            keep = set(atom_fields)
            atoms = frame["atoms"]
            for col in [k for k in atoms.keys() if k not in keep]:
                del atoms[col]
        return frame
