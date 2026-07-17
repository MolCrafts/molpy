"""Region-scoped force-field typing: the immutable snapshot and its write-back.

A typifier types an :class:`~molpy.typifier.affected_region.AffectedRegion` as an
ordinary standalone :class:`~molpy.core.atomistic.Atomistic` — the context shell
around the region gives its interior atoms complete ring/degree/SMARTS context,
so they receive the *same* types they would in the full structure — then snapshots
the types of every **interior** atom (within ``interior_reach`` hops of the edit)
plus the bonded terms wholly inside that set, keyed by the region's molrs
**canonical order**.

The result, a frozen :class:`RegionTypes`, holds plain data only (type strings +
scalar params + canonical positions) and **no live**
:class:`~molpy.core.entity.Entity` references, so it caches across structures: an
isomorphic region reuses it by lining its own canonical order up against the
cached one (see :class:`~molpy.typifier.cache.RetypeCache`).

Atoms outside the write-back set are never recorded. They exist only to give the
interior its context, and their own context is truncated — a truncated SMARTS
environment does not fail to match, it matches the **wrong rule**.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import molrs

from molpy.core import fields
from molpy.core.atomistic import Atomistic

if TYPE_CHECKING:
    from molpy.typifier.affected_region import AffectedRegion
    from molpy.core.atomistic import Atom, Atomistic
    from molpy.core.entity import Link
    from molpy.typifier.base import Typifier


#: A force-field parameter value a typifier writes onto an element. molrs Block
#: columns are scalar-typed, so a param is always one of these.
type ParamValue = str | int | float | bool


@dataclass(frozen=True)
class TypeInfo:
    """The type string + force-field params one typed element received.

    ``params`` is a sorted tuple of ``(key, value)`` pairs (never live objects),
    so :class:`TypeInfo` is immutable and hashable.
    """

    type: str | None
    params: tuple[tuple[str, ParamValue], ...]

    @staticmethod
    def as_param(value: object) -> ParamValue | None:
        """Return ``value`` if it is a storable scalar param, else ``None``."""
        if isinstance(value, (str, int, float, bool)):
            return value
        return None

    @classmethod
    def from_data(cls, data: dict[str, object], param_keys: tuple[str, ...]) -> Self:
        """Capture one atom's type + the shared ``param_keys`` it carries."""
        params: list[tuple[str, ParamValue]] = []
        for key in param_keys:
            value = cls.as_param(data.get(key))
            if value is not None:
                params.append((key, value))
        raw_type = data.get(fields.TYPE.key)
        return cls(
            type=raw_type if isinstance(raw_type, str) else None,
            params=tuple(params),
        )

    def as_dict(self) -> dict[str, ParamValue]:
        """Flatten back to a ``data`` patch (``params`` plus ``type``)."""
        patch: dict[str, ParamValue] = dict(self.params)
        if self.type is not None:
            patch[fields.TYPE.key] = self.type
        return patch


@dataclass(frozen=True)
class BondedTypeInfo:
    """A bonded term's assigned type, keyed by its endpoints' canonical order."""

    #: canonical-order positions of the term's endpoints (in link order).
    positions: tuple[int, ...]
    info: TypeInfo

    @classmethod
    def from_link(cls, data: dict[str, object], positions: tuple[int, ...]) -> Self:
        """Capture a bonded term's type + scalar params at ``positions``."""
        params: list[tuple[str, ParamValue]] = []
        for key, value in data.items():
            if key == fields.TYPE.key:
                continue
            scalar = TypeInfo.as_param(value)
            if scalar is not None:
                params.append((key, scalar))
        raw_type = data.get(fields.TYPE.key)
        return cls(
            positions=positions,
            info=TypeInfo(
                type=raw_type if isinstance(raw_type, str) else None,
                params=tuple(sorted(params)),
            ),
        )


@dataclass(frozen=True)
class RegionTypes:
    """Immutable, canonical-order snapshot of a region's assigned types.

    Only **interior** atoms (within ``interior_reach`` of the edit) and bonded
    terms all of whose endpoints are interior are recorded. ``atoms`` maps a
    canonical-order position to the type the atom at that position received; the
    same canonical labelling holds for any isomorphic region, which is what lets
    the cache reuse a snapshot across structures.
    """

    #: ``(canonical position, type info)`` for each interior atom.
    atoms: tuple[tuple[int, TypeInfo], ...]
    bonds: tuple[BondedTypeInfo, ...]
    angles: tuple[BondedTypeInfo, ...]
    dihedrals: tuple[BondedTypeInfo, ...]
    #: sp2 planarity terms; absent from a snapshot means a pyramidalised centre.
    impropers: tuple[BondedTypeInfo, ...]

    @classmethod
    def of(cls, region: AffectedRegion, typifier: Typifier) -> Self:
        """Type ``region`` with ``typifier`` and snapshot its interior.

        Region typing is not a kind of typifier — an
        :class:`~molpy.typifier.affected_region.AffectedRegion` *is* a MolGraph,
        so ``typifier.typify(region)`` is already legal. The only thing region
        typing adds is this cacheable snapshot of what landed on the interior.

        A region is a cut, always — so missing hydrogens are perceived here,
        always, with no condition to get wrong. The typifier is never asked to
        guess whether what it holds is a fragment.

        Hydrogen perception is not a convenience. ``extract_radius == interior_reach +
        reach`` means an interior atom's receptive field reaches *exactly* to the
        boundary atoms; a raw cut leaves those with unfilled valences, and a
        SMARTS matcher reads them as radicals. Measured on p-xylene at
        ``reach = 2``, 12 of 19 raw slices cannot be typed at all.
        """
        perceived = Atomistic.adopt(molrs.Perceive().find_hydrogens(region))
        before = [dict(atom.data) for atom in region.atoms]
        typed = typifier.typify(perceived)
        # Perception appends hydrogens, so the region's own atoms stay the
        # prefix. Added context is never part of the snapshot.
        after = [dict(atom.data) for atom in list(typed.atoms)[: len(before)]]
        return cls.capture(region, typed, before, after)

    @classmethod
    def capture(
        cls,
        region: AffectedRegion,
        typed: Atomistic,
        before: list[dict[str, object]],
        after: list[dict[str, object]],
    ) -> Self:
        """Snapshot ``typed``'s interior types in ``region``'s canonical order.

        ``typed`` is ``region`` after a typifier ran on it (possibly a copy); its
        atoms are in the same positional order as ``region``'s. ``before`` /
        ``after`` are the per-atom data dicts either side of typing, used to
        derive the shared param-key schema.

        Raises:
            ValueError: if an interior atom was left untyped. That means the
                extracted ball was too small to give it complete context — the
                one condition region typing may never paper over.
        """
        region_atoms = list(region.atoms)
        typed_atoms = list(typed.atoms)
        canon = region.canonical_order()
        pos_of_handle = {atom.handle: pos for pos, atom in enumerate(region_atoms)}
        canon_of_pos = {pos_of_handle[handle]: idx for idx, handle in enumerate(canon)}
        interior = {atom.handle for atom in region.interior}

        param_keys = tuple(sorted(cls._scalar_delta(before, after) - {fields.TYPE.key}))

        entries: list[tuple[int, TypeInfo]] = []
        for pos, region_atom in enumerate(region_atoms):
            if region_atom.handle not in interior:
                continue
            entries.append(
                (canon_of_pos[pos], TypeInfo.from_data(after[pos], param_keys))
            )
        entries.sort(key=lambda entry: entry[0])

        untyped = [idx for idx, info in entries if info.type is None]
        if untyped:
            raise ValueError(
                "region interior atom(s) left untyped at canonical positions "
                f"{untyped}: extract_radius={region.extract_radius} is too small "
                f"for interior_reach={region.interior_reach}"
            )

        def links(views: Iterable[Link[Atom]]) -> tuple[BondedTypeInfo, ...]:
            return cls._capture_links(
                views, typed_atoms, canon_of_pos, interior, region_atoms
            )

        return cls(
            atoms=tuple(entries),
            bonds=links(typed.bonds),
            angles=links(typed.angles),
            dihedrals=links(typed.dihedrals),
            impropers=links(typed.impropers),
        )

    def apply_to(self, region: AffectedRegion) -> None:
        """Write this snapshot onto ``region``'s parent atoms.

        Lines each stored canonical position up against ``region``'s **own**
        :meth:`~molpy.typifier.affected_region.AffectedRegion.canonical_order` — so a
        snapshot captured from a *different* but isomorphic region still maps
        correctly — then reaches the parent atom through ``region.entity_map``.
        Atoms outside the write-back set are absent from the snapshot and are
        never touched.
        """
        canon = region.canonical_order()
        handle_to_parent = {
            region_atom.handle: parent_atom
            for region_atom, parent_atom in region.entity_map.items()
        }
        for canon_index, info in self.atoms:
            handle_to_parent[canon[canon_index]].update(**info.as_dict())

    @staticmethod
    def _scalar_delta(
        before: list[dict[str, object]], after: list[dict[str, object]]
    ) -> frozenset[str]:
        """Keys whose scalar value typing added or changed on at least one atom.

        The union over all atoms defines a single, complete field schema (``type``
        plus force-field params). Geometry / identity fields (coordinates, ``id``,
        ``element`` …) are untouched by typing and so never enter the schema —
        which is exactly why a snapshot is safe to apply onto a *different*
        isomorphic region's parent atoms.
        """
        owned: set[str] = set()
        for pre, post in zip(before, after, strict=True):
            for key, value in post.items():
                if TypeInfo.as_param(value) is not None and pre.get(key) != value:
                    owned.add(key)
        return frozenset(owned)

    @staticmethod
    def _capture_links(
        links: Iterable[Link[Atom]],
        typed_atoms: list[Atom],
        canon_of_pos: dict[int, int],
        interior: set[int],
        region_atoms: list[Atom],
    ) -> tuple[BondedTypeInfo, ...]:
        """Snapshot every bonded term all of whose endpoints are interior.

        ``typed`` may carry the cap atoms the region's completion appended; they
        sit past the region's own atoms, so a term touching one is not a term of
        the region and is skipped.
        """
        pos_of_handle = {atom.handle: pos for pos, atom in enumerate(typed_atoms)}
        n_region = len(region_atoms)
        out: list[BondedTypeInfo] = []
        for link in links:
            if link.get(fields.TYPE.key) is None:
                # Undecided, not "typed as None". Writing that back would erase
                # whatever the parent's term already carried.
                continue
            positions = [pos_of_handle.get(ep.handle) for ep in link.endpoints]
            if any(pos is None or pos >= n_region for pos in positions):
                continue
            resolved = [pos for pos in positions if pos is not None]
            if any(region_atoms[pos].handle not in interior for pos in resolved):
                continue
            canon_positions = tuple(canon_of_pos[pos] for pos in resolved)
            out.append(BondedTypeInfo.from_link(dict(link.data), canon_positions))
        return tuple(out)
