"""Region-scoped force-field typing.

:func:`typify_region` types an :class:`~molpy.core.affected_region.AffectedRegion`
as an ordinary standalone :class:`~molpy.core.atomistic.Atomistic` — the context
``boundary`` shell gives the interior atoms complete ring/degree/SMARTS context,
so they receive the *same* types they would in the full structure — then reads
off the assigned types for every **non-boundary** atom plus the incident bonded
terms, keyed by the region's molrs **canonical order**.

The result, a frozen :class:`RegionTypes`, holds plain data only (type strings +
scalar params + canonical positions) and **no live** :class:`~molpy.core.entity.Entity`
references, so it caches across structures: an isomorphic region reuses it by
lining its own canonical order up against the cached one (see
:class:`~molpy.typifier.cache.RetypeCache`).

Boundary atoms have a bond-neighbour outside the ball, so their SMARTS context is
truncated and their types would be wrong; region typing runs the atom typifier
**non-strictly** so those atoms are simply left untyped and dropped here — never
written back.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from molpy.core.affected_region import AffectedRegion
    from molpy.core.atomistic import Atom, Atomistic
    from molpy.core.entity import Link
    from molpy.typifier.atomistic import ForceFieldTypifier


@runtime_checkable
class RegionTypifier(Protocol):
    """Structural type of the crosslinker/reacter retype hook.

    Anything that can type an :class:`~molpy.core.affected_region.AffectedRegion`
    and report its context depth: the SMARTS-based
    :class:`~molpy.typifier.atomistic.ForceFieldTypifier` and the AmberTools-backed
    :class:`~molpy.typifier.ambertools.AmberToolsTypifier` both satisfy it.
    """

    @property
    def context_radius(self) -> int: ...

    def typify_region(self, region: AffectedRegion) -> RegionTypes: ...


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

    def as_dict(self) -> dict[str, ParamValue]:
        """Flatten back to a ``data`` patch (``params`` plus ``type``)."""
        patch: dict[str, ParamValue] = dict(self.params)
        if self.type is not None:
            patch["type"] = self.type
        return patch


@dataclass(frozen=True)
class BondedTypeInfo:
    """A bonded term's assigned type, keyed by its endpoints' canonical order."""

    #: canonical-order positions of the term's endpoints (in link order).
    positions: tuple[int, ...]
    info: TypeInfo


@dataclass(frozen=True)
class RegionTypes:
    """Immutable, canonical-order snapshot of a region's assigned types.

    Only **non-boundary** atoms (and bonded terms all of whose endpoints are
    non-boundary) are recorded — boundary atoms carry truncated context and are
    discarded. ``atoms`` maps a canonical-order position to the type the atom at
    that position received; the same canonical labelling holds for any
    isomorphic region, which is what lets the cache reuse a snapshot across
    structures.
    """

    #: ``(canonical position, type info)`` for each non-boundary atom.
    atoms: tuple[tuple[int, TypeInfo], ...]
    bonds: tuple[BondedTypeInfo, ...]
    angles: tuple[BondedTypeInfo, ...]
    dihedrals: tuple[BondedTypeInfo, ...]


def _as_param(value: object) -> ParamValue | None:
    """Return ``value`` if it is a storable scalar param, else ``None``."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return None


def _scalar_delta(
    before: list[dict[str, object]], after: list[dict[str, object]]
) -> frozenset[str]:
    """Keys whose scalar value typing added or changed on at least one atom.

    The union over all atoms defines a single, complete field schema (``type``
    plus force-field params). Geometry / identity fields (coordinates, ``id``,
    ``element`` …) are untouched by typing and so never enter the schema — which
    is exactly why a snapshot is safe to apply onto a *different* isomorphic
    region's parent atoms.
    """
    owned: set[str] = set()
    for pre, post in zip(before, after, strict=True):
        for key, value in post.items():
            if _as_param(value) is not None and pre.get(key) != value:
                owned.add(key)
    return frozenset(owned)


def _atom_info(data: dict[str, object], param_keys: tuple[str, ...]) -> TypeInfo:
    """Build a :class:`TypeInfo` for one atom over the shared ``param_keys``."""
    params: list[tuple[str, ParamValue]] = []
    for key in param_keys:
        value = _as_param(data.get(key))
        if value is not None:
            params.append((key, value))
    raw_type = data.get("type")
    return TypeInfo(
        type=raw_type if isinstance(raw_type, str) else None,
        params=tuple(params),
    )


def _link_info(data: dict[str, object], positions: tuple[int, ...]) -> BondedTypeInfo:
    """Capture a bonded term's type + scalar params at ``positions``."""
    params: list[tuple[str, ParamValue]] = []
    for key, value in data.items():
        if key == "type":
            continue
        scalar = _as_param(value)
        if scalar is not None:
            params.append((key, scalar))
    raw_type = data.get("type")
    return BondedTypeInfo(
        positions=positions,
        info=TypeInfo(
            type=raw_type if isinstance(raw_type, str) else None,
            params=tuple(sorted(params)),
        ),
    )


def typify_region(typifier: ForceFieldTypifier, region: AffectedRegion) -> RegionTypes:
    """Type ``region`` as a standalone graph and snapshot its interior types.

    Runs the typifier's full pipeline on the region with atom typing forced
    **non-strict** (boundary atoms have truncated context and are meant to be
    left untyped), then records, in canonical order, the types of every
    non-boundary atom and every fully-interior bonded term.

    Args:
        typifier: the force-field typifier (must expose ``typify`` and, for the
            non-strict atom pass, ``atom_typifier``).
        region: the affected region to type.

    Returns:
        A frozen :class:`RegionTypes` snapshot (canonical-order keyed, no live
        entity references).

    Raises:
        ValueError: if a non-boundary (interior) atom is left untyped while the
            typifier's atom typing is strict — a signal that the extraction
            radius is too small to give interior atoms complete context.
    """
    region_atoms = list(region.atoms)
    before = [dict(atom.data) for atom in region_atoms]

    typed = _typify_nonstrict(typifier, region)
    typed_atoms = list(typed.atoms)
    after = [dict(atom.data) for atom in typed_atoms]

    canon = region.canonical_order()
    pos_of_handle = {atom.handle: pos for pos, atom in enumerate(region_atoms)}
    canon_of_pos = {pos_of_handle[handle]: idx for idx, handle in enumerate(canon)}
    boundary = {atom.handle for atom in region.boundary}

    param_keys = tuple(sorted(_scalar_delta(before, after) - {"type"}))

    atom_entries: list[tuple[int, TypeInfo]] = []
    for pos, region_atom in enumerate(region_atoms):
        if region_atom.handle in boundary:
            continue
        atom_entries.append((canon_of_pos[pos], _atom_info(after[pos], param_keys)))
    atom_entries.sort(key=lambda entry: entry[0])

    strict = bool(getattr(getattr(typifier, "atom_typifier", None), "strict", False))
    if strict:
        untyped = [idx for idx, info in atom_entries if info.type is None]
        if untyped:
            raise ValueError(
                "region interior atom(s) left untyped at canonical positions "
                f"{untyped}: extraction radius too small for full SMARTS context"
            )

    def _links(views: Iterable[Link[Atom]]) -> tuple[BondedTypeInfo, ...]:
        return _capture_links(views, typed_atoms, canon_of_pos, boundary, region_atoms)

    return RegionTypes(
        atoms=tuple(atom_entries),
        bonds=_links(typed.bonds),
        angles=_links(typed.angles),
        dihedrals=_links(typed.dihedrals),
    )


def apply_region_types(region_types: RegionTypes, region: AffectedRegion) -> None:
    """Write a :class:`RegionTypes` snapshot onto ``region``'s parent atoms.

    Lines each stored canonical position up against ``region``'s **own**
    :meth:`~molpy.core.affected_region.AffectedRegion.canonical_order` — so a
    snapshot captured from a *different* but isomorphic region still maps
    correctly — then reaches the parent atom through ``region.entity_map``.
    Boundary atoms are absent from the snapshot and never touched.
    """
    canon = region.canonical_order()
    handle_to_parent = {
        region_atom.handle: parent_atom
        for region_atom, parent_atom in region.entity_map.items()
    }
    for canon_index, info in region_types.atoms:
        handle_to_parent[canon[canon_index]].update(**info.as_dict())


def _typify_nonstrict(
    typifier: ForceFieldTypifier, region: AffectedRegion
) -> Atomistic:
    """Type ``region`` with atom typing forced non-strict, then restore.

    Only the atom typifier's strictness matters: the pipeline already skips
    bonded/pair typing for any atom left without a ``type``, so a
    truncated-context boundary atom never raises. The toggle is transient and
    restored even on error.
    """
    atom_typifier = getattr(typifier, "atom_typifier", None)
    if atom_typifier is None:
        return typifier.typify(region)
    saved = atom_typifier.strict
    atom_typifier.strict = False
    try:
        return typifier.typify(region)
    finally:
        atom_typifier.strict = saved


def _capture_links(
    links: Iterable[Link[Atom]],
    typed_atoms: list[Atom],
    canon_of_pos: dict[int, int],
    boundary: set[int],
    region_atoms: list[Atom],
) -> tuple[BondedTypeInfo, ...]:
    """Snapshot every bonded term all of whose endpoints are non-boundary."""
    pos_of_handle = {atom.handle: pos for pos, atom in enumerate(typed_atoms)}
    out: list[BondedTypeInfo] = []
    for link in links:
        positions = [pos_of_handle.get(ep.handle) for ep in link.endpoints]
        if any(pos is None for pos in positions):
            continue
        resolved = [pos for pos in positions if pos is not None]
        if any(region_atoms[pos].handle in boundary for pos in resolved):
            continue
        canon_positions = tuple(canon_of_pos[pos] for pos in resolved)
        out.append(_link_info(dict(link.data), canon_positions))
    return tuple(out)
