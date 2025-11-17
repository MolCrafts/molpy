from collections import UserDict
from collections.abc import Iterable, Iterator
from copy import deepcopy
from typing import (
    Any,
    Protocol,
    Self,
    TypeVar,
    cast,
    overload,
)

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
from molpy.core.utils import get_nearest_type


class EntityLike(Protocol):
    """Protocol for objects that can act as Entities (Entity or subclass)."""

    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, item: Any) -> None: ...
    def __hash__(self) -> int: ...
    def get(self, key: str, default: Any = None) -> Any: ...


class Entity(UserDict):
    """Dictionary-like base object for all structure elements.

    Minimal by design; no IDs, no persistence, no global context.
    """

    # Keep identity-based hashing/equality from UserDict's object semantics
    def __hash__(self) -> int:  # pragma: no cover - trivial identity
        return id(self)


class LinkLike(Protocol):
    """Protocol for objects that can act as Links (Link or subclass)."""

    data: dict[str, Any]
    endpoints: tuple[EntityLike, ...]

    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, item: Any) -> None: ...
    def __hash__(self) -> int: ...
    def get(self, key: str, default: Any = None) -> Any: ...


class Link[T: Entity](UserDict):
    """Connectivity object holding direct references to endpoint entities.

    Attributes
    ----------
    endpoints: tuple[Entity]
        The ordered tuple of endpoint entity references.
    """

    endpoints: tuple[T, ...]

    def __init__(self, endpoints: Iterable[T], /, **attrs: Any):
        super().__init__()
        self.endpoints = tuple(endpoints)
        # store remaining attributes in the mapping
        for k, v in attrs.items():
            self.data[k] = v

    def replace_endpoint(self, old: T, new: T) -> None:
        """Replace one endpoint reference with another in-place."""
        self.endpoints = tuple(new if e is old else e for e in self.endpoints)

    def __hash__(self) -> int:  # pragma: no cover - trivial identity
        return id(self)


# Note: EntityLike is already defined above (line 18), removing duplicate definition
# The first definition is the canonical one with full protocol methods

E = TypeVar("E", bound=Entity)
U = TypeVar("U", bound=Entity)


# ---------- Column-friendly list ----------
class Entities[E: Entity](list[E]):
    """A list of Entity-like objects supporting column-style access via a string key.

    When accessing with a string key, returns a numpy array if numpy is available,
    otherwise returns a list.
    """

    @overload
    def __getitem__(self, key: int) -> E: ...  # type: ignore[override]
    @overload
    def __getitem__(self, key: slice) -> list[E]: ...  # type: ignore[override]
    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: int | slice | str) -> E | list[E] | Any:  # type: ignore[override]
        if isinstance(key, str):
            # Column access - returns numpy array if available
            values = [ent.get(key) for ent in self]
            try:
                import numpy as np

                return np.array(values)
            except ImportError:
                return values
        return super().__getitem__(key)


# ---------- Helper: choose bucket key (override if needed) ----------
# Note: get_nearest_type is now imported from utils


# ---------- Entity-specific TypeBucket (returns Entities type) ----------
class TypeBucket[E: Entity]:
    """
    Entity-specific TypeBucket that groups and stores objects by their concrete type.
    Uses Entities as container to support column-style access.

    Bucket objects by (concrete) type using dict[type, Entities].
    - Key:  type[U], where U <: E
    - Item: Entities[U], paired with the same U as in the key

    Query methods return Entities[...] so you can do column-style access directly.
    """

    def __init__(self) -> None:
        # Internal store uses Any for flexibility across entity types
        self._items: dict[type[Any], Entities[Any]] = {}

    # ----- mutate -----
    def add(self, item: E) -> None:
        """Add one object to the bucket for its nearest type."""
        cls = get_nearest_type(item)  # type: ignore[arg-type]
        bucket = self._items.setdefault(cls, Entities())
        # Check if item already exists (use identity check, not equality)
        for existing in bucket:
            if existing is item:
                return  # Already in bucket, skip
        bucket.append(item)

    def add_many(self, items: Iterable[E]) -> None:
        """Add multiple objects."""
        for it in items:
            self.add(it)

    def remove(self, item: E) -> bool:
        """Remove an object from its bucket; returns True if removed."""
        cls = get_nearest_type(item)  # type: ignore[arg-type]
        bucket = self._items.get(cls)
        if not bucket:
            return False
        # Use identity comparison (is) not equality (==)
        # Find and remove the exact object instance
        for i, obj in enumerate(bucket):
            if obj is item:
                bucket.pop(i)
                if not bucket:
                    self._items.pop(cls, None)
                return True
        return False

    def register_type(self, cls: type[Any]) -> None:
        """Ensure a bucket exists for the given class."""
        self._items.setdefault(cls, Entities())

    # ----- queries (return Entities) -----
    def all(self) -> Entities[E]:
        """All items across all buckets (returns a new Entities)."""
        out: Entities[E] = Entities()
        for b in self._items.values():
            out.extend(cast(Entities[E], b))
        return out

    def exact_bucket(self, cls: type[U]) -> Entities[U]:
        """Items whose concrete class is exactly 'cls' (no subclasses)."""
        b = self._items.get(cls)
        return Entities(cast(Entities[U], b)) if b else Entities()

    def bucket(self, cls: type[U]) -> Entities[U]:
        """
        Items whose concrete class is 'cls' or any subclass of 'cls'.
        Returns a new Entities[U].
        """
        out: Entities[U] = Entities()
        if cls in self._items:
            out.extend(cast(Entities[U], self._items[cls]))
        for k, b in self._items.items():
            try:
                if k is not cls and isinstance(k, type) and issubclass(k, cls):  # type: ignore[arg-type]
                    out.extend(cast(Entities[U], b))
            except TypeError:
                # Skip if k is not a proper class
                pass
        return out

    def classes(self) -> Iterator[type[E]]:
        """Concrete classes that currently have buckets."""
        return cast(Iterator[type[E]], iter(self._items.keys()))

    def __len__(self) -> int:
        """Total number of stored objects across all buckets."""
        return sum(len(b) for b in self._items.values())

    def __getitem__(self, cls: type[U]) -> Entities[U]:
        """Get bucket for class (includes subclasses)."""
        return self.bucket(cls)

    def __setitem__(self, cls: type[U], items: Iterable[U]) -> None:
        """Set the bucket for a given class."""
        self._items[cls] = Entities(items)


class StructLike(Protocol):
    """Protocol for objects that can act as Structs (Struct or subclass).

    Defines the interface for structural containers that hold entities and links.
    """

    entities: TypeBucket[Any]
    links: TypeBucket[Any]


T = TypeVar("T", bound="Struct")


class Struct:
    """Container holding entities and links via typed buckets.

    This is the root class for all molecular structure types in MolPy.
    Supports entity/link management and serves as the base for wrappers.

    A Struct is a typed container that organizes entities (e.g., atoms, residues)
    and links (e.g., bonds, angles) into type-specific buckets for efficient
    access and manipulation.
    """

    def __init__(self, **props: Any) -> None:
        """Initialize a new Struct.

        Args:
            **props: Additional properties to store in the struct
        """
        self.entities: TypeBucket[Any] = TypeBucket()
        self.links: TypeBucket[Any] = TypeBucket()
        self._props: dict[str, Any] = dict(props)

    # ---------- dict-like access to props ----------
    def __getitem__(self, key: str) -> Any:
        """Get property by key.

        Args:
            key: Property key

        Returns:
            Property value

        Raises:
            KeyError: If key doesn't exist
        """
        return self._props[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set property by key.

        Args:
            key: Property key
            value: Property value
        """
        self._props[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in props.

        Args:
            key: Property key to check

        Returns:
            True if key exists, False otherwise
        """
        return key in self._props

    def get(self, key: str, default: Any = None) -> Any:
        """Get property with default.

        Args:
            key: Property key
            default: Default value if key not found

        Returns:
            Property value or default
        """
        return self._props.get(key, default)

    # ---------- helpers ----------
    def _iter_all_entities(self) -> Iterable[Entity]:
        for cls in self.entities.classes():
            yield from self.entities.bucket(cls)

    def _iter_all_links(self) -> Iterable[Link]:
        for cls in self.links.classes():
            yield from self.links.bucket(cls)

    # ---------- built-ins ----------
    def copy(self) -> Self:
        new = type(self)()
        # deep-copy entities
        emap: dict[Entity, Entity] = {}
        for ent in self._iter_all_entities():
            cloned = ent.__class__(deepcopy(getattr(ent, "data", None)))
            emap[ent] = cloned
            new.entities.add(cloned)

        # deep-copy links (remap endpoints)
        for link in self._iter_all_links():
            # Ensure all endpoints are in emap (defensive check)
            for ep in link.endpoints:
                if ep not in emap:
                    # Edge case: endpoint not in entities bucket
                    # This shouldn't happen in well-formed assemblies
                    cloned_ep = ep.__class__(deepcopy(getattr(ep, "data", None)))
                    emap[ep] = cloned_ep
                    new.entities.add(cloned_ep)

            mapped_eps = [emap[ep] for ep in link.endpoints]
            attrs = deepcopy(getattr(link, "data", {}))
            lcls: type[Link] = type(link)
            try:
                new_link = lcls(*mapped_eps, **attrs)  # Endpoints as positional args
            except TypeError:
                new_link = lcls(mapped_eps, **attrs)  # Or as list
            new.links.add(new_link)

        return new

    def merge(self, other: "Struct") -> Self:
        """
        Transfer all entities and links from another struct into self.

        **NO deep copy** - entities and links are directly transferred.
        After merge, `other` should not be used (its entities now belong to self).

        Args:
            other: Struct to merge into self

        Returns:
            Self for method chaining

        Raises:
            ValueError: If struct contains orphan links (endpoints not in entities)

        Example:
            >>> struct1.merge(struct2)  # Transfers struct2 into struct1
            >>> # struct2 should not be used after this!
        """
        # Collect all entities from other
        other_entities = set(other._iter_all_entities())

        # Transfer all entities directly (NO copy)
        for ent in other_entities:
            self.entities.add(ent)

        # Transfer all links directly (NO copy)
        for link in other._iter_all_links():
            # Verify all endpoints are in entities
            missing_endpoints = []
            for ep in link.endpoints:
                if ep not in other_entities:
                    missing_endpoints.append(ep)

            if missing_endpoints:
                # This indicates a malformed struct with orphan links
                raise ValueError(
                    "Found link with endpoints not in entities bucket. "
                    "This indicates orphan links in the struct."
                )

            self.links.add(link)

        return self


class SpatialMixin:
    """Geometry operations on entities with a "xyz" key only."""

    entities: TypeBucket[Any]
    links: TypeBucket[Any]

    def move(self, delta: list[float], *, entity_type: type[Entity]) -> Self:
        for e in self.entities.bucket(entity_type):
            e["xyz"] = _vec_add(e.get("xyz", [0, 0, 0]), delta)
        return self

    def rotate(
        self,
        axis: list[float],
        angle: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity],
    ) -> Self:
        k = _unit(axis)
        o = [0.0, 0.0, 0.0] if about is None else about
        for e in self.entities.bucket(entity_type):
            xyz = e.get("xyz")
            if isinstance(xyz, list) and len(xyz) == 3:
                e["xyz"] = _rodrigues_rotate(xyz, k, angle, o)
        return self

    def scale(
        self,
        factor: float,
        about: list[float] | None = None,
        *,
        entity_type: type[Entity],
    ) -> Self:
        o = [0.0, 0.0, 0.0] if about is None else about
        for e in self.entities.bucket(entity_type):
            xyz = e.get("xyz")
            if isinstance(xyz, list) and len(xyz) == 3:
                v = _vec_sub(xyz, o)
                e["xyz"] = _vec_add(o, _vec_scale(v, factor))
        return self

    def align(
        self,
        a: Entity,
        b: Entity,
        *,
        a_dir: list[float] | None = None,
        b_dir: list[float] | None = None,
        flip: bool = False,
        entity_type: type[Entity],
    ) -> Self:
        pa = a.get("xyz")
        pb = b.get("xyz")
        if not (
            isinstance(pa, list)
            and isinstance(pb, list)
            and len(pa) == 3
            and len(pb) == 3
        ):
            return self  # silently skip if missing positions

        ents = self.entities.bucket(entity_type)

        # rotate if directions provided
        if a_dir is not None and b_dir is not None:
            va = _unit(a_dir)
            vb = _unit(b_dir)
            if flip:
                vb = _vec_scale(vb, -1.0)
            # axis = va x vb; angle = atan2(|axis|, dot)
            axis = _cross(va, vb)
            na = _norm(axis)
            if na > 0:
                # angle via sin/cos components
                from math import atan2

                angle = atan2(na, _dot(va, vb))
                for e in ents:
                    xyz = e.get("xyz")
                    if isinstance(xyz, list) and len(xyz) == 3:
                        e["xyz"] = _rodrigues_rotate(
                            xyz, _vec_scale(axis, 1.0 / na), angle, pa
                        )
        # translate so that a -> b
        new_pa = a.get("xyz")
        if isinstance(new_pa, list) and len(new_pa) == 3:
            delta = _vec_sub(pb, new_pa)
            self.move(delta, entity_type=entity_type)

        return self


class MembershipMixin:
    """CRUD operations for entities and links within a StructLike."""

    entities: TypeBucket[Any]
    links: TypeBucket[Any]

    def register_type(self, cls: type[Any]) -> None:
        self.entities._items.setdefault(cls, Entities())

    # Entities -------------------------------------------------------------
    def add_entity(self, *ents: Entity) -> None:
        for e in ents:
            self.entities.add(e)

    def remove_entity(self, *ents: Entity, drop_incident_links: bool = True) -> None:
        to_remove = set(ents)
        # optionally drop incident links
        if drop_incident_links:
            # Convert to list to avoid RuntimeError: dictionary changed size during iteration
            for lcls in list(self.links.classes()):
                bucket = self.links.bucket(lcls)
                doomed: list[Link] = []
                for l in bucket:
                    if any(ep in to_remove for ep in l.endpoints):
                        doomed.append(l)
                if doomed:
                    self.remove_link(*doomed)
        # finally discard entities
        for e in ents:
            self.entities.remove(e)

    # Links ----------------------------------------------------------------
    def add_link(self, *links: Link, include_endpoints: bool = True) -> None:
        for l in links:
            self.links.add(l)
            if include_endpoints:
                for ep in l.endpoints:
                    self.entities.add(ep)

    def remove_link(self, *links: Link) -> None:
        for l in links:
            self.links.remove(l)

    # Normalize ------------------------------------------------------------
    def normalize(self, include_missing_endpoints: bool = False) -> None:
        present: set[Entity] = set()
        for ecls in self.entities.classes():
            present.update(self.entities.bucket(ecls))
        for lcls in self.links.classes():
            bucket = self.links.bucket(lcls)
            doomed: list[Link] = []
            for l in bucket:
                missing = [ep for ep in l.endpoints if ep not in present]
                if missing:
                    if include_missing_endpoints:
                        for ep in missing:
                            self.entities.add(ep)
                            present.add(ep)
                    else:
                        doomed.append(l)
            if doomed:
                self.remove_link(*doomed)


class ConnectivityMixin:
    entities: TypeBucket[Any]
    links: TypeBucket[Any]

    def get_neighbors(
        self, entity: Entity, link_type: type[Link] = Link
    ) -> list[Entity]:  # type: ignore[assignment]
        neighbors: list[Entity] = []
        try:
            bucket = self.links.bucket(link_type)  # type: ignore[arg-type]
        except KeyError:
            return neighbors
        for link in bucket:
            # Use identity check (is) not equality check (==)
            if any(ep is entity for ep in link.endpoints):
                for ep in link.endpoints:
                    if ep is not entity:
                        neighbors.append(ep)  # type: ignore[arg-type]
        return neighbors
