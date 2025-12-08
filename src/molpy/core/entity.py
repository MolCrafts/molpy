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
            e["x"] = e["x"] + delta[0]
            e["y"] = e["y"] + delta[1]
            e["z"] = e["z"] + delta[2]
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
            xyz = [e["x"], e["y"], e["z"]]
            xyz = _rodrigues_rotate(xyz, k, angle, o)
            e["x"] = xyz[0]
            e["y"] = xyz[1]
            e["z"] = xyz[2]
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
            xyz = [e["x"], e["y"], e["z"]]
            xyz = _vec_sub(xyz, o)
            xyz = _vec_add(o, _vec_scale(xyz, factor))
            e["x"] = xyz[0]
            e["y"] = xyz[1]
            e["z"] = xyz[2]
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
        pa = [a["x"], a["y"], a["z"]]
        pb = [b["x"], b["y"], b["z"]]
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
                    xyz = [e["x"], e["y"], e["z"]]
                    xyz = _rodrigues_rotate(xyz, _vec_scale(axis, 1.0 / na), angle, pa)
                    e["x"] = xyz[0]
                    e["y"] = xyz[1]
                    e["z"] = xyz[2]
        # translate so that a -> b
        delta = _vec_sub(pb, pa)
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

    def get_topo(
        self,
        entity_type: type[Entity] = Entity,
        link_type: type[Link] = Link,
    ) -> "Topology":
        """将结构导出为拓扑图。

        Args:
            entity_type: 要包含在拓扑图中的实体类型
            link_type: 用于构建连接的链接类型

        Returns:
            Topology: igraph.Graph 对象，包含实体到顶点的映射信息

        Note:
            返回的 Topology 图的顶点顺序与 entities[entity_type] 的顺序一致。
            可以使用 `entity_to_idx` 和 `idx_to_entity` 属性访问映射关系。
        """
        from molpy.core.topology import Topology

        # 收集所有相关实体
        entities_list = list(self.entities.bucket(entity_type))
        entity_to_idx: dict[Entity, int] = {
            ent: i for i, ent in enumerate(entities_list)
        }
        entity_set = set(entities_list)

        # 构建边列表（只考虑连接两个端点的链接）
        edges: list[tuple[int, int]] = []
        for link in self.links.bucket(link_type):  # type: ignore[arg-type]
            endpoints = link.endpoints
            # 只处理包含两个端点且都在实体集合中的链接
            if len(endpoints) >= 2:
                ep1, ep2 = endpoints[0], endpoints[1]
                if ep1 in entity_set and ep2 in entity_set:
                    idx1 = entity_to_idx[ep1]
                    idx2 = entity_to_idx[ep2]
                    if idx1 != idx2:  # 避免自环
                        edges.append((idx1, idx2))

        # 创建拓扑图，使用正式成员存储映射关系
        topo = Topology(
            n=len(entities_list),
            edges=edges,
            directed=False,
            entity_to_idx=entity_to_idx,
            idx_to_entity=entities_list,
        )

        return topo

    def get_topo_neighbors(
        self,
        entity: Entity,
        radius: int = 1,
        entity_type: type[Entity] = Entity,
        link_type: type[Link] = Link,
    ) -> list[Entity]:
        """获取指定实体在给定拓扑半径内的所有邻居。

        Args:
            entity: 中心实体
            radius: 拓扑半径（跳数）
            entity_type: 要考虑的实体类型
            link_type: 用于拓扑连接的链接类型

        Returns:
            list[Entity]: 在半径内的所有邻居实体列表（包括自身，如果半径>=0）
        """
        topo = self.get_topo(entity_type=entity_type, link_type=link_type)

        # 获取实体在图中的索引
        entity_to_idx: dict[Entity, int] = topo.entity_to_idx
        if entity not in entity_to_idx:
            return []

        center_idx = entity_to_idx[entity]

        # 获取距离
        distances = topo.distances(source=[center_idx])[0]

        # 收集在半径内的所有实体
        neighbors: list[Entity] = []
        idx_to_entity: list[Entity] = topo.idx_to_entity
        for i, dist in enumerate(distances):
            if dist <= radius and dist < float("inf"):
                neighbors.append(idx_to_entity[i])

        return neighbors

    def get_topo_distances(
        self,
        source: Entity,
        entity_type: type[Entity] = Entity,
        link_type: type[Link] = Link,
    ) -> dict[Entity, int]:
        """获取从源实体到所有其他实体的拓扑距离。

        Args:
            source: 源实体
            entity_type: 要考虑的实体类型
            link_type: 用于拓扑连接的链接类型

        Returns:
            dict[Entity, int]: 从源实体到每个实体的拓扑距离字典。
                如果实体不可达，距离为无穷大（float('inf')）。
        """
        topo = self.get_topo(entity_type=entity_type, link_type=link_type)

        # 获取源实体在图中的索引
        entity_to_idx: dict[Entity, int] = topo.entity_to_idx
        if source not in entity_to_idx:
            return {}

        source_idx = entity_to_idx[source]

        # 计算距离
        distances = topo.distances(source=[source_idx])[0]

        # 构建距离字典
        idx_to_entity: list[Entity] = topo.idx_to_entity
        result: dict[Entity, int] = {}
        for i, dist in enumerate(distances):
            if dist < float("inf"):
                result[idx_to_entity[i]] = int(dist)

        return result

    def extract_subgraph(
        self,
        center_entities: Iterable[Entity],
        radius: int,
        entity_type: type[Entity] = Entity,
        link_type: type[Link] = Link,
    ) -> tuple["Struct", list[Entity]]:
        """提取在指定拓扑半径内的子图。

        Args:
            center_entities: 中心实体集合
            radius: 拓扑半径（跳数）
            entity_type: 要考虑的实体类型
            link_type: 用于拓扑连接的链接类型

        Returns:
            tuple[Struct, list[Entity]]: 包含：
                - subgraph: 提取的子图（新的 Struct 实例）
                - edge_entities: 边界实体列表（在原图中有邻居但不在子图中的实体）

        Note:
            子图包含所有在半径内的实体和它们之间的链接。
            边界实体是指在原图中还有邻居不在子图里的实体。
        """
        center_entities_list = list(center_entities)
        topo = self.get_topo(entity_type=entity_type, link_type=link_type)

        entity_to_idx: dict[Entity, int] = topo.entity_to_idx
        idx_to_entity: list[Entity] = topo.idx_to_entity

        # 获取中心实体的索引
        center_indices: list[int] = []
        for cent in center_entities_list:
            if cent in entity_to_idx:
                center_indices.append(entity_to_idx[cent])

        if not center_indices:
            # 如果没有有效的中心实体，返回空子图
            from copy import deepcopy

            new_struct = type(self)()
            return new_struct, []

        # 收集在半径内的所有实体索引
        selected_indices: set[int] = set()
        for c in center_indices:
            distances = topo.distances(source=[c])[0]
            for i, d in enumerate(distances):
                if d <= radius and d < float("inf"):
                    selected_indices.add(i)

        selected_indices_list = sorted(selected_indices)
        selected_entities = [idx_to_entity[i] for i in selected_indices_list]
        selected_entities_set = set(selected_entities)

        # 找到边界实体（在子图中有邻居不在子图里的实体）
        selected_indices_set = set(selected_indices)
        edge_indices: set[int] = set()
        for i in selected_indices:
            for j in topo.neighbors(i):
                if j not in selected_indices_set:
                    edge_indices.add(i)
                    break

        edge_entities = [idx_to_entity[i] for i in sorted(edge_indices)]

        # 创建新的 Struct 实例
        from copy import deepcopy

        new_struct = type(self)()
        new_struct._props = deepcopy(self._props)

        # 添加选中的实体并创建实体映射（原实体 -> 克隆实体）
        entity_map: dict[Entity, Entity] = {}
        cloned_entities_list: list[Entity] = []
        for ent in selected_entities:
            # 深拷贝实体
            cloned_ent = ent.__class__(deepcopy(getattr(ent, "data", {})))
            new_struct.entities.add(cloned_ent)
            entity_map[ent] = cloned_ent
            cloned_entities_list.append(cloned_ent)

        # 添加在子图中的链接
        for link in self.links.bucket(link_type):  # type: ignore[arg-type]
            endpoints = link.endpoints
            if len(endpoints) >= 2:
                # 检查链接的所有端点是否都在子图中
                if all(ep in selected_entities_set for ep in endpoints):
                    # 创建链接的克隆，映射端点到新实体
                    cloned_eps = [entity_map[ep] for ep in endpoints]
                    attrs = deepcopy(getattr(link, "data", {}))
                    lcls: type[Link] = type(link)
                    try:
                        new_link = lcls(*cloned_eps, **attrs)
                    except TypeError:
                        new_link = lcls(cloned_eps, **attrs)
                    new_struct.links.add(new_link)

        # 更新边界实体列表为克隆后的实体
        cloned_edge_entities = [
            entity_map[ep] for ep in edge_entities if ep in entity_map
        ]

        return new_struct, cloned_edge_entities
