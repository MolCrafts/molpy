# Extending the Data Model

This page shows how to add new entity types, link types, and struct subclasses to MolPy's core data model.

## Architecture recap

MolPy's data model has three layers:

1. **Entity** (node) — `UserDict` subclass with identity-based hashing. Example: `Atom`, `Bead`.
2. **Link** (edge) — holds an ordered tuple of `Entity` endpoints. Example: `Bond`, `Angle`, `Dihedral`.
3. **Struct** (container) — holds `TypeBucket[Entity]` and `TypeBucket[Link]`, manages CRUD. Example: `Atomistic`, `CoarseGrain`.

`TypeBucket` stores items by their concrete type. When you register `Atom` in a bucket, calling `bucket[Atom]` returns all `Atom` instances. Subclasses are included in parent queries.


## Adding a new Entity type

Subclass `Entity`. No methods are required — the base class provides dict-like storage and identity-based hashing.

```python
from molpy.core.entity import Entity

class VirtualSite(Entity):
    """A massless interaction site (e.g., TIP4P oxygen lone pair)."""

    def __repr__(self) -> str:
        name = self.data.get("name", id(self))
        return f"<VirtualSite: {name}>"
```


## Adding a new Link type

Subclass `Link`. Enforce endpoint count and types in `__init__`. Add named endpoint properties for readability.

```python
from molpy.core.entity import Link
from molpy.core.atomistic import Atom

class Improper(Link):
    """Improper dihedral: one central atom with three outer atoms."""

    def __init__(self, center: Atom, a: Atom, b: Atom, c: Atom, /, **attrs):
        super().__init__([center, a, b, c], **attrs)

    @property
    def center(self) -> Atom:
        return self.endpoints[0]

    @property
    def outer(self) -> tuple[Atom, Atom, Atom]:
        return self.endpoints[1], self.endpoints[2], self.endpoints[3]
```


## Registering types in a Struct

New entity and link types must be registered in the struct's `__init__` so that `TypeBucket` creates a bucket for them. Without registration, `bucket[MyType]` returns an empty list.

```python
from molpy.core.entity import Struct, MembershipMixin, ConnectivityMixin

class ExtendedAtomistic(Atomistic):
    """Atomistic with virtual sites and impropers."""

    def __init__(self, **props):
        super().__init__(**props)
        self.entities.register_type(VirtualSite)
        self.links.register_type(Improper)

    @property
    def virtual_sites(self):
        return self.entities[VirtualSite]

    @property
    def impropers(self):
        return self.links[Improper]

    def def_virtual_site(self, **attrs) -> VirtualSite:
        vs = VirtualSite(**attrs)
        self.entities.add(vs)
        return vs

    def def_improper(self, center, a, b, c, /, **attrs) -> Improper:
        imp = Improper(center, a, b, c, **attrs)
        self.links.add(imp)
        return imp
```


## How TypeBucket works

`TypeBucket` uses `get_nearest_type(item)` to determine the bucket key (the item's concrete class). Key behaviors:

- `bucket.add(item)` — adds to the bucket for the item's concrete type, skips if already present (identity check)
- `bucket[SomeType]` — returns all items of `SomeType` and its subclasses
- `bucket.register_type(SomeType)` — ensures an empty bucket exists (so `.atoms` returns `[]` instead of raising)
- `bucket.remove(item)` — removes by identity (`is`), not by equality


## Struct.copy() and new types

`Struct.copy()` deep-copies all entities and links, remapping endpoint references. This works automatically for new `Link` subclasses as long as their `__init__` accepts either `(*endpoints, **attrs)` or `(endpoints, **attrs)`. The copy logic tries positional args first, then list form.

If your Link subclass has a different constructor signature (e.g., named parameters), you may need to override `copy()` in your Struct subclass.


## Checklist

- [ ] Entity subclass: `class MyEntity(Entity)` with `__repr__`
- [ ] Link subclass: `class MyLink(Link)` with endpoint assertions and properties
- [ ] Register in Struct's `__init__`: `self.entities.register_type(MyEntity)`
- [ ] Add `def_*` factory method on the Struct
- [ ] Add property accessor (e.g., `@property def my_links`)
- [ ] Verify `Struct.copy()` works with the new types
- [ ] Write tests in `tests/test_core/`
