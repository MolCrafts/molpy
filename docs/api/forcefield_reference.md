# ForceField Module API Reference

This document provides detailed API documentation for the `molpy.core.forcefield` module.

## Module Overview

The `molpy.core.forcefield` module provides classes for managing molecular force field parameters and interactions. The module follows a hierarchical structure:

- **Types**: Individual parameter sets for specific interactions
- **Styles**: Collections of related types
- **ForceField**: Complete force field containing multiple styles

## Classes

### Base Classes

#### `DictWithList`

A utility class that combines dictionary and list access patterns.

```python
class DictWithList(UserDict):
    def __init__(self, parms: list, kwparms: dict)
    def __getitem__(self, key: int | str) -> object
```

**Parameters:**
- `parms`: List of positional parameters
- `kwparms`: Dictionary of keyword parameters

**Methods:**
- `__getitem__(key)`: Access by integer index (for parms) or string key (for kwparms)

---

#### `Type`

Base class for all force field types.

```python
class Type(DictWithList):
    def __init__(self, name: str, parms: list = [], **kwparms)
    def __hash__(self) -> int
    def __repr__(self) -> str
    def __str__(self) -> str
    def __eq__(self, other: object) -> bool
    def match(self, other: Entity) -> bool
    @property
    def name(self) -> str
    @name.setter
    def name(self, value: str)
```

**Parameters:**
- `name`: Unique identifier for the type
- `parms`: Positional parameters
- `**kwparms`: Named parameters

**Properties:**
- `name`: Get/set the type name

**Methods:**
- `match(other)`: Check if this type matches a given entity (must be implemented by subclasses)
- `__hash__()`: Returns hash based on name
- `__eq__(other)`: Compare types by name or with strings
- `__repr__()`: String representation in format `<ClassName: name>`
- `__str__()`: Returns the name as string

---

#### `TypeContainer`

Container for managing collections of types.

```python
class TypeContainer:
    def __init__(self)
    def add(self, t: Type)
    def __iter__(self) -> iterator
    def __len__(self) -> int
    def get(self, name: str, default=None) -> Type | None
    def get_all_by(self, condition: Callable) -> list[Type]
    def update(self, other: 'TypeContainer')
```

**Methods:**
- `add(t)`: Add a type to the container
- `get(name, default=None)`: Retrieve type by name
- `get_all_by(condition)`: Get all types matching a condition
- `update(other)`: Merge types from another container
- `__iter__()`: Iterate over types
- `__len__()`: Number of types in container

---

#### `Style`

Base class for grouping related types under a common style.

```python
class Style(DictWithList):
    def __init__(self, name: str, parms=[], **kwparms)
    def __repr__(self) -> str
    def __eq__(self, other: object) -> bool
    def __hash__(self) -> int
    def merge(self, other: 'Style') -> 'Style'
    def get_types(self) -> list[Type]
    def get_by(self, condition: Callable[[Type], bool], default=None) -> Type | None
    def get(self, name: str, default=None) -> Type | None
    def get_all_by(self, condition: Callable) -> list[Type]
    @property
    def n_types(self) -> int
```

**Parameters:**
- `name`: Style identifier
- `parms`: Global style parameters
- `**kwparms`: Named style parameters

**Attributes:**
- `types`: TypeContainer holding the style's types

**Properties:**
- `n_types`: Number of types in this style

**Methods:**
- `merge(other)`: Merge another style into this one
- `get_types()`: Get all types as a list
- `get_by(condition, default=None)`: Find first type matching condition
- `get(name, default=None)`: Get type by name
- `get_all_by(condition)`: Get all types matching condition

---

#### `StyleContainer`

Container for managing collections of styles.

```python
class StyleContainer:
    def __init__(self)
    def add(self, style: Style)
    def __iter__(self) -> iterator
    def get(self, name: str, default=None) -> Style | None
```

**Methods:**
- `add(style)`: Add a style to the container
- `get(name, default=None)`: Retrieve style by name
- `__iter__()`: Iterate over styles

### Specific Type Classes

#### `AtomType`

Represents atom types with force field parameters.

```python
class AtomType(Type):
    def __init__(self, name: str, parms=[], **kwparms)
    def match(self, other: Entity) -> bool
    def apply(self, other: Atom)
```

**Methods:**
- `match(other)`: Check if atom entity has matching type
- `apply(other)`: Apply type parameters to an atom

---

#### `BondType`

Represents bond interactions between two atom types.

```python
class BondType(Type):
    def __init__(self, itype: AtomType, jtype: AtomType, name: str = "", parms: list = [], **kwparms)
    def match(self, other: Entity) -> bool
    @property
    def atomtypes(self) -> list[AtomType]
```

**Parameters:**
- `itype`: First atom type
- `jtype`: Second atom type
- `name`: Bond type name (auto-generated if not provided)

**Properties:**
- `atomtypes`: List of involved atom types [itype, jtype]

**Attributes:**
- `itype`: First atom type
- `jtype`: Second atom type

**Methods:**
- `match(other)`: Check if bond entity matches this type

---

#### `AngleType`

Represents three-body angle interactions.

```python
class AngleType(Type):
    def __init__(self, itype: AtomType, jtype: AtomType, ktype: AtomType, name: str = "", parms=[], **kwparms)
    def match(self, other: Entity) -> bool
    @property
    def atomtypes(self) -> list[AtomType]
```

**Parameters:**
- `itype`: First atom type
- `jtype`: Central atom type
- `ktype`: Third atom type
- `name`: Angle type name (auto-generated if not provided)

**Properties:**
- `atomtypes`: List of involved atom types [itype, jtype, ktype]

**Attributes:**
- `itype`: First atom type
- `jtype`: Central atom type
- `ktype`: Third atom type

**Methods:**
- `match(other)`: Check if angle entity matches this type

---

#### `DihedralType`

Represents four-body dihedral (torsion) interactions.

```python
class DihedralType(Type):
    def __init__(self, itype: AtomType, jtype: AtomType, ktype: AtomType, ltype: AtomType, name: str = "", parms=[], **kwparms)
    def match(self, other: Entity) -> bool
    @property
    def atomtypes(self) -> list[AtomType]
```

**Parameters:**
- `itype`: First atom type
- `jtype`: Second atom type
- `ktype`: Third atom type
- `ltype`: Fourth atom type
- `name`: Dihedral type name (auto-generated if not provided)

**Properties:**
- `atomtypes`: List of involved atom types [itype, jtype, ktype, ltype]

**Attributes:**
- `itype`, `jtype`, `ktype`, `ltype`: The four atom types

**Methods:**
- `match(other)`: Check if dihedral entity matches this type (supports reverse order)

---

#### `ImproperType`

Represents four-body improper (out-of-plane) interactions.

```python
class ImproperType(Type):
    def __init__(self, itype: AtomType, jtype: AtomType, ktype: AtomType, ltype: AtomType, name: str = "", parms=[], **kwparms)
    def match(self, other: Entity) -> bool
    @property
    def atomtypes(self) -> list[AtomType]
```

**Parameters:**
- `itype`: First atom type
- `jtype`: Second atom type
- `ktype`: Third atom type
- `ltype`: Fourth atom type
- `name`: Improper type name (auto-generated if not provided)

**Properties:**
- `atomtypes`: List of involved atom types [itype, jtype, ktype, ltype]

**Attributes:**
- `itype`, `jtype`, `ktype`, `ltype`: The four atom types

**Methods:**
- `match(other)`: Check if improper entity matches this type

---

#### `PairType`

Represents non-bonded pair interactions.

```python
class PairType(Type):
    def __init__(self, itype: AtomType, jtype: AtomType, name: str = "", parms=[], **kwparms)
    def match(self, other: Entity) -> bool
    @property
    def atomtypes(self) -> list[AtomType]
```

**Parameters:**
- `itype`: First atom type
- `jtype`: Second atom type
- `name`: Pair type name (auto-generated if not provided)

**Properties:**
- `atomtypes`: List of involved atom types [itype, jtype]

**Attributes:**
- `itype`: First atom type
- `jtype`: Second atom type

**Methods:**
- `match(other)`: Check if pair entity matches this type

### Style Classes

#### `AtomStyle`

Manages collections of atom types.

```python
class AtomStyle(Style):
    def __init__(self, name: str, parms, **kwparms)
    def def_type(self, name: str, class_=None, parms=[], **kwparms) -> AtomType
    def get_class(self, class_name: str) -> list[AtomType]
```

**Attributes:**
- `classes`: Dictionary mapping class names to sets of type names

**Methods:**
- `def_type(name, class_=None, parms=[], **kwparms)`: Define a new atom type
- `get_class(class_name)`: Get all atom types belonging to a class

---

#### `BondStyle`

Manages collections of bond types.

```python
class BondStyle(Style):
    def def_type(self, itype: AtomType, jtype: AtomType, name="", parms=[], **kwparms) -> BondType
```

**Methods:**
- `def_type(itype, jtype, name="", parms=[], **kwparms)`: Define a new bond type

---

#### `AngleStyle`

Manages collections of angle types.

```python
class AngleStyle(Style):
    def def_type(self, itype: AtomType, jtype: AtomType, ktype: AtomType, name="", parms=[], **kwparms) -> AngleType
```

**Methods:**
- `def_type(itype, jtype, ktype, name="", parms=[], **kwparms)`: Define a new angle type

---

#### `DihedralStyle`

Manages collections of dihedral types.

```python
class DihedralStyle(Style):
    def def_type(self, itype: AtomType, jtype: AtomType, ktype: AtomType, ltype: AtomType, name="", parms=[], **kwparms) -> DihedralType
```

**Methods:**
- `def_type(itype, jtype, ktype, ltype, name="", parms=[], **kwparms)`: Define a new dihedral type

---

#### `ImproperStyle`

Manages collections of improper types.

```python
class ImproperStyle(Style):
    def def_type(self, itype: AtomType, jtype: AtomType, ktype: AtomType, ltype: AtomType, name="", parms=[], **kwparms) -> ImproperType
```

**Methods:**
- `def_type(itype, jtype, ktype, ltype, name="", parms=[], **kwparms)`: Define a new improper type

---

#### `PairStyle`

Manages collections of pair types.

```python
class PairStyle(Style):
    def def_type(self, itype: AtomType, jtype: AtomType, name="", parms=[], **kwparms) -> PairType
```

**Methods:**
- `def_type(itype, jtype, name="", parms=[], **kwparms)`: Define a new pair type

### Main Container Class

#### `ForceField`

Main container class that organizes all styles into a complete force field.

```python
class ForceField:
    def __init__(self, name: str = "", unit: str = "real")
    def __repr__(self) -> str
    def __str__(self) -> str
    def __contains__(self, name: str) -> bool
    def __getitem__(self, name: str) -> Style
    def __len__(self) -> int
    def merge(self, other: 'ForceField') -> 'ForceField'
    def merge_(self, other: 'ForceField') -> 'ForceField'
    @classmethod
    def from_forcefields(cls, name: str = "", *forcefields: 'ForceField') -> 'ForceField'
```

**Parameters:**
- `name`: Force field identifier
- `unit`: Unit system (default: "real")

**Attributes:**
- `name`: Force field name
- `unit`: Unit system
- `atomstyles`: List of atom styles
- `bondstyles`: List of bond styles
- `anglestyles`: List of angle styles
- `dihedralstyles`: List of dihedral styles
- `improperstyles`: List of improper styles
- `pairstyles`: List of pair styles

**Properties (counts):**
- `n_atomstyles`, `n_bondstyles`, `n_anglestyles`, `n_dihedralstyles`, `n_improperstyles`, `n_pairstyles`: Number of each style type
- `n_atomtypes`, `n_bondtypes`, `n_angletypes`, `n_dihedraltypes`, `n_impropertypes`, `n_pairtypes`: Number of each type across all styles

**Style Definition Methods:**
```python
def def_atomstyle(self, name: str, parms=[], **data) -> AtomStyle
def def_bondstyle(self, style: str, parms=[], **data) -> BondStyle
def def_anglestyle(self, style: str, parms=[], **data) -> AngleStyle
def def_dihedralstyle(self, style: str, parms=[], **data) -> DihedralStyle
def def_improperstyle(self, style: str, parms=[], **data) -> ImproperStyle
def def_pairstyle(self, style: str, parms=[], **data) -> PairStyle
```

**Style Retrieval Methods:**
```python
def get_atomstyle(self, name: str) -> AtomStyle | None
def get_bondstyle(self, name: str) -> BondStyle | None
def get_anglestyle(self, name: str) -> AngleStyle | None
def get_dihedralstyle(self, name: str) -> DihedralStyle | None
def get_improperstyle(self, name: str) -> ImproperStyle | None
def get_pairstyle(self, name: str) -> PairStyle | None
```

**Type Retrieval Methods:**
```python
def get_atomtypes(self) -> list[AtomType]
def get_bondtypes(self) -> list[BondType]
def get_angletypes(self) -> list[AngleType]
def get_dihedraltypes(self) -> list[DihedralType]
def get_impropertypes(self) -> list[ImproperType]
def get_pairtypes(self) -> list[PairType]
```

**Container Methods:**
- `__contains__(name)`: Check if a style exists by name
- `__getitem__(name)`: Get a style by name (raises KeyError if not found)
- `__len__()`: Total number of styles
- `merge(other)`: Merge another force field into this one (returns self)
- `merge_(other)`: Alias for merge()

**Class Methods:**
- `from_forcefields(name, *forcefields)`: Create a new force field by merging multiple existing ones

## Usage Examples

### Basic Type Creation
```python
from molpy.core.forcefield import AtomType, BondType

# Create atom types
carbon = AtomType("C", mass=12.01, epsilon=0.07, sigma=3.4)
hydrogen = AtomType("H", mass=1.008, epsilon=0.03, sigma=2.5)

# Create bond type
ch_bond = BondType(carbon, hydrogen, k=340.0, r0=1.09)
```

### Style Management
```python
from molpy.core.forcefield import AtomStyle

# Create style and define types
lj_style = AtomStyle("lennard_jones")
carbon = lj_style.def_type("C", mass=12.01, epsilon=0.07, sigma=3.4)
hydrogen = lj_style.def_type("H", mass=1.008, epsilon=0.03, sigma=2.5)

print(f"Style has {lj_style.n_types} types")
```

### Complete Force Field
```python
from molpy.core.forcefield import ForceField

# Create force field
ff = ForceField("my_ff", unit="real")

# Define styles and types
atom_style = ff.def_atomstyle("lj/cut")
carbon = atom_style.def_type("C", mass=12.01, epsilon=0.07, sigma=3.4)

bond_style = ff.def_bondstyle("harmonic") 
ch_bond = bond_style.def_type(carbon, hydrogen, k=340.0, r0=1.09)

print(f"Force field: {ff}")
print(f"Contains {ff.n_atomtypes} atom types and {ff.n_bondtypes} bond types")
```

## Error Handling

The module includes error handling for common scenarios:

- **KeyError**: Raised when accessing non-existent styles by name
- **NotImplementedError**: Raised when calling `match()` on base `Type` class
- **TypeError**: Raised for invalid type operations (e.g., adding non-hashable styles)
- **AttributeError**: Handled gracefully in match methods for malformed entities

Always check for None returns when using `get()` methods, or use the `in` operator to check for existence before accessing styles or types.
