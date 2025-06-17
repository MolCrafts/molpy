# Core API Reference

This document provides detailed API reference for MolPy's core modules.

## Table of Contents

1. [Entity System](#entity-system)
2. [Spatial Classes](#spatial-classes)
3. [Atomic Components](#atomic-components)
4. [Molecular Structures](#molecular-structures)
5. [Frame System](#frame-system)
6. [Utility Functions](#utility-functions)

## Entity System

### Entity

Base class providing dictionary-like behavior for all MolPy objects.

```python
class Entity(UserDict):
```

#### Constructor

```python
def __init__(self, **kwargs)
```

**Parameters:**
- `**kwargs`: Initial attribute key-value pairs

**Example:**
```python
entity = mp.Entity(name="example", value=42, type="test")
```

#### Methods

##### `clone(self, **modify) -> Entity`

Create a deep copy with optional modifications.

**Parameters:**
- `**modify`: Key-value pairs to modify in the copy

**Returns:**
- `Entity`: New entity instance

**Example:**
```python
original = mp.Entity(name="test", value=1)
modified = original.clone(value=2, new_attr="added")
```

##### `__call__(self, **modify) -> Entity`

Shortcut for `clone` method.

##### `to_dict(self) -> dict`

Convert entity to standard dictionary.

**Returns:**
- `dict`: Dictionary representation

## Spatial Classes

### SpatialMixin

Abstract mixin providing spatial operations.

```python
class SpatialMixin(ABC):
```

#### Properties

##### `xyz: np.ndarray`

Get/set xyz coordinates as numpy array.

#### Methods

##### `distance_to(self, other: SpatialMixin) -> float`

Calculate distance to another spatial entity.

**Parameters:**
- `other`: Target spatial entity

**Returns:**
- `float`: Distance in Angstroms

##### `move(self, vector: ArrayLike) -> None`

Translate by given vector.

**Parameters:**
- `vector`: Translation vector [x, y, z]

##### `rotate(self, angle: float, axis: ArrayLike) -> None`

Rotate around axis by angle.

**Parameters:**
- `angle`: Rotation angle in radians
- `axis`: Rotation axis vector [x, y, z]

##### `reflect(self, normal: ArrayLike) -> None`

Reflect across plane defined by normal vector.

**Parameters:**
- `normal`: Plane normal vector [x, y, z]

## Atomic Components

### Atom

Represents a single atom with spatial and chemical properties.

```python
class Atom(Entity, SpatialMixin):
```

#### Constructor

```python
def __init__(self, name: str = "", element: str = "", xyz: ArrayLike = None, **props)
```

**Parameters:**
- `name`: Atom name/identifier
- `element`: Chemical element symbol
- `xyz`: Coordinates [x, y, z]
- `**props`: Additional properties

**Example:**
```python
atom = mp.Atom(
    name="CA",
    element="C", 
    xyz=[0.0, 0.0, 0.0],
    charge=0.1,
    mass=12.01
)
```

### Bond

Represents a chemical bond between two atoms.

```python
class Bond(Entity):
```

#### Constructor

```python
def __init__(self, atom1: Atom, atom2: Atom, **props)
```

**Parameters:**
- `atom1`, `atom2`: Connected atoms
- `**props`: Bond properties (bond_type, bond_order, etc.)

#### Properties

##### `atoms: tuple`

Get tuple of bonded atoms.

##### `length: float`

Calculate current bond length.

**Example:**
```python
bond = mp.Bond(atom1, atom2, bond_type="single", bond_order=1)
print(f"Bond length: {bond.length:.3f} Å")
```

### Angle

Represents a bond angle between three atoms.

```python
class Angle(Entity):
```

#### Constructor

```python
def __init__(self, atom1: Atom, vertex: Atom, atom2: Atom, **props)
```

**Parameters:**
- `atom1`: First atom
- `vertex`: Central atom
- `atom2`: Third atom
- `**props`: Angle properties

#### Properties

##### `value: float`

Get angle value in radians.

##### `atoms: tuple`

Get tuple of three atoms (atom1, vertex, atom2).

### Dihedral

Represents a dihedral angle between four atoms.

```python
class Dihedral(Entity):
```

#### Constructor

```python
def __init__(self, atom1: Atom, atom2: Atom, atom3: Atom, atom4: Atom, **props)
```

#### Properties

##### `value: float`

Get dihedral angle value in radians.

##### `atoms: tuple`

Get tuple of four atoms.

### Entities

Container class for managing collections of entities.

```python
class Entities(list):
```

#### Methods

##### `add(self, entity) -> entity`

Add entity to collection.

##### `remove(self, entity) -> None`

Remove entity from collection.

##### `get_by(self, condition: Callable) -> entity`

Get first entity matching condition.

**Parameters:**
- `condition`: Function returning boolean

**Example:**
```python
entities = mp.Entities()
entities.add(atom1)
carbon = entities.get_by(lambda x: x['element'] == 'C')
```

## Molecular Structures

### Struct

Base structure class providing entity functionality.

```python
class Struct(Entity):
```

#### Constructor

```python
def __init__(self, name: str = "", **props)
```

**Example:**
```python
struct = mp.Struct(name="molecule", type="organic")
```

### AtomicStructure

Complete atomic structure with atoms, bonds, and topology.

```python
class AtomicStructure(Struct, SpatialMixin, HierarchyMixin):
```

#### Constructor

```python
def __init__(self, name: str = "", **props)
```

#### Properties

##### `atoms: Entities`

Collection of atoms in structure.

##### `bonds: Entities`

Collection of bonds in structure.

##### `angles: Entities`

Collection of angles in structure.

##### `dihedrals: Entities`

Collection of dihedrals in structure.

#### Methods

##### `def_atom(self, **props) -> Atom`

Create and add atom with given properties.

**Parameters:**
- `**props`: Atom properties

**Returns:**
- `Atom`: Created atom

**Example:**
```python
atom = structure.def_atom(name="C1", element="C", xyz=[0, 0, 0])
```

##### `add_atom(self, atom: Atom) -> Atom`

Add existing atom to structure.

##### `def_bond(self, atom1: Atom, atom2: Atom, **kwargs) -> Bond`

Create and add bond between atoms.

**Parameters:**
- `atom1`, `atom2`: Atoms to connect
- `**kwargs`: Bond properties

**Returns:**
- `Bond`: Created bond

##### `add_bond(self, bond: Bond) -> Bond`

Add existing bond to structure.

##### `add_angle(self, angle: Angle) -> Angle`

Add angle to structure.

##### `add_dihedral(self, dihedral: Dihedral) -> Dihedral`

Add dihedral to structure.

##### `remove_atom(self, atom: Atom) -> None`

Remove atom from structure.

##### `remove_bond(self, bond: Bond) -> None`

Remove bond from structure.

##### `to_frame(self) -> Frame`

Convert structure to Frame object.

**Returns:**
- `Frame`: Frame containing atomic data

**Example:**
```python
structure = mp.AtomicStructure(name="water")
o = structure.def_atom(name="O", element="O", xyz=[0, 0, 0])
h1 = structure.def_atom(name="H", element="H", xyz=[1, 0, 0])

frame = structure.to_frame()
print(f"Frame datasets: {list(frame._data.keys())}")
```

##### `add_struct(self, struct: AtomicStructure) -> AtomicStructure`

Add another structure to this one.

**Parameters:**
- `struct`: Structure to add

**Returns:**
- `AtomicStructure`: Self for method chaining

#### Class Methods

##### `concat(cls, name: str, structs: Sequence[AtomicStructure]) -> AtomicStructure`

Concatenate multiple structures.

**Parameters:**
- `name`: Name for new structure
- `structs`: Sequence of structures to concatenate

**Returns:**
- `AtomicStructure`: New combined structure

### MolecularStructure

Complete molecular structure implementation.

```python
class MolecularStructure(AtomicStructure):
```

Inherits all functionality from `AtomicStructure` with enhanced molecular-specific features.

## Frame System

### Frame

High-performance tabular molecular data container.

```python
class Frame:
```

#### Constructor

```python
def __init__(self, data: Dict[str, Dict[str, Any]] = None, **meta)
```

**Parameters:**
- `data`: Dictionary mapping field names to data dictionaries
- `**meta`: Metadata properties

**Example:**
```python
frame = mp.Frame(
    atoms={
        'name': ['C1', 'H1'],
        'element': ['C', 'H'],
        'xyz': [[0, 0, 0], [1, 0, 0]]
    },
    timestep=100
)
```

#### Properties

##### `_data: Dict[str, xr.Dataset]`

Internal storage as xarray Datasets.

##### `_meta: Dict[str, Any]`

Metadata dictionary.

##### `timestep: Optional[float]`

Simulation timestep if applicable.

##### `box: Optional[Box]`

Simulation box if applicable.

#### Methods

##### `copy(self) -> Frame`

Create deep copy of frame.

##### `to_dict(self) -> Dict[str, Any]`

Convert frame to nested dictionary.

**Returns:**
- `Dict`: Nested dictionary representation

#### Class Methods

##### `concat(cls, frames: Sequence[Frame]) -> Frame`

Concatenate multiple frames with dtype consistency checking.

**Parameters:**
- `frames`: Sequence of frames to concatenate

**Returns:**
- `Frame`: New concatenated frame

**Raises:**
- `ValueError`: If variable dtypes don't match across frames

**Example:**
```python
frame1 = mp.Frame(atoms={'name': ['C1'], 'xyz': [[0, 0, 0]]})
frame2 = mp.Frame(atoms={'name': ['H1'], 'xyz': [[1, 0, 0]]})
combined = mp.Frame.concat([frame1, frame2])
```

#### Arithmetic Operations

##### `__add__(self, other: Frame) -> Frame`

Concatenate frames using + operator.

##### `__mul__(self, n: int) -> Frame`

Replicate frame n times using * operator.

## Utility Functions

### `_dict_to_dataset(data: Dict[str, Any]) -> xr.Dataset`

Convert dictionary of arrays to xarray Dataset.

**Parameters:**
- `data`: Dictionary mapping variable names to arrays

**Returns:**
- `xr.Dataset`: Dataset with proper dimensions and coordinates

**Features:**
- Handles empty dictionaries gracefully
- Automatically determines dimensions based on array shapes
- Supports 1D, 2D, and scalar data
- Creates proper index coordinates

**Example:**
```python
data = {
    'name': ['C1', 'H1'],
    'xyz': [[0, 0, 0], [1, 0, 0]],
    'charge': [0.0, 0.1]
}
dataset = mp._dict_to_dataset(data)
```

## Type Definitions

### ArrayLike

```python
ArrayLike = Union[np.ndarray, list, tuple]
```

Represents array-like objects accepted by spatial operations.

### Sequence

```python
from typing import Sequence
```

Used for sequence parameters in concatenation and similar operations.

## Error Handling

### Common Exceptions

- `ValueError`: Invalid parameter values or incompatible operations
- `TypeError`: Incorrect parameter types
- `RuntimeError`: Runtime errors during operations

### Best Practices

```python
# Check before operations
if atom1 is not atom2:  # Avoid self-bonds
    bond = mp.Bond(atom1, atom2)

# Handle missing properties
charge = atom.get('charge', 0.0)  # Default value

# Validate before concatenation
try:
    combined = mp.Frame.concat([frame1, frame2])
except ValueError as e:
    print(f"Concatenation failed: {e}")
```

## Performance Notes

### Optimization Tips

1. **Use Frame for bulk operations**: Convert structures to frames for large-scale analysis
2. **Vectorize operations**: Use xarray operations instead of loops
3. **Batch creation**: Create many atoms/bonds at once when possible
4. **Memory management**: Clean up large objects when done

### Memory Considerations

```python
# Good: Use frames for large datasets
large_frame = structure.to_frame()
result = large_frame._data['atoms']['xyz'].mean()

# Avoid: Iterating through individual atoms in large structures
# total = 0
# for atom in structure.atoms:  # Slow for large structures
#     total += atom['mass']
```

This API reference covers the core functionality of MolPy. For tutorials and examples, see the [Tutorials](../tutorials/) section.
