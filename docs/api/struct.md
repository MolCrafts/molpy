# Structure API Reference

The `molpy.core.struct` module provides classes for representing and manipulating molecular structures.

## Overview

This module contains classes for:
- **Basic entities**: `Entity` (base class with dict-like behavior)
- **Atomic components**: `Atom`, `Bond`, `Angle`, `Dihedral`
- **Structure containers**: `AtomicStructure`, `Struct`, `MolecularStructure`

## Complete API Documentation

::: molpy.core.struct
    options:
      show_source: true
      show_root_heading: false
      heading_level: 2

## Usage Examples

### Creating Basic Structures

```python
import molpy as mp

# Create individual atoms
atom1 = mp.Atom(symbol='C', x=0.0, y=0.0, z=0.0)
atom2 = mp.Atom(symbol='H', x=1.0, y=0.0, z=0.0)

# Create atomic structure
structure = mp.AtomicStructure()
structure.add_atom(atom1)
structure.add_atom(atom2)

# Add bond
bond = mp.Bond(atom1.id, atom2.id, order=1)
structure.add_bond(bond)
```

### Working with Higher-Level Structures

```python
# Create Struct with metadata
struct = mp.Struct("methane")
struct.add_atomic_structure(structure)

# Convert to molecular structure
mol_struct = mp.MolecularStructure(struct)
```

For complete tutorials, see [Structure Tutorial](../tutorials/struct.md).
