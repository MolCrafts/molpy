# MolPy Core Data Structures Guide

A comprehensive guide to understanding and using MolPy's core data structures for molecular modeling and analysis.

## Overview

MolPy provides two primary data structure paradigms:

- **Struct Module**: Object-oriented molecular building and manipulation
- **Frame Module**: High-performance tabular data operations

This guide will help you understand when and how to use each approach.

## Quick Start

### Basic Molecular Building with Struct

```python
import molpy as mp
import numpy as np

# Create atoms
carbon = mp.Atom(name="C1", element="C", xyz=[0.0, 0.0, 0.0])
hydrogen = mp.Atom(name="H1", element="H", xyz=[1.1, 0.0, 0.0])

# Create molecular structure
molecule = mp.AtomicStructure(name="CH_fragment")
molecule.add_atom(carbon)
molecule.add_atom(hydrogen)

# Add chemical bond
bond = mp.Bond(carbon, hydrogen, bond_type="single")
molecule.add_bond(bond)

print(f"Created {molecule['name']} with {len(molecule.atoms)} atoms")
```

### High-Performance Data Operations with Frame

```python
# Create Frame for bulk operations
frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'O1', 'H1'],
        'element': ['C', 'C', 'O', 'H'],
        'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0], [2.5,0,0]],
        'charge': [0.0, 0.0, -0.4, 0.1]
    }
)

# Efficient filtering and analysis
heavy_atoms = frame['atoms'].where(
    frame['atoms'].coords['element'] != 'H',
    drop=True
)
print(f"Heavy atoms: {len(heavy_atoms.atom)}")
```

## Core Concepts

### Entity System
All MolPy objects inherit from `Entity`, providing:
- Dictionary-like property storage
- Deep cloning with modifications
- Flexible attribute access
- Object identity-based operations

### Spatial Operations
Objects with coordinates inherit from `SpatialMixin`:
- Translation, rotation, reflection
- Distance calculations
- Geometric transformations
- Coordinate system management

### Hierarchical Design
MolPy supports multiple levels of organization:
- **Atoms**: Basic building blocks
- **Bonds/Angles/Dihedrals**: Connectivity and geometry
- **Structures**: Complete molecular systems
- **Frames**: Efficient data containers

## When to Use What

### Use Struct When:
- Building molecules step by step
- Need detailed object relationships
- Working with small to medium systems
- Require fine-grained control
- Developing molecular builders

### Use Frame When:
- Analyzing large datasets
- Performing bulk operations
- Need high-performance computations
- Working with trajectories
- Statistical analysis required

## Architecture Overview

```
MolPy Core Architecture

Entity (Base Class)
├── SpatialMixin (Coordinates + Operations)
│   ├── Atom (Individual atoms)
│   └── AtomicStructure (Molecular systems)
├── Bond (Atom connectivity)
├── Angle (3-atom geometry)
└── Dihedral (4-atom torsions)

Frame (Tabular Data)
├── atoms (Atomic properties)
├── bonds (Connectivity data)
├── trajectory (Time series)
└── custom (User-defined fields)
```

## Integration Patterns

### Structure ↔ Frame Conversion
```python
# Struct to Frame (for analysis)
structure = mp.AtomicStructure(name="molecule")
# ... build structure ...
frame = structure.to_frame()

# Frame to Struct (for manipulation)
new_structure = frame.to_structure()
```

### Hybrid Workflows
```python
# Build with Struct, analyze with Frame
molecule = mp.AtomicStructure(name="drug")
# ... detailed molecular building ...

# Convert for bulk analysis
frame = molecule.to_frame()
coords = frame['atoms'].coords['xyz'].values
center_of_mass = np.average(coords, weights=masses, axis=0)

# Apply results back to structure
molecule.move(-center_of_mass)  # Center the molecule
```

## Next Steps

1. **For Beginners**: Start with [Struct Tutorial](tutorials/struct_complete_tutorial.md)
2. **For Data Analysis**: Learn [Frame Tutorial](tutorials/frame_complete_tutorial.md)  
3. **API Reference**: See [Complete API](api/api_reference.md)
4. **Examples**: Browse [Example Collection](examples/examples_collection.md)

## Performance Guidelines

### Memory Efficiency
- Use Frame for large systems (>10,000 atoms)
- Struct for detailed manipulation (<1,000 atoms)
- Convert between formats as needed

### Computational Efficiency
- Vectorize operations using NumPy
- Use Frame for statistical analysis
- Leverage xarray for multi-dimensional data

### Best Practices
- Consistent naming conventions
- Proper error handling
- Memory cleanup for large systems
- Type hints for better code clarity

This guide provides the foundation for working with MolPy's core data structures. Choose the approach that best fits your specific use case and performance requirements.
