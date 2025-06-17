# Frame Module Tutorial

The Frame module provides high-performance tabular data storage and manipulation based on xarray.Dataset, designed for efficient molecular data analysis and bulk operations.

## Table of Contents

1. [Overview](#overview)
2. [Basic Frame Creation](#basic-frame-creation)
3. [Dataset Architecture](#dataset-architecture)
4. [Data Access and Manipulation](#data-access-and-manipulation)
5. [Frame Operations](#frame-operations)
6. [Advanced Features](#advanced-features)
7. [Performance Considerations](#performance-considerations)

## Overview

### What is Frame

Frame is MolPy's efficient container for molecular data, built on xarray.Dataset architecture:

```python
import molpy as mp
import numpy as np

# Create basic Frame
frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'O1', 'H1'],
        'element': ['C', 'C', 'O', 'H'],
        'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0], [2.5,0,0]],
        'charge': [0.0, 0.0, -0.4, 0.1]
    }
)

print("Frame created successfully")
print(f"Frame datasets: {list(frame._data.keys())}")
```

### Key Features

- **Dataset-Based Storage**: Each field is an xarray.Dataset
- **Mixed Data Types**: Different variables can have different dtypes
- **Efficient Operations**: Vectorized computations and filtering
- **Robust Concatenation**: Automatic dtype consistency checks
- **Integration**: Seamless conversion from Struct objects

## Basic Frame Creation

### From Dictionaries

```python
# Create Frame with mixed data types
frame = mp.Frame(
    atoms={
        'name': ['C1', 'H1', 'O1'],           # string data
        'element': ['C', 'H', 'O'],           # string data
        'xyz': [[0,0,0], [1,0,0], [0,1,0]],   # float coordinates
        'atomic_number': [6, 1, 8],           # integer data
        'charge': [0.0, 0.1, -0.2],           # float data
        'is_aromatic': [False, False, False]   # boolean data
    },
    bonds={
        'atom1_idx': [0, 0],
        'atom2_idx': [1, 2],
        'bond_type': ['single', 'single'],
        'length': [1.0, 1.4]
    }
)

print(f"Atoms dataset: {list(frame._data['atoms'].data_vars.keys())}")
print(f"Bonds dataset: {list(frame._data['bonds'].data_vars.keys())}")
```

### From Struct Objects

```python
# Create structure first
water = mp.AtomicStructure(name="water")
o = water.def_atom(name="O", element="O", xyz=[0.0, 0.0, 0.0])
h1 = water.def_atom(name="H", element="H", xyz=[0.757, 0.586, 0.0])
h2 = water.def_atom(name="H", element="H", xyz=[-0.757, 0.586, 0.0])

# Convert to Frame
frame = water.to_frame()
print(f"Converted frame datasets: {list(frame._data.keys())}")
```

### Empty Frames

```python
# Create empty frame
empty_frame = mp.Frame()
print(f"Empty frame datasets: {list(empty_frame._data.keys())}")

# Empty frame with specific structure
empty_atoms = mp.Frame(atoms={})
print("Empty atoms frame created")
```

## Dataset Architecture

### Understanding the Structure

```python
# Create sample frame
frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2'],
        'xyz': [[0,0,0], [1,0,0]],
        'mass': [12.01, 12.01]
    }
)

# Access the dataset
atoms_ds = frame._data['atoms']
print(f"Dataset type: {type(atoms_ds)}")
print(f"Dataset dimensions: {atoms_ds.dims}")
print(f"Dataset coordinates: {list(atoms_ds.coords.keys())}")
print(f"Dataset variables: {list(atoms_ds.data_vars.keys())}")
```

### Data Types and Shapes

```python
# Examine data types
atoms_ds = frame._data['atoms']

for var_name, var_data in atoms_ds.data_vars.items():
    print(f"{var_name}: dtype={var_data.dtype}, shape={var_data.shape}")

# Access specific variables
names = atoms_ds['name']
coordinates = atoms_ds['xyz']
masses = atoms_ds['mass']

print(f"Names: {names.values}")
print(f"Coordinates shape: {coordinates.shape}")
print(f"Masses: {masses.values}")
```

## Data Access and Manipulation

### Basic Access Patterns

```python
# Create test frame
frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'O1', 'H1'],
        'element': ['C', 'C', 'O', 'H'],
        'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0], [2.5,0,0]],
        'mass': [12.01, 12.01, 15.99, 1.008]
    }
)

atoms_ds = frame._data['atoms']

# Index-based access
first_atom_name = atoms_ds['name'][0].values
first_coord = atoms_ds['xyz'][0].values
print(f"First atom: {first_atom_name} at {first_coord}")

# Slice access
first_two_atoms = atoms_ds.isel(index=slice(0, 2))
print(f"First two atoms: {first_two_atoms['name'].values}")
```

### Filtering and Selection

```python
# Filter by condition
heavy_atoms = atoms_ds.where(atoms_ds['element'] != 'H', drop=True)
print(f"Heavy atoms: {heavy_atoms['name'].values}")

# Select by element
carbons = atoms_ds.where(atoms_ds['element'] == 'C', drop=True)
print(f"Carbon atoms: {carbons['name'].values}")

# Multiple conditions
large_atoms = atoms_ds.where(atoms_ds['mass'] > 10.0, drop=True)
print(f"Large atoms: {large_atoms['name'].values}")
```

### Vectorized Operations

```python
# Mathematical operations
coords = atoms_ds['xyz']
masses = atoms_ds['mass']

# Center of mass calculation
total_mass = masses.sum()
center_of_mass = (coords * masses.expand_dims('spatial')).sum(dim='index') / total_mass
print(f"Center of mass: {center_of_mass.values}")

# Distance calculations
first_coord = coords[0]
distances = np.sqrt(((coords - first_coord)**2).sum(dim='spatial'))
print(f"Distances from first atom: {distances.values}")
```

## Frame Operations

### Concatenation

```python
# Create two frames
frame1 = mp.Frame(
    atoms={
        'name': ['C1', 'C2'],
        'element': ['C', 'C'],
        'xyz': [[0,0,0], [1,0,0]]
    }
)

frame2 = mp.Frame(
    atoms={
        'name': ['O1', 'H1'],
        'element': ['O', 'H'],
        'xyz': [[2,0,0], [3,0,0]]
    }
)

# Concatenate frames
combined = mp.Frame.concat([frame1, frame2])
print(f"Combined frame atoms: {combined._data['atoms']['name'].values}")
```

### Dtype Consistency

```python
# Frames with different dtypes (will raise error)
try:
    frame_int = mp.Frame(atoms={'value': [1, 2, 3]})  # int
    frame_float = mp.Frame(atoms={'value': [1.5, 2.5, 3.5]})  # float
    
    # This will fail due to dtype mismatch
    combined = mp.Frame.concat([frame_int, frame_float])
except ValueError as e:
    print(f"Dtype mismatch error: {e}")

# Correct approach - ensure consistent dtypes
frame_int = mp.Frame(atoms={'value': [1.0, 2.0, 3.0]})  # float
frame_float = mp.Frame(atoms={'value': [1.5, 2.5, 3.5]})  # float
combined = mp.Frame.concat([frame_int, frame_float])
print("Concatenation successful with consistent dtypes")
```

### Frame Arithmetic

```python
# Create frame for arithmetic operations
frame = mp.Frame(
    atoms={
        'name': ['A1', 'A2'],
        'xyz': [[0,0,0], [1,1,1]],
        'charge': [0.5, -0.5]
    }
)

# Replicate frame
doubled = frame + frame  # Uses __add__ method
print(f"Doubled frame atoms: {len(doubled._data['atoms']['name'])}")

# Multiply frame
tripled = frame * 3  # Uses __mul__ method
print(f"Tripled frame atoms: {len(tripled._data['atoms']['name'])}")
```

## Advanced Features

### Coordinate Systems

```python
# Create frame with spatial coordinates
frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'C3'],
        'xyz': [[0,0,0], [1,0,0], [0,1,0]]
    }
)

atoms_ds = frame._data['atoms']
coords = atoms_ds['xyz']

# The 'spatial' dimension represents x, y, z
print(f"Coordinate dimensions: {coords.dims}")
print(f"Coordinate shape: {coords.shape}")

# Access x, y, z components
x_coords = coords.isel(spatial=0)
y_coords = coords.isel(spatial=1)
z_coords = coords.isel(spatial=2)

print(f"X coordinates: {x_coords.values}")
print(f"Y coordinates: {y_coords.values}")
print(f"Z coordinates: {z_coords.values}")
```

### Metadata Handling

```python
# Frame with metadata
frame = mp.Frame(
    atoms={'name': ['A1', 'A2'], 'xyz': [[0,0,0], [1,0,0]]},
    timestep=100,
    temperature=300.0
)

# Access metadata
print(f"Timestep: {frame.timestep}")
print(f"Temperature: {frame.temperature}")
print(f"Metadata: {frame._meta}")

# Add custom metadata
frame._meta['custom_field'] = "custom_value"
print(f"Custom field: {frame._meta['custom_field']}")
```

### Conversion to Dictionary

```python
# Convert frame to dictionary
frame_dict = frame.to_dict()
print(f"Dictionary keys: {list(frame_dict.keys())}")
print(f"Atoms data: {frame_dict['atoms']}")
```

## Performance Considerations

### Large Datasets

```python
# Create large frame efficiently
n_atoms = 10000
large_frame = mp.Frame(
    atoms={
        'name': [f'A{i}' for i in range(n_atoms)],
        'element': ['C'] * n_atoms,
        'xyz': np.random.random((n_atoms, 3)),
        'charge': np.random.random(n_atoms) - 0.5
    }
)

print(f"Large frame created with {n_atoms} atoms")

# Efficient operations on large frames
atoms_ds = large_frame._data['atoms']
coords = atoms_ds['xyz']

# Vectorized distance calculation
center = coords.mean(dim='index')
distances = np.sqrt(((coords - center)**2).sum(dim='spatial'))
print(f"Computed {len(distances)} distances efficiently")
```

## Best Practices

### Data Type Consistency

```python
# Ensure consistent data types for concatenation
frame1 = mp.Frame(atoms={'mass': [12.01, 1.008]})  # float64
frame2 = mp.Frame(atoms={'mass': [15.999, 14.007]})  # float64

# This works - same dtypes
combined = mp.Frame.concat([frame1, frame2])
```

### Efficient Operations

```python
# Use vectorized operations
atoms_ds = frame._data['atoms']

# Good: Vectorized
total_mass = atoms_ds['mass'].sum()
center_coords = atoms_ds['xyz'].mean(dim='index')

# Avoid: Loops when possible
```

### Memory Efficiency

```python
# For large frames, clean up intermediate results
large_frame = create_large_frame()
processed_data = large_frame._data['atoms']['xyz'].mean(dim='index')

# Clear reference to large frame when done
large_frame = None
```

## Summary

The Frame module provides efficient molecular data processing:

- **Dataset Architecture**: xarray.Dataset for each field with mixed data types
- **Vectorized Operations**: Efficient computations on large datasets
- **Robust Concatenation**: Automatic dtype consistency checking
- **Integration**: Seamless conversion from Struct objects
- **Performance**: Optimized for large-scale molecular data analysis

For object-oriented molecular building, see the [Struct Tutorial](struct.md). For detailed API documentation, see the [API Reference](../api/index.md).
