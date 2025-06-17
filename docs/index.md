# MolPy Documentation

Welcome to MolPy - a comprehensive Python framework for molecular modeling and analysis.

## Overview

MolPy provides two complementary paradigms for molecular data handling:

- **Struct Module**: Object-oriented molecular building and manipulation
- **Frame Module**: High-performance tabular data processing

## Quick Start

### Installation

```bash
pip install molpy
```

### Basic Usage

```python
import molpy as mp
import numpy as np

# Create a molecular structure
water = mp.AtomicStructure(name="water")
o = water.def_atom(name="O", element="O", xyz=[0.0, 0.0, 0.0])
h1 = water.def_atom(name="H", element="H", xyz=[0.757, 0.586, 0.0])
h2 = water.def_atom(name="H", element="H", xyz=[-0.757, 0.586, 0.0])

# Add bonds
water.def_bond(o, h1)
water.def_bond(o, h2)

# Convert to Frame for analysis
frame = water.to_frame()
print(f"Molecular data: {frame._data['atoms']['name'].values}")
```

## Documentation Structure

### Tutorials
- [Struct Module Tutorial](tutorials/struct.md) - Object-oriented molecular structures
- [Frame Module Tutorial](tutorials/frame.md) - High-performance data processing
- [Force Field Guide](tutorials/forcefield_guide.md) - Force field management

### API Reference
- [Core API](api/core.md) - Core classes and functions
- [Force Field API](api/forcefield_reference.md) - Force field utilities

### Examples
- [Basic Examples](examples/) - Practical usage examples

## Key Features

### Struct Module
- **Flexible Entity System**: Dictionary-like property storage
- **Spatial Operations**: 3D transformations and geometry calculations
- **Hierarchical Design**: From atoms to complex molecular systems
- **Seamless Integration**: Convert to Frame for data analysis

### Frame Module
- **Dataset-Based Storage**: xarray.Dataset for each data field
- **Mixed Data Types**: Support for different dtypes in the same frame
- **Efficient Operations**: Vectorized computations and filtering
- **Robust Concatenation**: Automatic dtype consistency checks

## Getting Help

- Check the [Struct Tutorial](tutorials/struct.md) for structure manipulation
- Read the [Frame Tutorial](tutorials/frame.md) for data processing
- Browse the [API Reference](api/index.md) for detailed documentation

## Contributing

MolPy is an open-source project. Contributions are welcome!

