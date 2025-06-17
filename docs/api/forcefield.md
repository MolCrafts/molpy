# ForceField API Reference

The `molpy.core.forcefield` module provides classes for managing force field parameters and types.

## Overview

This module handles force field parameter management:
- **Parameter types**: `AtomType`, `BondType`, `AngleType`, `DihedralType`
- **Force field container**: `ForceField` class for parameter organization
- **Type matching**: Automatic parameter assignment based on structure
- **File I/O**: Loading and saving force field parameters

## Complete API Documentation

::: molpy.core.forcefield
    options:
      show_source: true
      show_root_heading: false
      heading_level: 2

## Key Components

### Parameter Types
Force field parameters are organized by interaction type:
- **AtomType**: Atomic parameters (mass, charge, VdW)
- **BondType**: Bond parameters (force constant, equilibrium length)
- **AngleType**: Angle parameters (force constant, equilibrium angle)
- **DihedralType**: Dihedral parameters (barrier height, periodicity)

### Usage Examples

```python
import molpy as mp

# Create force field
ff = mp.ForceField("AMBER")

# Define atom types
atom_type_c = mp.AtomType("CA", mass=12.01, charge=0.0)
atom_type_h = mp.AtomType("HA", mass=1.008, charge=0.0)

# Add to force field
ff.add_atom_type(atom_type_c)
ff.add_atom_type(atom_type_h)

# Define bond types
bond_type = mp.BondType("CA-HA", k=340.0, r0=1.09)
ff.add_bond_type(bond_type)
```

### Parameter Assignment
```python
# Match structure to force field
structure = mp.AtomicStructure()
# ... add atoms and bonds ...

# Assign parameters
ff.assign_parameters(structure)
```

For complete tutorials, see the forcefield documentation.
