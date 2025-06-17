# Struct Module Tutorial

The Struct module provides object-oriented molecular structures for building and manipulating molecular systems. This tutorial covers all aspects from basic atom creation to complex molecular systems.

## Table of Contents

1. [Basic Concepts](#basic-concepts)
2. [Atoms](#atoms)
3. [Bonds and Connectivity](#bonds-and-connectivity)
4. [Molecular Structures](#molecular-structures)
5. [Spatial Operations](#spatial-operations)
6. [Structure-Frame Conversion](#structure-frame-conversion)
7. [Advanced Features](#advanced-features)

## Basic Concepts

### Entity System

All MolPy objects inherit from the `Entity` class, providing flexible dictionary-like behavior:

```python
import molpy as mp
import numpy as np

# Create basic entity
entity = mp.Entity(name="example", type="test")

# Dictionary-style access
print(entity["name"])  # "example"
entity["new_prop"] = 42

# Clone with modifications
modified = entity.clone(name="modified", value=100)
print(modified["name"])  # "modified"
print(modified["value"])  # 100

# Shortcut syntax
another = entity(name="another", extra="data")
```

**Key Features:**
- Flexible attribute storage
- Deep copy with modifications
- Dictionary-style access
- Object identity-based hashing

## Atoms

### Creating Atoms

```python
# Basic atom creation
carbon = mp.Atom(name="C1", element="C", xyz=[0, 0, 0])
hydrogen = mp.Atom(name="H1", element="H", xyz=[1.1, 0, 0])

# With additional properties
oxygen = mp.Atom(
    name="O1", 
    element="O", 
    xyz=[1.4, 0, 0],
    charge=-0.4,
    mass=15.999,
    atom_type="O.3"
)

print(f"Carbon: {carbon}")
print(f"Oxygen charge: {oxygen['charge']}")
```

### Atom Properties

```python
atom = mp.Atom(name="N1", element="N", xyz=[0, 0, 0])

# Standard properties
atom["mass"] = 14.007
atom["charge"] = -0.3
atom["residue"] = "ALA"

# Custom properties
atom["custom_field"] = "custom_value"
atom["is_aromatic"] = True

# Check property existence
if "charge" in atom:
    print(f"Atom has charge: {atom['charge']}")
```

### Spatial Operations

```python
# Create atoms for spatial operations
atom1 = mp.Atom(name="C1", xyz=[0, 0, 0])
atom2 = mp.Atom(name="C2", xyz=[1.5, 0, 0])

# Distance calculation
distance = atom1.distance_to(atom2)
print(f"Distance: {distance:.2f} Å")

# Transformations
atom1.move([1, 0, 0])  # Translation
atom1.rotate(np.pi/4, [0, 0, 1])  # Rotation around z-axis
atom1.reflect([1, 0, 0])  # Reflection across x=0 plane

print(f"Transformed position: {atom1.xyz}")

# Clone atoms
new_atom = atom1.clone(name="C3", xyz=[0, 1.5, 0])
```

## Bonds and Connectivity

### Creating Bonds

```python
# Create atoms
c1 = mp.Atom(name="C1", element="C", xyz=[0, 0, 0])
c2 = mp.Atom(name="C2", element="C", xyz=[1.5, 0, 0])

# Create bond
bond = mp.Bond(c1, c2, bond_type="single", bond_order=1)

print(f"Bond atoms: {[atom['name'] for atom in bond.atoms]}")
print(f"Bond length: {bond.length:.3f} Å")
print(f"Bond type: {bond['bond_type']}")
```

### Bond Properties

```python
# Create bond with properties
bond = mp.Bond(
    c1, c2,
    bond_type="aromatic",
    bond_order=1.5,
    force_constant=500.0,
    is_rotatable=False
)

# Access bond properties
print(f"Current length: {bond.length:.3f} Å")
print(f"Force constant: {bond['force_constant']}")
print(f"Rotatable: {bond['is_rotatable']}")
```

### Angles and Dihedrals

```python
# Create atoms for angle
c1 = mp.Atom(name="C1", xyz=[0, 0, 0])
o = mp.Atom(name="O", xyz=[1, 0, 0])
c2 = mp.Atom(name="C2", xyz=[2, 1, 0])

# Create angle
angle = mp.Angle(c1, o, c2, angle_type="ether")
print(f"Angle value: {np.degrees(angle.value):.1f}°")

# Create dihedral
c3 = mp.Atom(name="C3", xyz=[3, 1, 1])
c4 = mp.Atom(name="C4", xyz=[4, 1, 1])
dihedral = mp.Dihedral(c1, c2, c3, c4)
print(f"Dihedral angle: {np.degrees(dihedral.value):.1f}°")
```

## Molecular Structures

### Creating AtomicStructure

```python
# Create empty structure
molecule = mp.AtomicStructure(name="water")

# Add atoms using def_atom
o = molecule.def_atom(name="O", element="O", xyz=[0, 0, 0])
h1 = molecule.def_atom(name="H1", element="H", xyz=[0.757, 0.586, 0])
h2 = molecule.def_atom(name="H2", element="H", xyz=[-0.757, 0.586, 0])

print(f"Molecule: {molecule['name']}")
print(f"Atoms: {len(molecule.atoms)}")
print(f"Atom names: {[atom['name'] for atom in molecule.atoms]}")
```

### Adding Bonds and Topology

```python
# Add bonds
bond1 = molecule.def_bond(o, h1, bond_type="covalent")
bond2 = molecule.def_bond(o, h2, bond_type="covalent")

# Add angle
angle = mp.Angle(h1, o, h2, angle_type="bent")
molecule.add_angle(angle)

print(f"Bonds: {len(molecule.bonds)}")
print(f"Angles: {len(molecule.angles)}")
print(f"H-O-H angle: {np.degrees(angle.value):.1f}°")
```

### Complex Molecular Systems

```python
# Create benzene ring
benzene = mp.AtomicStructure(name="benzene")

# Add carbon atoms in hexagonal arrangement
import math
radius = 1.4
carbon_atoms = []

for i in range(6):
    angle = i * math.pi / 3
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    z = 0.0
    
    atom = benzene.def_atom(
        name=f"C{i+1}",
        element="C",
        xyz=[x, y, z]
    )
    carbon_atoms.append(atom)

# Add ring bonds
for i in range(6):
    next_i = (i + 1) % 6
    benzene.def_bond(
        carbon_atoms[i],
        carbon_atoms[next_i],
        bond_type="aromatic",
        bond_order=1.5
    )

print(f"Benzene: {len(benzene.atoms)} atoms, {len(benzene.bonds)} bonds")
```

## Spatial Operations

### Structure-Level Transformations

```python
# Create a simple molecule
mol = mp.AtomicStructure(name="methane")
c = mol.def_atom(name="C", element="C", xyz=[0, 0, 0])
h1 = mol.def_atom(name="H1", element="H", xyz=[1.09, 0, 0])
h2 = mol.def_atom(name="H2", element="H", xyz=[-0.363, 1.027, 0])
h3 = mol.def_atom(name="H3", element="H", xyz=[-0.363, -0.513, 0.889])
h4 = mol.def_atom(name="H4", element="H", xyz=[-0.363, -0.513, -0.889])

# Transform entire structure
mol.move([0, 0, 5])  # Move up
mol.rotate(math.pi/4, [0, 0, 1])  # Rotate 45°

# Access all coordinates
coords = mol.xyz
print(f"Coordinates shape: {coords.shape}")

# Set all coordinates at once
new_coords = coords + np.array([1, 1, 1])
mol.xyz = new_coords
```

### Geometric Analysis

```python
# Calculate molecular properties
center_of_mass = mol.center_of_mass()
print(f"Center of mass: {center_of_mass}")

# Get molecular formula
formula = mol.molecular_formula()
print(f"Molecular formula: {formula}")

# Distance matrix
distances = mol.distance_matrix()
print(f"Distance matrix shape: {distances.shape}")
```

## Structure-Frame Conversion

### Converting to Frame

```python
# Create a molecular structure
water = mp.AtomicStructure(name="water_molecule")
o = water.def_atom(name="O", element="O", xyz=[0.0, 0.0, 0.0])
h1 = water.def_atom(name="H", element="H", xyz=[0.757, 0.586, 0.0])
h2 = water.def_atom(name="H", element="H", xyz=[-0.757, 0.586, 0.0])

# Convert to Frame
frame = water.to_frame()
print(f"Frame datasets: {list(frame._data.keys())}")

# Access atomic data
atoms_ds = frame._data['atoms']
print(f"Variables: {list(atoms_ds.data_vars.keys())}")
print(f"Coordinates shape: {atoms_ds['xyz'].shape}")
print(f"Names: {atoms_ds['name'].values}")

# Check metadata
print(f"Structure name: {frame._meta.get('structure_name')}")
```

### Benefits of Frame Conversion

```python
# Efficient analysis using Frame
frame = water.to_frame()
atoms_ds = frame._data['atoms']

# Vectorized operations
center = atoms_ds['xyz'].mean(dim='index')
print(f"Geometric center: {center.values}")

# Filtering
heavy_atoms = atoms_ds.where(atoms_ds['element'] != 'H', drop=True)
print(f"Heavy atoms: {len(heavy_atoms.index)}")

# Statistical analysis
if 'mass' in atoms_ds.data_vars:
    total_mass = atoms_ds['mass'].sum()
    print(f"Total mass: {total_mass.values} amu")
```

## Advanced Features

### Hierarchical Structures

```python
# Create protein system
protein = mp.AtomicStructure(name="mini_protein")

# Create residues
residues = []
for i in range(3):
    residue = mp.AtomicStructure(name=f"RES{i+1}")
    
    # Add backbone atoms
    n = residue.def_atom(name="N", element="N", xyz=[i*3.8, 0, 0])
    ca = residue.def_atom(name="CA", element="C", xyz=[i*3.8+1.5, 0, 0])
    c = residue.def_atom(name="C", element="C", xyz=[i*3.8+3, 0, 0])
    
    # Add bonds
    residue.def_bond(n, ca)
    residue.def_bond(ca, c)
    
    residues.append(residue)
    protein.add_struct(residue)

print(f"Protein with {len(residues)} residues")
print(f"Total atoms: {len(protein.atoms)}")
print(f"Hierarchical children: {len(protein.children)}")
```

### Structure Concatenation

```python
# Create multiple structures
mol1 = mp.AtomicStructure(name="mol1")
mol1.def_atom(name="C1", element="C", xyz=[0, 0, 0])

mol2 = mp.AtomicStructure(name="mol2")
mol2.def_atom(name="C2", element="C", xyz=[5, 0, 0])

# Concatenate structures
combined = mp.AtomicStructure.concat("combined", [mol1, mol2])
print(f"Combined structure: {len(combined.atoms)} atoms")
```

## Best Practices

### Naming Conventions

```python
# Good naming practices
atom = mp.Atom(name="CA_1", element="C")  # Clear, consistent names
bond = mp.Bond(atom1, atom2, bond_type="single")  # Standard parameters
```

### Memory Management

```python
# For large structures, clean up when done
large_structure = create_large_structure()
frame = large_structure.to_frame()

# Perform analysis on frame
results = analyze_frame(frame)

# Clean up
large_structure = None
```

### Error Handling

```python
try:
    atom = mp.Atom(name="C1", element="C", xyz=[0, 0, 0])
    # Validate before creating bonds
    if atom1 is not atom2:  # Avoid self-bonds
        bond = mp.Bond(atom1, atom2)
except ValueError as e:
    print(f"Invalid operation: {e}")

# Check properties safely
charge = atom.get("charge", 0.0)  # Default value if not present
```

## Summary

The Struct module provides a comprehensive framework for molecular structure representation:

- **Entity System**: Flexible property storage and manipulation
- **Spatial Operations**: Complete 3D geometric operations
- **Hierarchical Design**: From atoms to complex molecular systems
- **Performance**: Optimized for both small molecules and large systems
- **Integration**: Seamless conversion to Frame for data analysis

This tutorial covered the essential concepts and practical usage patterns. For detailed API documentation, see the [API Reference](../api/index.md).
