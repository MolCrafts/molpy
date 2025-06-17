# ForceField Module User Guide

The `molpy.core.forcefield` module provides a comprehensive framework for managing molecular force fields in computational chemistry and molecular dynamics simulations. This module implements a hierarchical structure for organizing force field parameters, including atoms, bonds, angles, dihedrals, impropers, and pair interactions.

## Overview

A force field in molecular simulation defines the interactions between particles (atoms) in a molecular system. The `molpy.core.forcefield` module provides:

- **Type System**: Define specific interaction types (AtomType, BondType, etc.)
- **Style System**: Group related types under styles (AtomStyle, BondStyle, etc.)
- **ForceField Container**: Organize multiple styles into complete force fields
- **Matching System**: Match molecular entities to their corresponding force field types

## Core Classes

### Base Classes

#### `Type`
The base class for all force field types. Provides common functionality for parameter storage and type matching.

```python
from molpy.core.forcefield import Type

# Create a basic type
type_obj = Type("example_type", parms=[1.0, 2.0], epsilon=0.1, sigma=3.4)
print(type_obj.name)  # "example_type"
print(type_obj[0])    # 1.0 (positional parameter)
print(type_obj["epsilon"])  # 0.1 (named parameter)
```

#### `TypeContainer`
Manages collections of types with efficient lookup and filtering capabilities.

```python
from molpy.core.forcefield import TypeContainer, Type

container = TypeContainer()
type1 = Type("C", mass=12.01)
type2 = Type("H", mass=1.008)

container.add(type1)
container.add(type2)

# Retrieve by name
carbon = container.get("C")
print(carbon["mass"])  # 12.01

# Filter by condition
light_atoms = container.get_all_by(lambda t: t.get("mass", 0) < 2.0)
print(len(light_atoms))  # 1 (hydrogen)
```

#### `Style`
Groups related types under a common style name, providing organization and batch operations.

```python
from molpy.core.forcefield import Style, Type

style = Style("lennard_jones", global_param=1.0)
carbon_type = Type("C", epsilon=0.07, sigma=3.4)
style.types.add(carbon_type)

print(style.n_types)  # 1
print(style.get("C"))  # Returns carbon_type
```

### Specific Type Classes

#### `AtomType`
Represents atom types with their force field parameters.

```python
from molpy.core.forcefield import AtomType

# Create an atom type for carbon
carbon = AtomType("C", mass=12.01, epsilon=0.07, sigma=3.4)

# Match against an atom entity
atom_entity = {"type": "C", "id": 1}
if carbon.match(atom_entity):
    print("Carbon type matches the atom entity")
    
# Apply type parameters to an atom
carbon.apply(atom_entity)  # Adds mass, epsilon, sigma to atom_entity
```

#### `BondType`
Defines bond interactions between two atom types.

```python
from molpy.core.forcefield import AtomType, BondType

carbon = AtomType("C")
hydrogen = AtomType("H")

# Create a C-H bond type
ch_bond = BondType(carbon, hydrogen, k=340.0, r0=1.09)
print(ch_bond.name)  # "C-H"
print(ch_bond.atomtypes)  # [carbon, hydrogen]

# Custom bond name
custom_bond = BondType(carbon, hydrogen, name="alkyl_CH", k=350.0, r0=1.08)
print(custom_bond.name)  # "alkyl_CH"
```

#### `AngleType`
Defines three-body angle interactions.

```python
from molpy.core.forcefield import AtomType, AngleType

carbon = AtomType("C")
hydrogen = AtomType("H")

# Create an H-C-H angle type
hch_angle = AngleType(hydrogen, carbon, hydrogen, k=35.0, theta0=109.5)
print(hch_angle.name)  # "H-C-H"
```

#### `DihedralType` and `ImproperType`
Define four-body torsional and improper interactions.

```python
from molpy.core.forcefield import AtomType, DihedralType, ImproperType

c1 = AtomType("C1")
c2 = AtomType("C2")
c3 = AtomType("C3")
h = AtomType("H")

# Dihedral: C1-C2-C3-H
dihedral = DihedralType(c1, c2, c3, h, k=1.0, n=3, phi0=0.0)

# Improper: C1-C2-C3-H (out-of-plane)
improper = ImproperType(c1, c2, c3, h, k=2.0, phi0=0.0)
```

#### `PairType`
Defines non-bonded pair interactions.

```python
from molpy.core.forcefield import AtomType, PairType

carbon = AtomType("C")
oxygen = AtomType("O")

# Create a C-O pair interaction
co_pair = PairType(carbon, oxygen, epsilon=0.05, sigma=3.2)
```

### Style Classes

Each interaction type has a corresponding style class that manages collections of types:

#### `AtomStyle`
```python
from molpy.core.forcefield import AtomStyle

lj_style = AtomStyle("lennard_jones", cutoff=12.0)

# Define atom types within the style
carbon = lj_style.def_type("C", mass=12.01, epsilon=0.07, sigma=3.4)
hydrogen = lj_style.def_type("H", mass=1.008, epsilon=0.03, sigma=2.5)

# Organize types by classes
aromatic_carbon = lj_style.def_type("CA", class_="aromatic", mass=12.01, epsilon=0.07, sigma=3.4)
print(lj_style.get_class("aromatic"))  # ['CA']
```

#### `BondStyle`, `AngleStyle`, etc.
```python
from molpy.core.forcefield import BondStyle, AtomType

harmonic_bonds = BondStyle("harmonic")
carbon = AtomType("C")
hydrogen = AtomType("H")

# Define bond type within the style
ch_bond = harmonic_bonds.def_type(carbon, hydrogen, k=340.0, r0=1.09)
```

### ForceField Class

The main container that organizes all styles into a complete force field:

```python
from molpy.core.forcefield import ForceField

# Create a new force field
ff = ForceField("my_forcefield", unit="real")

# Define styles
atom_style = ff.def_atomstyle("lennard_jones")
bond_style = ff.def_bondstyle("harmonic")
angle_style = ff.def_anglestyle("harmonic")

# Define types within styles
carbon = atom_style.def_type("C", mass=12.01, epsilon=0.07, sigma=3.4)
hydrogen = atom_style.def_type("H", mass=1.008, epsilon=0.03, sigma=2.5)
ch_bond = bond_style.def_type(carbon, hydrogen, k=340.0, r0=1.09)

# Access force field information
print(f"Force field has {ff.n_atomstyles} atom styles")
print(f"Force field has {ff.n_atomtypes} atom types")
print(f"Force field has {ff.n_bondtypes} bond types")

# Check if styles exist
if "lennard_jones" in ff:
    print("Lennard-Jones style found")

# Get specific styles
lj_style = ff["lennard_jones"]
harmonic_style = ff["harmonic"]
```

## Advanced Usage

### Merging Force Fields

```python
# Create two separate force fields
ff1 = ForceField("base_ff")
ff2 = ForceField("additional_ff")

# Set up force fields...
ff1.def_atomstyle("lennard_jones")
ff2.def_bondstyle("harmonic")

# Merge ff2 into ff1
ff1.merge(ff2)
print(f"Merged force field has {len(ff1)} styles")

# Create a new force field from multiple existing ones
combined_ff = ForceField.from_forcefields("combined", ff1, ff2)
```

### Type Matching

The matching system allows you to determine which force field types apply to specific molecular entities:

```python
from molpy.core.forcefield import AtomType

# Create atom type
carbon_type = AtomType("C")

# Mock molecular entities (would typically come from molpy.core.struct)
atom_entity = {"type": "C", "id": 1, "coords": [0, 0, 0]}

if carbon_type.match(atom_entity):
    print("Carbon force field type matches this atom")
    carbon_type.apply(atom_entity)  # Apply force field parameters
```

### Custom Type Filtering

```python
# Find all atom types with epsilon > 0.05
high_epsilon_types = ff.get_atomstyle("lennard_jones").get_all_by(
    lambda t: t.get("epsilon", 0) > 0.05
)

# Find all bond types involving carbon
carbon_bonds = []
for bond_style in ff.bondstyles:
    carbon_bonds.extend(
        bond_style.get_all_by(
            lambda bt: "C" in [bt.itype.name, bt.jtype.name]
        )
    )
```

## Best Practices

1. **Organize by Style**: Group related interaction types under appropriate styles
2. **Use Descriptive Names**: Give meaningful names to types and styles
3. **Set Units Consistently**: Specify units when creating force fields
4. **Validate Parameters**: Check that all required parameters are set
5. **Document Custom Types**: Add comments or documentation for custom force field types

## Integration with Other Modules

The force field module is designed to work seamlessly with other molpy modules:

- **molpy.core.struct**: Apply force field types to molecular structures
- **molpy.io**: Read and write force field files
- **molpy.engine**: Use force fields in simulation engines

## Example: Complete Force Field Setup

```python
from molpy.core.forcefield import ForceField

# Create a simple alkane force field
alkane_ff = ForceField("alkane_forcefield", unit="real")

# Set up atom style
lj_style = alkane_ff.def_atomstyle("lj/cut/coul/long", cutoff=12.0)
carbon = lj_style.def_type("C3", mass=12.01, epsilon=0.066, sigma=3.5, charge=0.0)
hydrogen = lj_style.def_type("H", mass=1.008, epsilon=0.030, sigma=2.5, charge=0.0)

# Set up bond style
bond_style = alkane_ff.def_bondstyle("harmonic")
cc_bond = bond_style.def_type(carbon, carbon, k=268.0, r0=1.529)
ch_bond = bond_style.def_type(carbon, hydrogen, k=340.0, r0=1.09)

# Set up angle style
angle_style = alkane_ff.def_anglestyle("harmonic")
ccc_angle = angle_style.def_type(carbon, carbon, carbon, k=58.35, theta0=112.7)
cch_angle = angle_style.def_type(carbon, carbon, hydrogen, k=37.5, theta0=110.7)

# Set up dihedral style
dihedral_style = alkane_ff.def_dihedralstyle("opls")
cccc_dihedral = dihedral_style.def_type(carbon, carbon, carbon, carbon, 
                                      k1=1.3, k2=-0.05, k3=0.2, k4=0.0)

print(f"Alkane force field contains:")
print(f"  {alkane_ff.n_atomtypes} atom types")
print(f"  {alkane_ff.n_bondtypes} bond types") 
print(f"  {alkane_ff.n_angletypes} angle types")
print(f"  {alkane_ff.n_dihedraltypes} dihedral types")
```

This force field can then be used to parametrize alkane molecules and run molecular dynamics simulations.
