# API Reference

Complete API documentation for all MolPy modules, automatically generated from source code.

## Core Modules

### [Struct Module](struct.md)
Object-oriented molecular structures and building blocks.

### [Frame Module](frame.md)
High-performance tabular data processing with xarray Datasets.

### [Trajectory Module](trajectory.md)
Molecular dynamics trajectory handling.

### [ForceField Module](forcefield.md) 
Force field parameter management.

## Quick Reference

### Most Used Classes

| Class | Module | Description |
|-------|--------|-------------|
| `Entity` | struct | Base class with dictionary-like behavior |
| `Atom` | struct | Individual atom representation |
| `Bond` | struct | Chemical bond between atoms |
| `Angle` | struct | Three-atom angle representation |
| `Dihedral` | struct | Four-atom dihedral angle |
| `AtomicStructure` | struct | Collection of atoms and bonds |
| `Struct` | struct | High-level structure with metadata |
| `Frame` | frame | xarray Dataset-based data container |
| `Trajectory` | trajectory | Sequence of molecular frames |
| `ForceField` | forcefield | Force field parameters |

### Quick Links

- [Entity System](struct.md#molpy.core.struct.Entity) - Base functionality
- [Atomic Structures](struct.md#molpy.core.struct.AtomicStructure) - Structure creation
- [Data Processing](frame.md#molpy.core.frame.Frame) - Dataset-based operations
- [Trajectory Analysis](trajectory.md#molpy.core.trajectory.Trajectory) - MD data
- [Force Fields](forcefield.md#molpy.core.forcefield.ForceField) - Parameter management

## Navigation

For detailed examples and tutorials, see:
- [Struct Tutorial](../tutorials/struct.md)
- [Frame Tutorial](../tutorials/frame.md)

## Usage Examples

### Basic Structure Operations
```python
import molpy as mp

# Create atoms
atom1 = mp.Atom(symbol='C', x=0.0, y=0.0, z=0.0)
atom2 = mp.Atom(symbol='H', x=1.0, y=0.0, z=0.0)

# Create structure
structure = mp.AtomicStructure()
structure.add_atom(atom1)
structure.add_atom(atom2)
```

### Working with Frames
```python
# Convert structure to frame
frame = structure.to_frame()

# Create trajectory
trajectory = mp.Trajectory([frame])
```

### Force Field Operations
```python
# Load force field
ff = mp.ForceField("AMBER")
ff.assign_parameters(structure)
```
