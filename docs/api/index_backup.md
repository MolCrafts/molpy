# API Reference

Complete API documentation for all MolPy modules, automatically generated from source code.

## Core Modules

### [Struct Module](struct.md)
Object-oriented molecular structures and building blocks.

::: molpy.core.struct.Entity
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

::: molpy.core.struct.Atom
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

::: molpy.core.struct.Bond
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

::: molpy.core.struct.AtomicStructure
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

::: molpy.core.struct.Struct
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### [Frame Module](frame.md)
High-performance tabular data processing with xarray Datasets.

::: molpy.core.frame.Frame
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### [Trajectory Module](trajectory.md)
Molecular dynamics trajectory handling.

::: molpy.core.trajectory.Trajectory
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
### [ForceField Module](forcefield.md) 
Force field parameter management.

::: molpy.core.forcefield.ForceField
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

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
