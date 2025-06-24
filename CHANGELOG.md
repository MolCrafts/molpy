# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Unified Serialization Interface**: Complete to_dict/from_dict implementation across all core components
- **Enhanced Frame Class**: Built on xarray.DataTree for improved data management
- **JSON Compatibility**: Full JSON serialization support for all molecular data
- **Dynamic Type Reconstruction**: Automatic object restoration from serialized data
- **Comprehensive Examples**: New example scripts demonstrating key features
- **HDF5 Support**: Built-in HDF5 serialization for high-performance storage

### Changed
- **Frame Architecture**: Migrated from legacy data structures to xarray.DataTree
- **Box Serialization**: Updated to unified interface with matrix/pbc/origin format
- **Entity Base Class**: Enhanced with recursive serialization capabilities
- **ForceField Handling**: Complete serialization of all forcefield components

### Improved
- **Performance**: Optimized concatenation and data manipulation operations
- **Memory Usage**: More efficient data storage with xarray backend
- **Error Handling**: Robust error recovery and validation throughout
- **Documentation**: Enhanced docstrings and type hints across codebase

### Fixed
- **Dimension Alignment**: Resolved issues with frame concatenation
- **Type Safety**: Improved type validation and conversion
- **Memory Leaks**: Fixed potential memory issues in large dataset handling

## [0.1.0] - 2024-12-01

### Added
- Initial release with basic molecular data structures
- Force field support for common formats
- Chemical file I/O via chemfiles
- Basic trajectory handling
- Graph-based molecular representation

### Core Features
- Atom, Bond, Angle, Dihedral entities
- Simulation box with PBC support
- Frame-based data organization
- Force field type assignment
- Basic optimization algorithms

---

**Note**: This project is under active development. Version numbers and release dates are preliminary.
