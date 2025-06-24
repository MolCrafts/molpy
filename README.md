# MolPy 🧬

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

**MolPy** is a modern, high-performance Python framework for molecular simulation and analysis. Built with flexibility and extensibility in mind, MolPy provides elegant data structures and comprehensive toolkits that let researchers focus on science rather than infrastructure.

## ✨ Key Features

- **🏗️ Modern Architecture**: Built on xarray.DataTree for hierarchical data management
- **🔄 Unified Serialization**: Consistent to_dict/from_dict interface across all components
- **📦 Flexible Data Structures**: Support for atoms, bonds, angles, dihedrals, and complex molecular systems
- **🎯 Force Field Integration**: Comprehensive force field management and type assignment
- **📊 Trajectory Analysis**: Efficient handling of molecular dynamics trajectories
- **🧮 Advanced Algorithms**: Built-in optimization, packing, and reaction modeling
- **🔌 Extensible Design**: Plugin-ready architecture for custom functionality
- **⚡ High Performance**: Optimized for large-scale molecular systems

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MolCrafts/molpy.git
cd molpy

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
import molpy as mp
import numpy as np

# Create a simulation box
box = mp.Box.cubic(10.0)

# Create atoms
atoms_data = {
    'x': [0.0, 1.0, 2.0],
    'y': [0.0, 0.0, 0.0],
    'z': [0.0, 0.0, 0.0],
    'element': ['C', 'C', 'C'],
    'type': [1, 1, 1]
}

# Create a frame
frame = mp.Frame(data={'atoms': atoms_data}, box=box)

# Save and load
frame_dict = frame.to_dict()  # Serialize to dictionary
restored_frame = mp.Frame.from_dict(frame_dict)  # Restore from dictionary

# Save to file
frame.save('system.h5', format='hdf5')
loaded_frame = mp.Frame.load('system.h5')
```

## 📋 Dependencies

**Core Dependencies:**
- [numpy](https://github.com/numpy/numpy) - Numerical computing
- [xarray](https://github.com/pydata/xarray) - N-dimensional labeled arrays
- [python-igraph](https://github.com/igraph/python-igraph) - Graph analysis
- [lark](https://github.com/lark-parser/lark) - Parsing SMILES/SMARTS


> 💡 **Note**: This project is actively developed. We welcome suggestions, feature requests, and contributions!

## 🗺️ Roadmap

### ✅ Completed Features
- **Data Structures**: Static and dynamic molecular data containers
- **File I/O**: Read and write support via Chemfiles integration
- **Geometry**: Triclinic box support with PBC handling
- **Force Fields**: Comprehensive force field management system
- **Serialization**: Unified to_dict/from_dict interface for all components
- **Frame System**: Advanced frame concatenation and manipulation

### 🚧 In Development
- **Performance**: Cell lists & neighbor lists for efficient calculations
- **Potentials**: Built-in potential function library
- **Optimization**: Molecular structure optimization algorithms
- **Analysis**: Advanced trajectory analysis tools

### 📅 Planned Features
- **Modeling**: Advanced molecular modeling capabilities
- **Typification**: Automated atom type assignment
- **SMARTS/SMILES**: Chemical pattern matching and manipulation
- **Visualization**: Interactive 3D molecular visualization
- **Plugin System**: Extensible architecture for custom modules
- **Documentation**: Comprehensive user guides and API documentation

### 🎯 Future Vision
After core functionality stabilization, we plan to:
- Abstract performance-critical components to high-performance C++
- Provide bindings for multiple programming languages
- Develop GPU acceleration for large-scale simulations

## 🌟 Ecosystem

### 🧠 Machine Learning Integration
**[MolNex](https://github.com/MolCrafts/molnex)** - Universal potential training platform
- Neural network potential development
- Transfer learning for molecular systems
- Integration with popular ML frameworks

### 🎨 Interactive Visualization
**[MolVis](https://github.com/Roy-Kid/molvis)** *(coming soon)*
- Production-level visualization using Babylon.js
- Real-time molecular manipulation in Jupyter notebooks
- WebGL-accelerated rendering for complex systems
- Interactive debugging and analysis tools

## 📚 Documentation

### Examples & Tutorials
Check out the `examples/` directory for:
- **Basic Usage**: Simple molecule creation and manipulation
- **Force Fields**: OPLS-AA, AMBER, and custom force field setup
- **Trajectories**: MD trajectory analysis and processing
- **Advanced**: Polymer building, reaction modeling, and optimization

### API Reference
Comprehensive API documentation is available in the `docs/` directory.

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **🐛 Bug Reports**: Use GitHub Issues to report bugs
2. **💡 Feature Requests**: Suggest new features and improvements
3. **📖 Documentation**: Help improve documentation and examples
4. **🔧 Code Contributions**: Submit pull requests with new features or fixes

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/MolCrafts/molpy.git
cd molpy

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run code formatting
black src/
isort src/
```

## 📄 License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ by the MolCrafts team** 