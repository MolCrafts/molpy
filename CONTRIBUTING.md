# Contributing to MolPy

Thank you for your interest in contributing to MolPy! We welcome contributions from the community and are excited to see what you'll build.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of molecular simulation concepts

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR-USERNAME/molpy.git
   cd molpy
   ```

2. **Create Development Environment**
   ```bash
   # Using conda (recommended)
   conda create -n molpy-dev python=3.10
   conda activate molpy-dev
   
   # Or using venv
   python -m venv molpy-dev
   source molpy-dev/bin/activate  # Linux/Mac
   # molpy-dev\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Verify Installation**
   ```bash
   pytest tests/
   ```

## 🛠️ Development Guidelines

### Code Style
We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting

Run before committing:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

### Testing
- Write tests for all new features
- Maintain test coverage above 80%
- Use pytest for testing framework

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=molpy --cov-report=html
```

### Documentation
- Use Google-style docstrings
- Include type hints
- Add examples for public APIs
- Update README.md if needed

## 🐛 Reporting Issues

### Bug Reports
When reporting bugs, please include:
- Python version and OS
- MolPy version
- Minimal code example
- Full error traceback
- Expected vs actual behavior

### Feature Requests
For new features, please provide:
- Clear description of the feature
- Use case and motivation
- Proposed API (if applicable)
- Willingness to implement

## 📝 Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   pytest tests/
   black --check src/
   isort --check-only src/
   flake8 src/
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Use descriptive title and description
   - Reference any related issues
   - Include test results
   - Request review from maintainers

### Commit Message Convention
We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `test:` test additions/modifications
- `refactor:` code refactoring
- `perf:` performance improvements

## 🎯 Areas for Contribution

### High Priority
- **Performance Optimization**: Cell lists, neighbor lists
- **Force Field Support**: Additional force field formats
- **Analysis Tools**: Trajectory analysis functions
- **Documentation**: Examples and tutorials

### Medium Priority  
- **Visualization**: Integration with plotting libraries
- **File I/O**: Additional file format support
- **Testing**: Expand test coverage
- **Type Safety**: Improve type annotations

### Good First Issues
Look for issues labeled `good-first-issue` in our GitHub repository. These are great starting points for new contributors.

## 🏗️ Architecture Overview

### Core Components
- **Entity System**: Base classes for molecular entities
- **Frame System**: xarray.DataTree-based data containers
- **Force Fields**: OPLS-AA, AMBER, custom force fields
- **I/O System**: File reading/writing via chemfiles
- **Serialization**: Unified to_dict/from_dict interface

### Design Principles
- **Modularity**: Components should be loosely coupled
- **Extensibility**: Easy to add new features
- **Performance**: Efficient for large systems
- **Usability**: Intuitive APIs for researchers

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Help others learn and grow
- Focus on what's best for the community
- Show empathy towards other contributors

### Communication
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas  
- **Pull Requests**: Code changes and improvements

## 📚 Resources

### Documentation
- [API Reference](docs/api/)
- [User Guide](docs/guide/)
- [Examples](examples/)

### Related Projects
- [ChemFiles](https://chemfiles.org/): Chemical file I/O
- [xarray](https://xarray.pydata.org/): N-dimensional arrays
- [MDAnalysis](https://www.mdanalysis.org/): Trajectory analysis

## 🙏 Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for their contributions
- README.md contributors section
- GitHub contributors page

Thank you for contributing to MolPy! 🧬
