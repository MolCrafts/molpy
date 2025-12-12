# Contributing to MolPy

Thank you for your interest in contributing to MolPy! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git
- Basic knowledge of molecular modeling (helpful but not required)

### Setting Up Development Environment

1. **Fork and clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/molpy.git
cd molpy
```

2. **Create a virtual environment:**

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies:**

```bash
pip install -e ".[dev]"
```

4. **Install pre-commit hooks:**

```bash
pre-commit install
```

5. **Verify installation:**

```bash
pytest tests/
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

### 2. Make Changes

- Write clear, concise code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_core/test_frame.py

# Run with coverage
pytest --cov=molpy tests/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature"
```

**Commit message format:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/improvements
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting (configured in `pyproject.toml`)
- Use [Ruff](https://docs.astral.sh/ruff/) for linting
- Use type hints for all function signatures

**Example:**

```python
from typing import Optional
import numpy as np
from molpy.core import Frame

def process_frame(
    frame: Frame,
    threshold: float = 0.5,
    normalize: bool = True
) -> Optional[np.ndarray]:
    """Process a frame with optional normalization.
    
    Args:
        frame: Input frame to process
        threshold: Threshold value for filtering
        normalize: Whether to normalize the output
        
    Returns:
        Processed array or None if frame is empty
    """
    if frame.is_empty():
        return None
    
    # Implementation here
    pass
```

### Docstring Style

Use Google-style docstrings:

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
        
    Examples:
        >>> result = function_name(value1, value2)
        >>> print(result)
        expected_output
    """
```

### Code Organization

- One class per file (with exceptions for small helper classes)
- Group related functions in modules
- Use `__init__.py` to expose public APIs
- Keep functions small and focused (< 50 lines ideally)

## Testing

### Writing Tests

- Use `pytest` for all tests
- Aim for >80% code coverage
- Test edge cases and error conditions
- Use descriptive test names

**Example:**

```python
import pytest
from molpy.core import Frame, Block

def test_frame_creation_empty():
    """Test creating an empty frame."""
    frame = Frame()
    assert len(frame.blocks()) == 0

def test_frame_add_block():
    """Test adding a block to a frame."""
    frame = Frame()
    block = Block({"x": [1, 2, 3]})
    frame["atoms"] = block
    assert "atoms" in frame.blocks()

def test_frame_invalid_block_raises():
    """Test that invalid block raises TypeError."""
    frame = Frame()
    with pytest.raises(TypeError):
        frame["atoms"] = "invalid"
```

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_core/

# With coverage
pytest --cov=molpy --cov-report=html

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Documentation

### Updating Documentation

Documentation is built with MkDocs and includes:

1. **Docstrings** - In-code documentation
2. **User Guide** - Jupyter notebooks in `docs/user-guide/`
3. **Tutorials** - Jupyter notebooks in `docs/tutorials/`
4. **API Reference** - Auto-generated from docstrings

### Building Documentation Locally

```bash
# Install doc dependencies
pip install -e ".[doc]"

# Serve documentation
cd docs
mkdocs serve

# Open http://127.0.0.1:8000
```

### Adding Examples

When adding new features, please include:

1. Docstring examples
2. Unit tests
3. User guide notebook (if major feature)
4. Tutorial (if workflow-related)

## Submitting Changes

### Pull Request Process

1. **Update documentation** - Ensure all changes are documented
2. **Add tests** - All new code should have tests
3. **Run checks** - Ensure all tests and linters pass
4. **Update CHANGELOG** - Add entry for your changes
5. **Create PR** - Use the PR template
6. **Address feedback** - Respond to review comments

### PR Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main
- [ ] No merge conflicts

### Review Process

1. Automated checks run (tests, linting)
2. Maintainer reviews code
3. Feedback addressed
4. PR approved and merged

## Questions?

- **General questions:** [GitHub Discussions](https://github.com/MolCrafts/molpy/discussions)
- **Bug reports:** [GitHub Issues](https://github.com/MolCrafts/molpy/issues)
- **Documentation:** [MolPy Docs](https://molcrafts.github.io/molpy/)

## Recognition

Contributors will be acknowledged in:
- `CONTRIBUTORS.md`
- Release notes
- Documentation credits

Thank you for contributing to MolPy! 🎉
