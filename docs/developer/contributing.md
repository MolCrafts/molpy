# Contributing to MolPy

Thank you for your interest in contributing to MolPy! We welcome contributions of all kinds: bug fixes, new features, documentation improvements, and more.

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](https://github.com/MolCrafts/molpy/blob/main/CODE_OF_CONDUCT.md) before contributing.

## How to Contribute

### Reporting Bugs

If you find a bug:

1. **Check existing issues** - Search [GitHub Issues](https://github.com/MolCrafts/molpy/issues) to see if it's already reported
2. **Create a new issue** - If not found, create a detailed bug report including:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs. actual behavior
   - Your environment (OS, Python version, MolPy version)
   - Minimal code example if applicable

### Suggesting Features

Have an idea for a new feature?

1. **Open a discussion** - Start a conversation in [GitHub Discussions](https://github.com/MolCrafts/molpy/discussions)
2. **Describe the feature** - Explain:
   - What problem it solves
   - How it would work
   - Why it's useful for MolPy users
   - Any alternatives you considered
3. **Get feedback** - Discuss with maintainers before implementing

### Contributing Code

#### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/molpy.git
cd molpy
```

#### 2. Set Up Development Environment

Follow the [Development Setup](development-setup.md) guide to configure your environment.

#### 3. Create a Branch

Use descriptive branch names following these conventions:

```bash
# For new features
git checkout -b feature/your-feature-name

# For bug fixes
git checkout -b fix/issue-description

# For documentation
git checkout -b docs/what-you-are-documenting

# For refactoring
git checkout -b refactor/what-you-are-refactoring
```

#### 4. Make Your Changes

Follow these guidelines:

- **Code Style** - Follow the [Coding Style Guide](coding-style.md)
- **Type Hints** - Use type hints for all function signatures
- **Documentation** - Update docstrings and user docs as needed
- **Tests** - Add tests for new functionality (see [Testing Guide](testing.md))

#### 5. Write Good Commit Messages

Use clear, descriptive commit messages:

```bash
# Format: [type] brief description
git commit -m "[feature] add support for XYZ file format"
git commit -m "[fix] correct atom indexing in topology"
git commit -m "[docs] update installation guide"
git commit -m "[test] add tests for Frame.merge()"
```

**Commit types:**
- `[feature]` - New feature
- `[fix]` - Bug fix
- `[docs]` - Documentation changes
- `[test]` - Test additions/improvements
- `[refactor]` - Code refactoring
- `[perf]` - Performance improvements
- `[chore]` - Maintenance tasks

#### 6. Test Your Changes

Before submitting, ensure all tests pass:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=molpy tests/

# Run specific tests
pytest tests/test_core/test_frame.py
```

#### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- **Clear title** - Describe what the PR does
- **Description** - Explain the changes and why they're needed
- **Related issues** - Link to any related issues
- **Checklist** - Complete the PR checklist (see below)

## Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows the [Coding Style Guide](coding-style.md)
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated (docstrings, user guides, tutorials)
- [ ] Commit messages are clear and follow conventions
- [ ] Branch is up to date with `main`
- [ ] No merge conflicts
- [ ] Pre-commit hooks pass (if installed)

## Code Review Process

1. **Automated checks** - CI runs tests and linters
2. **Maintainer review** - A maintainer reviews your code
3. **Feedback** - Address any requested changes
4. **Approval** - Once approved, your PR will be merged
5. **Recognition** - You'll be added to contributors list!

## Development Principles

When contributing, keep these principles in mind:

### Predictable

- Clear data model with minimal surprises
- Explicit is better than implicit
- Avoid hidden state and side effects

### Well-tested

- Every feature backed by tests
- Bug fixes include regression tests
- Use realistic test cases (water, methane, small polymers)

### Documented

- User-facing features have documentation
- Examples are runnable and up-to-date
- Docstrings follow Google style

### Type-safe

- Full type hints throughout
- Use modern Python typing (`list[T]`, `dict[K, V]`)
- Avoid `Any` unless necessary

### Composable

- Small, focused functions and classes
- Core types free from engine-specific logic
- Heavy logic in helper modules, not core classes

## Code Style Guidelines

### Python Style

- Follow **PEP 8**
- Use **Black** for formatting (line length: 88)
- Use **isort** for import sorting
- Type hints for all function signatures

### Naming Conventions

- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Descriptive names over abbreviations

### Imports

```python
# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
from igraph import Graph

# MolPy
from molpy.core import Frame, Block
from molpy.io import read_pdb
```

### Docstrings

Use Google-style docstrings:

```python
def process_frame(
    frame: Frame,
    threshold: float = 0.5,
    normalize: bool = True
) -> np.ndarray | None:
    """Process a frame with optional normalization.

    Args:
        frame: Input frame to process
        threshold: Threshold value for filtering
        normalize: Whether to normalize the output

    Returns:
        Processed array or None if frame is empty

    Raises:
        ValueError: If threshold is negative

    Examples:
        >>> frame = Frame(...)
        >>> result = process_frame(frame, threshold=0.3)
    """
    if threshold < 0:
        raise ValueError("Threshold must be non-negative")

    if frame.is_empty():
        return None

    # Implementation here
    ...
```

## Testing Guidelines

### Writing Tests

- Use **pytest** for all tests
- Aim for >80% code coverage
- Test edge cases and error conditions
- Use descriptive test names

### Test Structure

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

## Documentation

### Types of Documentation

1. **Docstrings** - In-code documentation for all public APIs
2. **User Guide** - Jupyter notebooks in `docs/user-guide/`
3. **Tutorials** - Jupyter notebooks in `docs/tutorials/`
4. **API Reference** - Auto-generated from docstrings
5. **Developer Docs** - This section!

### Adding Documentation

When adding new features:

- Update relevant docstrings
- Add examples to docstrings
- Update user guide if it's a major feature
- Create tutorial if it's a new workflow
- Update API reference structure if needed

### Building Documentation

```bash
# Install doc dependencies
pip install -e ".[doc]"

# Serve documentation locally
mkdocs serve

# Open http://127.0.0.1:8000
```

## Release Notes

When adding user-visible features or behavior changes:

- Add an entry to `changelog/releases.md`
- Use clear, user-friendly language
- Mention breaking changes prominently

## Questions?

- **General questions** - [GitHub Discussions](https://github.com/MolCrafts/molpy/discussions)
- **Bug reports** - [GitHub Issues](https://github.com/MolCrafts/molpy/issues)
- **Documentation** - [MolPy Docs](https://molcrafts.github.io/molpy/)

## License

By contributing to MolPy, you agree that your contributions will be licensed under the BSD-3-Clause License.

Thank you for contributing to MolPy! 🚀
