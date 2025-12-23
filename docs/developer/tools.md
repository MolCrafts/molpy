# Development Tools

This guide covers the development tools used in MolPy and their configurations.

## Overview

MolPy uses modern Python development tools:

- **pytest** - Testing framework
- **Black** - Code formatting
- **isort** - Import sorting
- **pre-commit** - Git hooks
- **mkdocs** - Documentation
- **coverage** - Code coverage

## Testing: pytest

### Installation

```bash
pip install pytest pytest-cov pytest-mock
```

### Configuration

Configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "external: tests requiring external tools",
    "slow: slow tests",
]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

### Basic Usage

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run specific file
pytest tests/test_core/test_frame.py

# Run specific test
pytest tests/test_core/test_frame.py::test_frame_creation

# Run tests matching pattern
pytest -k "lammps"
```

### Advanced Usage

```bash
# Parallel execution (requires pytest-xdist)
pytest -n auto

# Show print statements
pytest -s

# Show local variables on failure
pytest -l

# Run last failed tests
pytest --lf

# Run failed tests first, then others
pytest --ff

# Generate JUnit XML report
pytest --junit-xml=report.xml
```

### Markers

Use markers to categorize tests:

```python
import pytest

@pytest.mark.external
def test_lammps():
    """Test requiring LAMMPS."""
    ...

@pytest.mark.slow
def test_large_system():
    """Slow test."""
    ...
```

Run specific markers:

```bash
# Skip external tests
pytest -m "not external"

# Run only slow tests
pytest -m "slow"

# Combine markers
pytest -m "slow and not external"
```

### Fixtures

Define reusable test data:

```python
import pytest

@pytest.fixture
def water_frame():
    """Provide water molecule frame."""
    from molpy.core import Frame, Block
    frame = Frame()
    frame["atoms"] = Block({
        "x": [0.0, 0.757, -0.757],
        "y": [0.0, 0.586, 0.586],
        "z": [0.0, 0.0, 0.0],
    })
    return frame

def test_something(water_frame):
    """Test using water frame."""
    assert len(water_frame["atoms"]) == 3
```

### Coverage

```bash
# Run with coverage
pytest --cov=molpy tests/

# Generate HTML report
pytest --cov=molpy --cov-report=html tests/

# Show missing lines
pytest --cov=molpy --cov-report=term-missing tests/

# Fail if coverage below threshold
pytest --cov=molpy --cov-fail-under=80 tests/
```

## Formatting: Black

### Installation

```bash
pip install black
```

### Configuration

Configuration in `pyproject.toml`:

```toml
[tool.black]
line-length = 88
target-version = ['py312']
include = '\\.pyi?$'
extend-exclude = '''
/(
  \\.eggs
  | \\.git
  | \\.mypy_cache
  | \\.venv
  | build
  | dist
)/
'''
```

### Usage

```bash
# Format all files
black .

# Format specific file
black molpy/core/frame.py

# Check without modifying
black --check .

# Show diff
black --diff .

# Exclude files
black --exclude "/(build|dist)/" .
```

### Editor Integration

**VS Code:**
```json
{
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

**PyCharm:**
1. Install Black plugin
2. Settings → Tools → Black → Enable
3. Settings → Tools → Actions on Save → Reformat code

## Import Sorting: isort

### Installation

```bash
pip install isort
```

### Configuration

Configuration in `pyproject.toml`:

```toml
[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
```

### Usage

```bash
# Sort imports in all files
isort .

# Sort specific file
isort molpy/core/frame.py

# Check without modifying
isort --check .

# Show diff
isort --diff .
```

### Import Groups

isort organizes imports into groups:

```python
# 1. Standard library
import os
import sys
from pathlib import Path

# 2. Third-party
import numpy as np
from igraph import Graph

# 3. Local
from molpy.core import Frame
from molpy.io import read_pdb
```

## Pre-commit Hooks

### Installation

```bash
pip install pre-commit
pre-commit install
```

### Configuration

Configuration in `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1024']
      - id: check-merge-conflict
      - id: mixed-line-ending
      - id: check-toml
      - id: check-json

  - repo: https://github.com/psf/black
    rev: 25.11.0
    hooks:
    -   id: black

  - repo: https://github.com/kynan/nbstripout
    rev: 0.8.2
    hooks:
      - id: nbstripout
        files: \\.ipynb$
        args: ["--extra-keys=metadata.kernelspec metadata.language_info"]
```

### Usage

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files molpy/core/frame.py

# Update hook versions
pre-commit autoupdate

# Skip hooks for a commit
git commit --no-verify
```

### Hooks Included

- **trailing-whitespace** - Remove trailing whitespace
- **end-of-file-fixer** - Ensure files end with newline
- **check-yaml** - Validate YAML files
- **check-toml** - Validate TOML files
- **check-json** - Validate JSON files
- **check-added-large-files** - Prevent large files
- **black** - Format code
- **nbstripout** - Strip notebook outputs

## Documentation: MkDocs

### Installation

```bash
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-jupyter
```

### Configuration

Configuration in `mkdocs.yml`:

```yaml
site_name: molpy
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.instant
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
          options:
            docstring_style: google
            show_source: true
  - mkdocs-jupyter:
      execute: true

markdown_extensions:
  - admonition
  - pymdownx.highlight
  - pymdownx.superfences
  - pymdownx.tabbed
```

### Usage

```bash
# Serve documentation locally
mkdocs serve

# Build static site
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy

# Validate configuration
mkdocs build --strict
```

### Writing Documentation

**Markdown files:**
```markdown
# Page Title

Regular markdown content.

## Code Examples

```python
from molpy.core import Frame
frame = Frame()
```

## Admonitions

!!! note
    This is a note.

!!! warning
    This is a warning.
```

**API documentation:**
```markdown
# API Reference

::: molpy.core.Frame
    options:
      show_source: true
      members: true
```

## Coverage: coverage.py

### Installation

```bash
pip install coverage pytest-cov
```

### Usage with pytest

```bash
# Run tests with coverage
pytest --cov=molpy tests/

# Generate HTML report
pytest --cov=molpy --cov-report=html tests/

# Generate XML report (for CI)
pytest --cov=molpy --cov-report=xml tests/

# Show missing lines
pytest --cov=molpy --cov-report=term-missing tests/
```

### Standalone Usage

```bash
# Run tests with coverage
coverage run -m pytest

# Generate report
coverage report

# Generate HTML report
coverage html

# Erase previous data
coverage erase
```

### Configuration

Configuration in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["molpy"]
omit = [
    "*/tests/*",
    "*/test_*.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

## Type Checking: mypy (Optional)

### Installation

```bash
pip install mypy
```

### Configuration

Configuration in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = "igraph.*"
ignore_missing_imports = true
```

### Usage

```bash
# Check all files
mypy molpy

# Check specific file
mypy molpy/core/frame.py

# Generate HTML report
mypy --html-report mypy-report molpy
```

## Linting: Ruff (Optional)

### Installation

```bash
pip install ruff
```

### Configuration

Configuration in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
]
ignore = [
    "E501",  # line too long (handled by black)
]
```

### Usage

```bash
# Check all files
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Check specific file
ruff check molpy/core/frame.py
```

## Build Tools

### Building Packages

```bash
# Install build tools
pip install build

# Build source distribution and wheel
python -m build

# Output in dist/
ls dist/
# molcrafts-molpy-0.3.0.tar.gz
# molcrafts_molpy-0.3.0-py3-none-any.whl
```

### Publishing to PyPI

```bash
# Install twine
pip install twine

# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Continuous Integration

### GitHub Actions

Example workflow (`.github/workflows/test.yml`):

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run tests
        run: |
          pytest --cov=molpy --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Editor Configuration

### VS Code

Recommended `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "editor.formatOnSave": true,
  "editor.rulers": [88],
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

Recommended extensions:
- Python (Microsoft)
- Pylance
- Jupyter
- autoDocstring

### PyCharm

Settings:
1. **Code Style → Python → Line length:** 88
2. **Tools → Python Integrated Tools → Testing:** pytest
3. **Tools → Black:** Enable
4. **Editor → Code Style → Python → Imports:** Use isort

## Summary

### Essential Tools

- **pytest** - Testing
- **Black** - Formatting
- **isort** - Import sorting
- **pre-commit** - Git hooks

### Optional Tools

- **mypy** - Type checking
- **ruff** - Fast linting
- **coverage** - Code coverage

### Quick Setup

```bash
# Install all dev tools
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black .
isort .

# Build docs
mkdocs serve
```

## Further Reading

- [pytest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)
- [isort Documentation](https://pycqa.github.io/isort/)
- [pre-commit Documentation](https://pre-commit.com/)
- [MkDocs Documentation](https://www.mkdocs.org/)
