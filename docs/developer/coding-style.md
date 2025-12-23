# Coding Style

MolPy follows a "clean, explicit, type-driven" coding style designed to make the library easy to read, test, and extend.

This guide provides detailed coding standards and best practices for contributors.

## Core Principles

### 1. Explicit Over Implicit

Prefer clear, explicit code over clever tricks:

**Good:**
```python
def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculate Euclidean distance between two positions."""
    diff = pos1 - pos2
    return np.sqrt(np.sum(diff ** 2))
```

**Bad:**
```python
def calc_dist(p1, p2):
    return np.linalg.norm(p1 - p2)  # Unclear what p1, p2 are
```

### 2. Predictable Behavior

Avoid hidden state and side effects:

**Good:**
```python
def merge_frames(frame1: Frame, frame2: Frame) -> Frame:
    """Merge two frames into a new frame."""
    merged = Frame()
    # ... merge logic
    return merged
```

**Bad:**
```python
def merge_frames(frame1: Frame, frame2: Frame) -> None:
    """Merge frame2 into frame1 (modifies frame1)."""
    # Modifying input arguments is surprising
    frame1.blocks.update(frame2.blocks)
```

### 3. Type Safety

Use type hints everywhere:

**Good:**
```python
from typing import Sequence

def process_atoms(
    positions: np.ndarray,
    masses: Sequence[float],
    box: Box | None = None
) -> dict[str, np.ndarray]:
    """Process atom data."""
    ...
```

**Bad:**
```python
def process_atoms(positions, masses, box=None):
    """Process atom data."""
    ...
```

## Python Style Guidelines

### PEP 8 Compliance

Follow [PEP 8](https://pep8.org/) with these specifics:

- **Line length:** 88 characters (Black default)
- **Indentation:** 4 spaces (never tabs)
- **Blank lines:** 2 between top-level definitions, 1 between methods
- **Imports:** Grouped and sorted (see [Imports](#imports))

### Naming Conventions

#### Variables and Functions

Use `snake_case`:

```python
atom_count = 100
bond_length = 1.5

def calculate_center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    ...
```

#### Classes

Use `PascalCase`:

```python
class Frame:
    ...

class AtomicForceField:
    ...

class LammpsDataReader:
    ...
```

#### Constants

Use `UPPER_CASE`:

```python
DEFAULT_CUTOFF = 10.0
MAX_ITERATIONS = 1000
AVOGADRO_NUMBER = 6.022e23
```

#### Private Members

Prefix with single underscore:

```python
class Frame:
    def __init__(self):
        self._blocks: dict[str, Block] = {}

    def _validate_block(self, block: Block) -> None:
        """Private validation method."""
        ...
```

### Imports

Organize imports in three groups, sorted alphabetically:

```python
# 1. Standard library
import os
import sys
from pathlib import Path
from typing import Any, Sequence

# 2. Third-party packages
import numpy as np
from igraph import Graph

# 3. MolPy modules
from molpy.core import Block, Frame
from molpy.io import read_pdb, write_lammps_data
```

**Rules:**
- No wildcard imports (`from x import *`)
- Use absolute imports, not relative
- Import modules, not individual items (exceptions for common items)
- Use `as` for common abbreviations: `numpy as np`, `pandas as pd`

### Type Hints

Use modern Python 3.12+ type hints:

**Good:**
```python
from typing import Literal

def read_file(
    path: str | Path,
    format: Literal["pdb", "xyz", "lammps"] = "pdb",
    options: dict[str, Any] | None = None
) -> Frame:
    """Read a molecular structure file."""
    ...
```

**Bad:**
```python
from typing import Dict, List, Optional, Union

def read_file(
    path: Union[str, Path],
    format: str = "pdb",
    options: Optional[Dict[str, Any]] = None
) -> Frame:
    ...
```

**Prefer:**
- `list[T]` over `List[T]`
- `dict[K, V]` over `Dict[K, V]`
- `X | Y` over `Union[X, Y]`
- `X | None` over `Optional[X]`
- `tuple[int, ...]` for variable-length tuples

**Avoid `Any`** unless absolutely necessary. If you must use it, add a comment explaining why.

## Code Organization

### File Structure

One class per file (with exceptions for small helper classes):

```
molpy/
├── core/
│   ├── __init__.py
│   ├── frame.py          # Frame class
│   ├── block.py          # Block class
│   ├── atomistic.py      # Atomistic classes
│   └── box.py            # Box class
```

### Module Structure

Organize modules consistently:

```python
"""Module docstring describing the module's purpose.

This module provides...
"""

# Imports
import os
from typing import Any

import numpy as np

from molpy.core import Frame

# Constants
DEFAULT_TOLERANCE = 1e-6

# Private helpers
def _validate_input(data: np.ndarray) -> None:
    """Private helper function."""
    ...

# Public classes
class MyClass:
    """Public class."""
    ...

# Public functions
def public_function() -> None:
    """Public function."""
    ...
```

### Function Length

Keep functions focused and concise:

- **Ideal:** < 20 lines
- **Acceptable:** < 50 lines
- **Refactor if:** > 50 lines

If a function is too long, break it into smaller helper functions.

## Docstrings

Use **Google-style docstrings** for all public APIs:

### Function Docstrings

```python
def calculate_rdf(
    frame: Frame,
    r_max: float = 10.0,
    n_bins: int = 100,
    atom_types: tuple[str, str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate radial distribution function.

    Computes the RDF g(r) for atom pairs in the frame up to a maximum
    distance r_max.

    Args:
        frame: Input frame containing atomic positions
        r_max: Maximum distance for RDF calculation in Angstroms
        n_bins: Number of bins for the histogram
        atom_types: Pair of atom types to consider. If None, use all atoms.

    Returns:
        Tuple of (r, g_r) where:
            - r: Array of distances (bin centers)
            - g_r: RDF values at each distance

    Raises:
        ValueError: If r_max is negative or n_bins < 1
        KeyError: If specified atom_types not found in frame

    Examples:
        >>> frame = read_pdb("water.pdb")
        >>> r, g_r = calculate_rdf(frame, r_max=5.0)
        >>> plt.plot(r, g_r)

    Notes:
        The RDF is normalized by the ideal gas density.
    """
    if r_max <= 0:
        raise ValueError("r_max must be positive")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    # Implementation...
    ...
```

### Class Docstrings

```python
class Frame:
    """Container for molecular structure data.

    A Frame represents a molecular system at a single point in time,
    containing atomic positions, topology, and metadata organized into
    named blocks.

    Attributes:
        box: Simulation box defining periodic boundaries
        metadata: Dictionary of arbitrary metadata

    Examples:
        >>> frame = Frame()
        >>> frame["atoms"] = Block({"x": [0, 1, 2], "y": [0, 0, 0]})
        >>> print(frame.blocks())
        ['atoms']

    Notes:
        Frames are designed to be immutable containers. Modifications
        create new Frame instances rather than modifying in place.
    """

    def __init__(self, box: Box | None = None):
        """Initialize a new Frame.

        Args:
            box: Optional simulation box. If None, creates infinite box.
        """
        ...
```

### Property Docstrings

```python
@property
def n_atoms(self) -> int:
    """Number of atoms in the frame.

    Returns:
        Total atom count across all blocks.
    """
    return sum(len(block) for block in self._blocks.values())
```

## Error Handling

### Validate Early

Fail fast with clear error messages:

```python
def set_positions(self, positions: np.ndarray) -> None:
    """Set atomic positions.

    Args:
        positions: Array of shape (n_atoms, 3)

    Raises:
        ValueError: If positions shape is invalid
        TypeError: If positions is not a numpy array
    """
    if not isinstance(positions, np.ndarray):
        raise TypeError(f"positions must be ndarray, got {type(positions)}")

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions must have shape (n, 3), got {positions.shape}"
        )

    self._positions = positions
```

### Use Appropriate Exceptions

- `ValueError` - Invalid value
- `TypeError` - Wrong type
- `KeyError` - Missing key
- `FileNotFoundError` - File not found
- `RuntimeError` - Runtime error

### Don't Silently Swallow Exceptions

**Bad:**
```python
try:
    result = risky_operation()
except Exception:
    pass  # Silent failure!
```

**Good:**
```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.warning(f"Operation failed: {e}")
    result = default_value
```

## Logging

Use Python's logging module, not print statements:

```python
import logging

logger = logging.getLogger(__name__)

def process_data(data: np.ndarray) -> np.ndarray:
    """Process data with logging."""
    logger.debug(f"Processing data with shape {data.shape}")

    if data.size == 0:
        logger.warning("Empty data array provided")
        return data

    # Process...
    result = ...

    logger.info(f"Processed {len(result)} items")
    return result
```

## Core Data Structures

Keep core types (`Frame`, `Block`, `Box`, `Atomistic`) clean:

### Value-like Containers

Core types should behave like immutable values:

```python
# Good: Returns new Frame
def merge(self, other: Frame) -> Frame:
    """Merge with another frame, returning new Frame."""
    merged = Frame(box=self.box)
    # ... merge logic
    return merged

# Bad: Modifies in place
def merge(self, other: Frame) -> None:
    """Merge other frame into this one."""
    self._blocks.update(other._blocks)
```

### Avoid Engine-Specific Logic

Don't put LAMMPS, GROMACS, or format-specific code in core classes:

**Bad:**
```python
class Frame:
    def to_lammps_data(self, filename: str) -> None:
        """Write to LAMMPS data file."""
        # LAMMPS-specific logic in core class!
        ...
```

**Good:**
```python
# In molpy/io/data/lammps.py
def write_lammps_data(filename: str, frame: Frame) -> None:
    """Write frame to LAMMPS data file."""
    # LAMMPS-specific logic in IO module
    ...
```

## Testing Code Style

Tests should also follow style guidelines:

```python
import pytest
import numpy as np
from molpy.core import Frame, Block

class TestFrame:
    """Tests for Frame class."""

    def test_creation_empty(self):
        """Test creating an empty frame."""
        frame = Frame()
        assert len(frame.blocks()) == 0

    def test_add_block(self):
        """Test adding a block to frame."""
        frame = Frame()
        block = Block({"x": [1, 2, 3]})
        frame["atoms"] = block

        assert "atoms" in frame.blocks()
        assert len(frame["atoms"]) == 3

    def test_invalid_block_raises_type_error(self):
        """Test that invalid block type raises TypeError."""
        frame = Frame()

        with pytest.raises(TypeError, match="must be Block"):
            frame["atoms"] = [1, 2, 3]  # List, not Block
```

## Tools

### Black

Automatic code formatting:

```bash
# Format all files
black .

# Check without modifying
black --check .

# Format specific file
black molpy/core/frame.py
```

Configuration in `pyproject.toml`:
```toml
[tool.black]
line-length = 88
target-version = ['py312']
```

### isort

Import sorting:

```bash
# Sort imports
isort .

# Check without modifying
isort --check .
```

Configuration in `pyproject.toml`:
```toml
[tool.isort]
profile = "black"
line_length = 88
```

### Pre-commit

Automatically run checks before commits:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Summary Checklist

Before submitting code, ensure:

- [ ] Follows PEP 8 style
- [ ] Uses type hints everywhere
- [ ] Has Google-style docstrings
- [ ] Imports are organized and sorted
- [ ] Functions are focused and concise
- [ ] Error messages are clear and helpful
- [ ] No print statements (use logging)
- [ ] Black formatting applied
- [ ] Tests follow same style guidelines

## Further Reading

- [PEP 8 - Style Guide for Python Code](https://pep8.org/)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Black Documentation](https://black.readthedocs.io/)
