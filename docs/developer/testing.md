# Testing

MolPy uses **pytest** for all automated tests. Comprehensive testing is critical for maintaining code quality and enabling safe refactoring.

This guide provides detailed testing guidelines and best practices.

## Testing Philosophy

### Why We Test

- **Correctness** - Ensure code works as intended
- **Regression prevention** - Catch bugs before they reach users
- **Documentation** - Tests show how code should be used
- **Refactoring confidence** - Change code without fear
- **API contracts** - Verify public interfaces remain stable

### What to Test

Test these aspects of your code:

1. **Happy path** - Normal, expected usage
2. **Edge cases** - Boundary conditions, empty inputs, large inputs
3. **Error handling** - Invalid inputs, error conditions
4. **Integration** - Components working together
5. **Regression** - Previously fixed bugs stay fixed

## Test Organization

### Directory Structure

Tests mirror the package structure:

```
tests/
├── test_core/
│   ├── test_frame.py
│   ├── test_block.py
│   ├── test_atomistic.py
│   └── test_box.py
├── test_io/
│   ├── test_pdb.py
│   ├── test_lammps.py
│   └── test_xyz.py
├── test_reacter/
│   └── test_basic.py
└── test_data/
    ├── water.pdb
    ├── methane.xyz
    └── polymer.lammps
```

### Test Files

- Name test files `test_*.py`
- Group related tests in classes
- Use descriptive test names

## Writing Tests

### Basic Test Structure

```python
import pytest
import numpy as np
from molpy.core import Frame, Block

def test_frame_creation_empty():
    """Test creating an empty frame."""
    frame = Frame()
    assert len(frame.blocks()) == 0
    assert frame.box is not None  # Should have default box

def test_frame_add_block():
    """Test adding a block to a frame."""
    frame = Frame()
    block = Block({"x": [1.0, 2.0, 3.0]})

    frame["atoms"] = block

    assert "atoms" in frame.blocks()
    assert len(frame["atoms"]) == 3
    assert np.allclose(frame["atoms"]["x"], [1.0, 2.0, 3.0])
```

### Test Classes

Group related tests in classes:

```python
class TestFrame:
    """Tests for Frame class."""

    def test_creation_empty(self):
        """Test creating an empty frame."""
        frame = Frame()
        assert len(frame.blocks()) == 0

    def test_creation_with_box(self):
        """Test creating frame with custom box."""
        box = Box([10, 10, 10])
        frame = Frame(box=box)
        assert frame.box.lengths == [10, 10, 10]

    def test_add_multiple_blocks(self):
        """Test adding multiple blocks."""
        frame = Frame()
        frame["atoms"] = Block({"x": [1, 2]})
        frame["bonds"] = Block({"i": [0], "j": [1]})

        assert len(frame.blocks()) == 2
        assert "atoms" in frame.blocks()
        assert "bonds" in frame.blocks()
```

### Fixtures

Use fixtures for common test data:

```python
import pytest
from molpy.core import Frame, Block, Box

@pytest.fixture
def empty_frame():
    """Provide an empty frame."""
    return Frame()

@pytest.fixture
def water_frame():
    """Provide a simple water molecule frame."""
    frame = Frame(box=Box([20, 20, 20]))
    frame["atoms"] = Block({
        "x": [0.0, 1.0, -1.0],
        "y": [0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0],
        "type": ["O", "H", "H"]
    })
    return frame

def test_frame_merge(empty_frame, water_frame):
    """Test merging frames."""
    merged = empty_frame.merge(water_frame)
    assert len(merged["atoms"]) == 3
```

### Parametrized Tests

Test multiple inputs with parametrize:

```python
@pytest.mark.parametrize("input_val,expected", [
    (0, 0),
    (1, 1),
    (-1, 1),
    (5, 25),
])
def test_square(input_val, expected):
    """Test square function with various inputs."""
    assert square(input_val) == expected

@pytest.mark.parametrize("format", ["pdb", "xyz", "lammps"])
def test_read_write_roundtrip(format, tmp_path):
    """Test read/write roundtrip for various formats."""
    original = create_test_frame()
    filepath = tmp_path / f"test.{format}"

    write_file(filepath, original, format=format)
    loaded = read_file(filepath, format=format)

    assert_frames_equal(original, loaded)
```

### Testing Exceptions

Test that errors are raised correctly:

```python
def test_invalid_block_type_raises():
    """Test that invalid block type raises TypeError."""
    frame = Frame()

    with pytest.raises(TypeError, match="must be Block"):
        frame["atoms"] = [1, 2, 3]

def test_negative_box_size_raises():
    """Test that negative box size raises ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        Box([-10, 10, 10])

def test_missing_file_raises():
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_pdb("nonexistent.pdb")
```

### Testing Numerical Code

Use appropriate tolerances for floating-point comparisons:

```python
import numpy as np

def test_distance_calculation():
    """Test distance calculation."""
    pos1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([3.0, 4.0, 0.0])

    distance = calculate_distance(pos1, pos2)

    # Use np.allclose for floating-point comparison
    assert np.allclose(distance, 5.0, rtol=1e-6)

def test_center_of_mass():
    """Test center of mass calculation."""
    positions = np.array([[0, 0, 0], [2, 0, 0]])
    masses = np.array([1.0, 1.0])

    com = calculate_com(positions, masses)

    expected = np.array([1.0, 0.0, 0.0])
    assert np.allclose(com, expected)
```

## Test Types

### Unit Tests

Test individual functions and classes in isolation:

```python
def test_block_creation():
    """Unit test for Block creation."""
    data = {"x": [1, 2, 3], "y": [4, 5, 6]}
    block = Block(data)

    assert len(block) == 3
    assert "x" in block.keys()
    assert "y" in block.keys()
```

### Integration Tests

Test components working together:

```python
def test_read_write_integration(tmp_path):
    """Integration test for reading and writing."""
    # Create a frame
    frame = Frame()
    frame["atoms"] = Block({"x": [1, 2], "y": [3, 4]})

    # Write to file
    filepath = tmp_path / "test.pdb"
    write_pdb(filepath, frame)

    # Read back
    loaded = read_pdb(filepath)

    # Verify
    assert len(loaded["atoms"]) == 2
```

### Regression Tests

Test that previously fixed bugs stay fixed:

```python
def test_issue_123_empty_topology():
    """Regression test for issue #123.

    Previously, empty topology caused a crash.
    Now it should handle gracefully.
    """
    frame = Frame()
    topology = Topology(frame)

    # Should not crash
    bonds = topology.bonds()
    assert len(bonds) == 0
```

## Test Data

### Using Test Files

Store small test files in `tests/test_data/`:

```python
from pathlib import Path

TEST_DATA = Path(__file__).parent / "test_data"

def test_read_water_pdb():
    """Test reading water PDB file."""
    filepath = TEST_DATA / "water.pdb"
    frame = read_pdb(filepath)

    assert len(frame["atoms"]) == 3
```

### Creating Test Data

Prefer small, realistic molecules:

```python
def create_water_molecule() -> Frame:
    """Create a simple water molecule for testing."""
    frame = Frame()
    frame["atoms"] = Block({
        "x": [0.0, 0.757, -0.757],
        "y": [0.0, 0.586, 0.586],
        "z": [0.0, 0.0, 0.0],
        "type": ["O", "H", "H"],
        "mass": [16.0, 1.0, 1.0]
    })
    return frame

def create_methane_molecule() -> Frame:
    """Create a methane molecule for testing."""
    # Tetrahedral geometry
    frame = Frame()
    # ... create geometry
    return frame
```

### Temporary Files

Use `tmp_path` fixture for temporary files:

```python
def test_write_file(tmp_path):
    """Test writing to temporary file."""
    frame = create_water_molecule()
    filepath = tmp_path / "output.pdb"

    write_pdb(filepath, frame)

    assert filepath.exists()
    assert filepath.stat().st_size > 0
```

## Markers

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.external
def test_lammps_execution():
    """Test that requires LAMMPS installation."""
    # This test will be skipped unless LAMMPS is available
    ...

@pytest.mark.slow
def test_large_system():
    """Test with large system (slow)."""
    # This test takes a long time
    ...
```

Run specific markers:

```bash
# Skip external tests
pytest -m "not external"

# Run only slow tests
pytest -m "slow"
```

## Coverage

### Running Coverage

```bash
# Run tests with coverage
pytest --cov=molpy tests/

# Generate HTML report
pytest --cov=molpy --cov-report=html tests/

# Open htmlcov/index.html to view
```

### Coverage Goals

- **Overall:** Aim for >80% coverage
- **Core modules:** Aim for >90% coverage
- **New code:** Should have 100% coverage

### What to Cover

Focus coverage on:
- All public APIs
- Error handling paths
- Edge cases
- Critical algorithms

Don't obsess over:
- Trivial getters/setters
- Debug/logging code
- Deprecated code

## Mocking

Use mocking for external dependencies:

```python
from unittest.mock import Mock, patch

def test_external_tool_wrapper(monkeypatch):
    """Test wrapper for external tool."""
    # Mock subprocess.run
    mock_run = Mock(return_value=Mock(returncode=0, stdout="success"))
    monkeypatch.setattr("subprocess.run", mock_run)

    result = run_external_tool("input.txt")

    assert result.success
    mock_run.assert_called_once()
```

## Running Tests

### Basic Commands

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

### Useful Options

```bash
# Show print statements
pytest -s

# Show local variables on failure
pytest -l

# Run last failed tests
pytest --lf

# Run failed tests first
pytest --ff

# Parallel execution (requires pytest-xdist)
pytest -n auto
```

### Configuration

Configure pytest in `pyproject.toml`:

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

## Best Practices

### Test Naming

Use descriptive names that explain what is being tested:

**Good:**
```python
def test_frame_merge_combines_blocks():
    """Test that merge combines blocks from both frames."""
    ...

def test_invalid_atom_type_raises_value_error():
    """Test that invalid atom type raises ValueError."""
    ...
```

**Bad:**
```python
def test_merge():
    """Test merge."""
    ...

def test_error():
    """Test error."""
    ...
```

### Test Independence

Each test should be independent:

**Good:**
```python
def test_a():
    frame = Frame()
    # Test A
    ...

def test_b():
    frame = Frame()  # Fresh frame
    # Test B
    ...
```

**Bad:**
```python
frame = Frame()  # Shared state!

def test_a():
    frame["atoms"] = ...
    ...

def test_b():
    # Depends on test_a running first!
    assert "atoms" in frame.blocks()
```

### Assertions

Use specific assertions:

**Good:**
```python
assert len(atoms) == 3
assert "x" in block.keys()
assert np.allclose(distance, 5.0)
```

**Bad:**
```python
assert atoms  # Too vague
assert block  # What are we checking?
```

### Test Documentation

Document complex tests:

```python
def test_periodic_boundary_wrapping():
    """Test that atoms are wrapped into the box correctly.

    This test verifies that atoms outside the box boundaries
    are wrapped back into the box using periodic boundary
    conditions. It tests all three dimensions and both
    positive and negative overflow.
    """
    box = Box([10, 10, 10])
    positions = np.array([
        [15, 5, 5],   # x overflow
        [5, -3, 5],   # y underflow
        [5, 5, 12],   # z overflow
    ])

    wrapped = wrap_positions(positions, box)

    # All positions should be in [0, 10)
    assert np.all(wrapped >= 0)
    assert np.all(wrapped < 10)
    assert np.allclose(wrapped[0], [5, 5, 5])
    assert np.allclose(wrapped[1], [5, 7, 5])
    assert np.allclose(wrapped[2], [5, 5, 2])
```

## Continuous Integration

Tests run automatically on:

- Every push to GitHub
- Every pull request
- Before merging to main

CI checks:
- All tests pass
- Coverage meets threshold
- Code style (Black, isort)
- Type checking (mypy, if configured)

## Troubleshooting

### Tests Pass Locally but Fail in CI

**Possible causes:**
- Different Python version
- Missing dependencies
- Platform-specific behavior
- Test order dependency

**Solutions:**
- Check CI Python version matches local
- Ensure all dependencies in `pyproject.toml`
- Use `pytest --random-order` to catch order dependencies

### Slow Tests

**Solutions:**
- Use smaller test data
- Mock expensive operations
- Mark slow tests with `@pytest.mark.slow`
- Use fixtures to share setup

### Flaky Tests

**Causes:**
- Random number generation
- Timing dependencies
- External resources

**Solutions:**
- Set random seeds: `np.random.seed(42)`
- Use deterministic test data
- Mock external resources

## Summary Checklist

Before submitting code:

- [ ] All new code has tests
- [ ] Tests are independent
- [ ] Tests have descriptive names
- [ ] Edge cases are tested
- [ ] Error conditions are tested
- [ ] Coverage is adequate (>80%)
- [ ] All tests pass locally
- [ ] No flaky tests

## Further Reading

- [pytest Documentation](https://docs.pytest.org/)
- [pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Testing Best Practices](https://testdriven.io/blog/testing-best-practices/)
