# MolPy Data Module

This module provides a unified interface for accessing built-in data files such as force field parameters, molecule templates, and other resources.

## Features

- **Unified API**: Access all data files through a single interface
- **Python Package Resources**: Uses `importlib.resources` for reliable package data access
- **Backward Compatible**: Falls back to file-based access for older Python versions
- **Type-Safe**: Returns `Path` objects for file paths
- **Convenient Functions**: Provides convenience functions for specific data types

## Usage

### Basic Usage

```python
from molpy.data import get_path, list_files, exists

# Get path to a data file
path = get_path("forcefield/oplsaa.xml")
print(path)  # /path/to/molpy/data/forcefield/oplsaa.xml

# Check if a file exists
if exists("forcefield/oplsaa.xml"):
    print("File exists")

# List files in a subdirectory
for file in list_files("forcefield"):
    print(file)  # forcefield/oplsaa.xml, forcefield/tip3p.xml
```

### Force Field Data

```python
from molpy.data import get_forcefield_path, list_forcefields

# Get path to a force field file
path = get_forcefield_path("oplsaa.xml")
print(path)  # /path/to/molpy/data/forcefield/oplsaa.xml

# List available force fields
forcefields = list_forcefields()
print(forcefields)  # ['oplsaa.xml', 'tip3p.xml']
```

### Using Submodules

```python
from molpy.data.forcefield import get_forcefield_path, list_forcefields

# Same as above
path = get_forcefield_path("oplsaa.xml")
forcefields = list_forcefields()
```

### Integration with IO Modules

The data module is automatically integrated with IO modules. For example, when loading a force field, you can use just the filename:

```python
from molpy.io.forcefield.xml import read_xml_forcefield

# This will automatically look for the file in molpy/data/forcefield/
ff = read_xml_forcefield("oplsaa.xml")
```

## API Reference

### `get_path(relative_path: str | Path) -> Path`

Get the absolute path to a data file.

**Parameters:**
- `relative_path`: Relative path to the data file (e.g., "forcefield/oplsaa.xml")

**Returns:**
- `Path` object pointing to the data file

**Raises:**
- `FileNotFoundError`: If the file does not exist

### `list_files(subdirectory: str | Path = "", exclude_python: bool = True) -> Iterator[str]`

List all files in a data subdirectory.

**Parameters:**
- `subdirectory`: Subdirectory to list (e.g., "forcefield"), empty string for root
- `exclude_python`: If True, exclude Python files (__init__.py, *.py, etc.)

**Yields:**
- Relative paths to files in the subdirectory

### `exists(relative_path: str | Path) -> bool`

Check if a data file exists.

**Parameters:**
- `relative_path`: Relative path to the data file

**Returns:**
- `True` if the file exists, `False` otherwise

### `get_forcefield_path(filename: str) -> Path`

Get the path to a force field file.

**Parameters:**
- `filename`: Name of the force field file (e.g., "oplsaa.xml")

**Returns:**
- `Path` object pointing to the force field file

**Raises:**
- `FileNotFoundError`: If the file does not exist

### `list_forcefields() -> list[str]`

List all available force field files.

**Returns:**
- List of force field filenames

## Adding New Data Files

To add new data files:

1. Place the file in the appropriate subdirectory (e.g., `data/forcefield/`)
2. The file will be automatically available through the data module
3. No code changes are required - just use `get_path()` or the appropriate convenience function

## Implementation Details

The data module uses `importlib.resources` (Python 3.9+) for reliable package data access. For older Python versions, it falls back to a file-based approach using `__file__`.

The module is designed to work with both:
- **Installed packages**: When MolPy is installed as a package
- **Development mode**: When MolPy is used from source

## Examples

### Example 1: Loading a Force Field

```python
from molpy.data import get_forcefield_path
from molpy.io.forcefield.xml import read_xml_forcefield

# Get the path to the force field file
path = get_forcefield_path("oplsaa.xml")

# Load the force field
ff = read_xml_forcefield(path)
```

### Example 2: Listing Available Data Files

```python
from molpy.data import list_files

# List all files in the forcefield directory
for file in list_files("forcefield"):
    print(file)
```

### Example 3: Checking File Existence

```python
from molpy.data import exists

# Check if a file exists before using it
if exists("forcefield/oplsaa.xml"):
    # Use the file
    pass
```
