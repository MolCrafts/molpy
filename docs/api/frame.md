# Frame API Reference

The `molpy.core.frame` module provides the `Frame` class for high-performance molecular data processing using xarray Datasets.

## Overview

The `Frame` class is the core data structure for handling molecular properties and coordinates:
- Built on **xarray Datasets** for robust mixed-datatype support
- Supports **concatenation** with dtype consistency checking
- Handles **empty frames** gracefully
- Integrates seamlessly with AtomicStructure via `to_frame()`

## Complete API Documentation

::: molpy.core.frame
    options:
      show_source: true
      show_root_heading: false
      heading_level: 2

## Key Features

### Dataset-Based Architecture
Each field in a Frame is stored as an xarray Dataset, providing:
- Mixed datatype support (float, int, string, bool)
- Labeled dimensions and coordinates
- Built-in metadata handling
- Efficient memory usage

### Concatenation with Type Safety
```python
# Frames with consistent dtypes concatenate successfully
frame1 = Frame({'x': [1.0, 2.0], 'name': ['A', 'B']})
frame2 = Frame({'x': [3.0, 4.0], 'name': ['C', 'D']})
combined = Frame.concat([frame1, frame2])  # Works

# Inconsistent dtypes raise errors
frame3 = Frame({'x': [1, 2]})  # int instead of float
Frame.concat([frame1, frame3])  # Raises ValueError
```

### Integration with Structures
```python
# Convert AtomicStructure to Frame
structure = mp.AtomicStructure()
structure.add_atom(mp.Atom(symbol='C', x=1.0, y=2.0, z=3.0))
frame = structure.to_frame()  # Automatic conversion
```

For complete tutorials, see [Frame Tutorial](../tutorials/frame.md).
