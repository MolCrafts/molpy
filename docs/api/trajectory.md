# Trajectory API Reference

The `molpy.core.trajectory` module provides the `Trajectory` class for handling sequences of molecular data frames.

## Overview

The `Trajectory` class manages time-series molecular data:
- **Frame sequences**: Ordered collection of Frame objects
- **Time-based indexing**: Access frames by time or index
- **Batch operations**: Apply operations across all frames
- **Analysis tools**: Statistical and geometric analysis methods

## Complete API Documentation

::: molpy.core.trajectory
    options:
      show_source: true
      show_root_heading: false
      heading_level: 2

## Key Capabilities

### Frame Management
```python
import molpy as mp

# Create frames
frame1 = mp.Frame({'x': [1.0, 2.0], 'y': [3.0, 4.0]})
frame2 = mp.Frame({'x': [1.1, 2.1], 'y': [3.1, 4.1]})

# Create trajectory
traj = mp.Trajectory([frame1, frame2])

# Access frames
first_frame = traj[0]
last_frame = traj[-1]
```

### Analysis Operations
```python
# Iterate through trajectory
for frame in traj:
    print(f"Frame with {len(frame)} atoms")

# Batch operations
mean_coords = traj.mean()  # Average coordinates
std_coords = traj.std()   # Standard deviations
```

### Integration with Structures
```python
# Convert structures to trajectory
structures = [struct1, struct2, struct3]
frames = [s.to_frame() for s in structures]
trajectory = mp.Trajectory(frames)
```

For complete tutorials, see [Frame Tutorial](../tutorials/frame.md).
