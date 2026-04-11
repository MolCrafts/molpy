# Trajectory

This page explains how MolPy represents trajectories as ordered sequences of frames, and how lazy access and transformation are handled at that level.

## One frame is rarely enough

A single `Frame` captures the state of a system at one instant. Simulation and analysis almost always involve many such states ordered in time. Storing them as a plain Python list would work for small datasets, but it hides two concerns that become important at scale: memory management and lazy access.

**A `Trajectory` is an ordered sequence of `Frame` objects that supports lazy evaluation, so system meaning survives even when the data becomes large.**

The key idea is continuity with `Frame`. Each element of a trajectory is still one frame — named blocks, metadata, and optionally a box. Time does not replace the snapshot model. It stacks snapshots in order.


## Building a trajectory from a list

The simplest trajectory comes from an in-memory list of frames. This supports random access, `len`, and slicing.

```python
import molpy as mp

frames = []
for i in range(5):
    f = mp.Frame()
    f["atoms"] = mp.Block({"x": [float(i)], "y": [0.0], "z": [0.0]})
    f.metadata["time"] = i * 10.0
    frames.append(f)

traj = mp.Trajectory(frames)
print(len(traj))             # 5
print(traj.has_length())     # True
print(traj[0]["atoms"]["x"]) # [0.]
```


## Generator-based trajectories stay lazy

For large or streaming data, pass a generator instead of a list. The trajectory yields frames on demand without loading everything into memory. The trade-off: you lose `len` and indexing until the generator is materialized.

```python
def make_frames(n):
    for i in range(n):
        f = mp.Frame()
        f["atoms"] = mp.Block({"x": [float(i)], "y": [0.0], "z": [0.0]})
        f.metadata["time"] = i * 0.5
        yield f

lazy_traj = mp.Trajectory(make_frames(1000))
print(lazy_traj.has_length())   # False

# Iterate without materializing all frames at once
for frame in lazy_traj:
    if frame.metadata["time"] > 2.0:
        break
```

Generators are consumed by iteration. If you need to read the same data multiple times, materialize a subset or create a fresh generator each time.


## Slicing and indexing

For list-backed trajectories, standard Python indexing and slicing work as expected. Indexing returns a `Frame`; slicing returns a new `Trajectory`.

```python
first_two = traj[:2]
print(len(first_two))   # 2

strided = traj[::2]
print(len(strided))     # 3

last = traj[-1]
print(last.metadata["time"])   # 40.0
```

Slicing with a stride (`traj[::n]`) is a convenient way to downsample for quick inspection.


## Lazy transforms with map

`map` applies a function to each frame and returns a new trajectory. The transformation is lazy — the function runs only when a frame is accessed, not when `map` is called. This means you can chain several transforms without paying for all of them upfront.

```python
def shift_x(frame):
    new = mp.Frame()
    x = frame["atoms"]["x"]
    new["atoms"] = mp.Block({
        "x": x + 10.0,
        "y": frame["atoms"]["y"],
        "z": frame["atoms"]["z"],
    })
    new.metadata = frame.metadata.copy()
    return new

shifted = traj.map(shift_x)
```

Because `map` returns a generator-based trajectory, you need to iterate or materialize to see the results.

```python
shifted_list = list(shifted)
print(shifted_list[0]["atoms"]["x"])   # [10.]
print(traj[0]["atoms"]["x"])           # [0.] — original unchanged
```


## When to use Trajectory

Use `Trajectory` when time is part of the scientific question — following an observable over many snapshots, computing time correlations, or iterating through an I/O stream. If you only need a single state, `Frame` remains the right abstraction.

The trajectory does not invent a new kind of system state. It keeps frame meaning intact while adding temporal ordering. That is the entire point: one structure, many times.

See also: [Block and Frame](02_block_and_frame.md), [Box and Periodicity](03_box_and_periodicity.md).
