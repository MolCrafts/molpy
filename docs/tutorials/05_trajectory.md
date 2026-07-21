# Trajectory

A `Trajectory` stacks an in-memory sequence of frames in time order. Lazy, seekable file access is provided by molrs trajectory readers.

## One frame is rarely enough

A single `Frame` captures the state of a system at one instant. Simulation and analysis almost always involve many such states ordered in time. Storing them as a plain Python list would work for small datasets, but it hides two concerns that become important at scale: memory management and lazy access.

**A `Trajectory` is an eager, ordered sequence of `Frame` objects.**

The key idea is continuity with `Frame`. Each element of a trajectory is still one frame — named blocks, exact-dtype metadata, and optionally a box. Time does not replace the snapshot model. It stacks snapshots in order.


## Building a trajectory from a list

The simplest trajectory comes from an in-memory list of frames. This supports random access, `len`, and slicing.

```python
import molpy as mp

frames = []
for i in range(5):
    f = mp.Frame()
    f["atoms"] = mp.Block({"x": [float(i)], "y": [0.0], "z": [0.0]})
    f.meta = {"time": mp.MetaValue("f64", i * 10.0)}
    frames.append(f)

traj = mp.Trajectory(frames)
print(len(traj))             # 5
print(traj[0]["atoms"]["x"]) # [0.]
```


## Iterables are materialized

The constructor accepts any iterable, but materializes it immediately into the native container. Use `molrs.read_lammps_trajectory` or `molrs.read_xyz_trajectory` when data must remain lazy and seekable on disk.

```python
def make_frames(n):
    for i in range(n):
        f = mp.Frame()
        f["atoms"] = mp.Block({"x": [float(i)], "y": [0.0], "z": [0.0]})
        f.meta = {"time": mp.MetaValue("f64", i * 0.5)}
        yield f

traj_from_iterable = mp.Trajectory(make_frames(1000))
print(len(traj_from_iterable))   # 1000
```

The generator is consumed during construction. File readers avoid that eager materialization.


## Slicing and indexing

For list-backed trajectories, standard Python indexing and slicing work as expected. Indexing returns a `Frame`; slicing returns a new `Trajectory`.

```python
first_two = traj[:2]
print(len(first_two))   # 2

strided = traj[::2]
print(len(strided))     # 3

last = traj[-1]
print(last.meta["time"].value)   # 40.0
```

Slicing with a stride (`traj[::n]`) is a convenient way to downsample for quick inspection.


## Transforms with map

`map` applies a function to every frame immediately and returns a new trajectory. The original frames are unchanged.

```python
def shift_x(frame):
    new = mp.Frame()
    x = frame["atoms"]["x"]
    new["atoms"] = mp.Block({
        "x": x + 10.0,
        "y": frame["atoms"]["y"],
        "z": frame["atoms"]["z"],
    })
    new.meta = frame.meta
    return new

shifted = traj.map(shift_x)
```

```python
shifted_list = list(shifted)
print(shifted_list[0]["atoms"]["x"])   # [10.]
print(traj[0]["atoms"]["x"])           # [0.] — original unchanged
```


## When to use Trajectory

Use `Trajectory` when time is part of the scientific question — following an observable over many snapshots, computing time correlations, or iterating through an I/O stream. If you only need a single state, `Frame` remains the right abstraction.

The trajectory does not invent a new kind of system state. It keeps frame meaning intact while adding temporal ordering. That is the entire point: one structure, many times.

See also: [Block and Frame](02_block_and_frame.md), [Box and Periodicity](03_box_and_periodicity.md).
