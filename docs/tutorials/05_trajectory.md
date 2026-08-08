# Trajectory

Analysis almost never cares about a single snapshot. How do you hold *many*
frames in time order without inventing a second data model?

**A `Trajectory` is an eager, ordered sequence of `Frame` objects.** Each element
is still one frame — blocks, metadata, optional box. Time stacks snapshots; it
does not replace them.

What it is **not**: a lazy file cursor. Seekable readers live under
`molpy.io` / molrs trajectory readers; construct a `Trajectory` when you want
an in-memory sequence with `len`, indexing, and slicing.

## Building a trajectory from a list

Pass a list (or any iterable that is materialised on construction):

```python
import molpy as mp

frames = []
for i in range(5):
    f = mp.Frame()
    f["atoms"] = mp.Block({"x": [float(i)], "y": [0.0], "z": [0.0]})
    f.meta = {"time": mp.MetaValue("f64", i * 10.0)}
    frames.append(f)

traj = mp.Trajectory(frames)
print(len(traj))  # 5
print(traj[0]["atoms"]["x"])  # [0.]
```


## Iterables are materialized

The constructor accepts any iterable, but materializes it immediately into the native container. Use `molpy.io.read_lammps_trajectory` or `molpy.io.read_xyz_trajectory` when data must remain lazy and seekable on disk.

```python
def make_frames(n):
    for i in range(n):
        f = mp.Frame()
        f["atoms"] = mp.Block({"x": [float(i)], "y": [0.0], "z": [0.0]})
        f.meta = {"time": mp.MetaValue("f64", i * 0.5)}
        yield f


traj_from_iterable = mp.Trajectory(make_frames(1000))
print(len(traj_from_iterable))  # 1000
```

The generator is consumed during construction. File readers avoid that eager materialization.


## Slicing and indexing

For list-backed trajectories, standard Python indexing and slicing work as expected. Indexing returns a `Frame`; slicing returns a new `Trajectory`.

```python
first_two = traj[:2]
print(len(first_two))  # 2

strided = traj[::2]
print(len(strided))  # 3

last = traj[-1]
print(last.meta["time"].value)  # 40.0
```

Slicing with a stride (`traj[::n]`) is a convenient way to downsample for quick inspection.


## Transforms with map

`map` applies a function to every frame immediately and returns a new trajectory. The original frames are unchanged.

```python
def shift_x(frame):
    new = mp.Frame()
    x = frame["atoms"]["x"]
    new["atoms"] = mp.Block(
        {
            "x": x + 10.0,
            "y": frame["atoms"]["y"],
            "z": frame["atoms"]["z"],
        }
    )
    new.meta = frame.meta
    return new


shifted = traj.map(shift_x)
```

```python
shifted_list = list(shifted)
print(shifted_list[0]["atoms"]["x"])  # [10.]
print(traj[0]["atoms"]["x"])  # [0.] — original unchanged
```


## When to use Trajectory

Use `Trajectory` when time is part of the scientific question — following an observable over many snapshots, computing time correlations, or iterating through an I/O stream. If you only need a single state, `Frame` remains the right abstraction.

The trajectory does not invent a new kind of system state. It keeps frame meaning intact while adding temporal ordering. That is the entire point: one structure, many times.

See also: [Block and Frame](02_block_and_frame.md), [Box and Periodicity](03_box_and_periodicity.md).
