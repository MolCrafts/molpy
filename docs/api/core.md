# Core

Foundational data structures for molecular systems. All available via
`import molpy as mp`. `mp.core` mirrors molrs's core, `molrs.core`, by identity, in one
module (`mp.core.Cuboid is molrs.core.Cuboid`). The data classes you handle
directly are promoted to the root as the same objects (`mp.Frame is
mp.core.Frame is molrs.core.Frame`, `mp.Box`, `mp.Atomistic`, …); everything
else is `mp.core.<Name>`. molpy's additions here are the column selectors and
the trajectory splitters, both in `mp.core` beside the `Block` and
`Trajectory` they act on. Perception (`SmartsPattern`, `perceive_rings`, `add_hydrogens`, …) is
`mp.perceive`.

## Quick reference

| Symbol | Summary | Preferred for | Avoid when |
|--------|---------|---------------|------------|
| `Atomistic` | Editable molecular graph (atoms + bonds) | Building, editing, reacting on chemistry | Array-backed analysis or export |
| `Block` | Columnar table: column names → NumPy arrays | Tabular data, vectorized computation | Graph-level chemical editing |
| `Frame` | Named Blocks + `box` + dict-like `meta` | System snapshots, file I/O | Editing individual atoms |
| `Box` | Periodic simulation cell (3×3 matrix + PBC) | Wrapping, minimum-image distances | Non-periodic systems |
| `Trajectory` | Ordered in-memory sequence of Frames with `step` / `time` labels (slicing, `map`); lazy readers are `mp.io.read_*_trajectory` | Time-series analysis | Single-snapshot work |
| `CoarseGrain` | CG molecular graph (beads + CG bonds) | Coarse-grained modelling; mirrors `Atomistic` | All-atom work (use `Atomistic`) |
| `mp.ff.forcefield.ForceField` | Force field container (styles → types → potentials) | Defining parameters before execution | Direct numerical computation |
| `NodeRef` / `RelationRef` / `Refs` | Live handles onto graph nodes / relations and collections of them (an `Atom` is a node view, a `Bond` a relation view) | Code generic over node / relation kinds | Everyday atom / bond editing |
| `Cuboid` / `Sphere` / `HalfSpace` / … / `Region` | Geometric regions (`mask(block)`, `region(block)`, `&` / `\|` / `~` into a `Region`) | Spatial selection, packing constraints | Non-geometric masks (use a `Selector`) |
| `UnitRegistry` / `UnitPreset` | Unit registry (with `k_B` and reduced LJ units) and named presets (`real`, `metal`, `openmm`, …) | Unit conversions and custom presets | Unit-agnostic array math |

## Canonical examples

```python
import numpy as np
import molpy as mp

# Atomistic: editable molecular graph (core APIs mutate in place)
mol = mp.Atomistic(name="water")
o = mol.def_atom(element="O", x=0.0, y=0.0, z=0.0)
h = mol.def_atom(element="H", x=0.957, y=0.0, z=0.0)
mol.def_bond(o, h)
mol.generate_topology(gen_angle=True) # writes angles on mol; returns counts added
# bulk reads: mol.atoms["x"] / mol.xyz — no full view materialization

# Block + Frame: tabular snapshot. `meta` is a dict of Python scalars.
frame = mp.Frame(
 blocks={"atoms": {"element": ["O", "H"], "x": [0.0, 0.957]}},
 meta={"timestep": 0},
)

# Box: periodic cell
box = mp.Box.cube(20.0)
wrapped = box.wrap(np.array([[21.0, 0.0, 0.0]]))
d = box.distances(np.array([[0.0, 0.0, 0.0]]), np.array([[19.5, 0.0, 0.0]])) # minimum-image distance

# ForceField: parameter data
ff = mp.ff.forcefield.ForceField(name="demo", units="real")
style = ff.def_style("atom", "full")
ct = style.def_type("CT", mass=12.011) # returns the AtomType handle
cc = ff.def_style("bond", "harmonic").def_type("CT-CT", ct, ct, k=536.0, r0=1.529)
```

## Related

- [Concepts: Atomistic](../tutorials/01_atomistic_and_topology.md)
- [Concepts: Block and Frame](../tutorials/02_block_and_frame.md)
- [Concepts: Box](../tutorials/03_box_and_periodicity.md)
- [Concepts: Force Field](../tutorials/04_force_field.md)

---

## Full API

### Atomistic

::: molpy.Atomistic

### Box

::: molpy.Box

### Forcefield

::: molpy.ff.forcefield.ForceField

::: molpy.ff.forcefield.Style

::: molpy.ff.forcefield.ForceFieldType

::: molpy.ff.compile.PotentialCompiler

### Frame and Block

Re-exported from the native core — `mp.Frame is molrs.core.Frame`:

::: molpy.Frame

::: molpy.Block

### Trajectory

::: molpy.Trajectory

::: molpy.core.TrajectorySplitter

::: molpy.core.SplitStrategy

::: molpy.core.FrameIntervalStrategy

::: molpy.core.TimeIntervalStrategy

::: molpy.core.CustomStrategy

### Coarse-Grain

::: molpy.CoarseGrain

### Script

::: molpy.engine.Script

### Node and relation handles

::: molpy.core.NodeRef

::: molpy.core.RelationRef

::: molpy.core.Refs

### Selector

::: molpy.core.MaskPredicate

::: molpy.core.ElementSelector

::: molpy.core.AtomTypeSelector

::: molpy.core.AtomIndexSelector

### Region

::: molpy.core.Region

::: molpy.core.Cuboid

::: molpy.core.Sphere

### Units

::: molpy.core.UnitRegistry

::: molpy.core.UnitPreset
