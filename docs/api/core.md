# Core

Foundational data structures for molecular systems. All available via
`import molpy as mp`.

## Quick reference

| Symbol | Summary | Preferred for | Avoid when |
|--------|---------|---------------|------------|
| `Atomistic` | Editable molecular graph (atoms + bonds) | Building, editing, reacting on chemistry | Array-backed analysis or export |
| `Block` | Columnar table: column names → NumPy arrays | Tabular data, vectorized computation | Graph-level chemical editing |
| `Frame` | Named Blocks + `box` + dict-like `meta` | System snapshots, file I/O | Editing individual atoms |
| `Box` | Periodic simulation cell (3×3 matrix + PBC) | Wrapping, minimum-image distances | Non-periodic systems |
| `Trajectory` | Ordered sequence of Frames (eager or lazy) | Time-series analysis, streaming I/O | Single-snapshot work |
| `CoarseGrain` | CG molecular graph (beads + CG bonds) | Coarse-grained modelling; mirrors `Atomistic` | All-atom work (use `Atomistic`) |
| `Config` | Thread-safe global configuration singleton | Logging level | Per-run overrides (use `Config.temporary`) |
| `ForceField` | Force field container (styles → types → potentials) | Defining parameters before execution | Direct numerical computation |
| `NodeRef` / `RelationRef` / `Refs` | Live handles onto graph nodes / relations and collections of them (an `Atom` is a node view, a `Bond` a relation view) | Code generic over node / relation kinds | Everyday atom / bond editing |
| `Region` | Geometric region (box, sphere, boolean combinations) | Spatial selection, packing constraints | Non-geometric masks (use a `Selector`) |
| `UnitSystem` | Unit registry with named presets (`real`, `metal`, `openmm`, …) and reduced LJ units | Unit conversions and custom presets | Unit-agnostic array math |

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
ff = mp.ForceField(name="demo", units="real")
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

::: molpy.core.box

### Forcefield

::: molpy.ForceField

::: molpy.Style

::: molpy.Type

::: molpy.PotentialCompiler

### Frame and Block

Re-exported from the native core — `mp.Frame is molrs.Frame`:

::: molpy.Frame

::: molpy.Block

### Trajectory

::: molpy.Trajectory

### Coarse-Grain

::: molpy.CoarseGrain

### Config

::: molpy.core.config

### Script

::: molpy.core.script

### Node and relation handles

::: molpy.NodeRef

::: molpy.RelationRef

::: molpy.Refs

### Selector

::: molpy.core.selector

### Region

::: molpy.core.region

### Units

::: molpy.core.unit
