# Core

Foundational data structures for molecular systems. All available via `import molpy as mp`.

## Quick reference

| Symbol | Summary | Preferred for | Avoid when |
|--------|---------|---------------|------------|
| `Atomistic` | Editable molecular graph (atoms + bonds) | Building, editing, reacting on chemistry | Array-backed analysis or export |
| `Block` | Columnar table: column names → NumPy arrays | Tabular data, vectorized computation | Graph-level chemical editing |
| `Frame` | Named collection of Blocks + metadata | System snapshots, file I/O | Editing individual atoms |
| `Box` | Periodic simulation cell (3×3 matrix + PBC) | Wrapping, minimum-image distances | Non-periodic systems |
| `Trajectory` | Ordered sequence of Frames (eager or lazy) | Time-series analysis, streaming I/O | Single-snapshot work |
| `Topology` | igraph-based bond graph → derived angles/dihedrals | Graph algorithms, connectivity queries | Storing atom properties |
| `CoarseGrain` | CG molecular graph (beads + CG bonds) | Coarse-grained modelling, CG/AA conversion | All-atom work (use `Atomistic`) |
| `Config` | Thread-safe global configuration singleton | Logging level, thread count settings | Per-run overrides (use `Config.temporary`) |
| `AtomisticForcefield` | Force field container (styles → types → potentials) | Defining parameters before execution | Direct numerical computation |

## Canonical examples

```python
import molpy as mp

# Atomistic: editable molecular graph
mol = mp.Atomistic(name="water")
o = mol.def_atom(symbol="O", x=0.0, y=0.0, z=0.0)
h = mol.def_atom(symbol="H", x=0.957, y=0.0, z=0.0)
mol.def_bond(o, h)

# Block + Frame: tabular snapshot
frame = mp.Frame(blocks={
    "atoms": {"element": ["O", "H"], "x": [0.0, 0.957]},
}, timestep=0)

# Box: periodic cell
box = mp.Box.cubic(20.0)
wrapped = box.wrap(coords)
d = box.dist(r1, r2)  # minimum-image distance

# ForceField: parameter data
ff = mp.AtomisticForcefield(name="demo", units="real")
style = ff.def_atomstyle("full")
ct = style.def_type("CT", mass=12.011)
```

## Related

- [Concepts: Atomistic](../tutorials/01_atomistic_and_topology.md)
- [Concepts: Block and Frame](../tutorials/02_block_and_frame.md)
- [Concepts: Box](../tutorials/03_box_and_periodicity.md)
- [Concepts: Force Field](../tutorials/04_force_field.md)

---

## Full API

### Atomistic

::: molpy.core.atomistic

### Box

::: molpy.core.box

### Forcefield

::: molpy.core.forcefield

### Frame

::: molpy.core.frame

### Topology

::: molpy.core.topology

### Trajectory

::: molpy.core.trajectory

### Coarse-Grain

::: molpy.core.cg

### Config

::: molpy.core.config

### Script

::: molpy.core.script
