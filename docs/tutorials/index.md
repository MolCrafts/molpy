# Concepts

MolPy employs distinct representations at each stage of a modeling workflow. This section explains that design from the editable molecular graph to the numerical frame representation, and covers the auxiliary structures — periodic geometry, force-field parameters, time-ordered sequences, file-format boundaries, packaged workflows, and engine integration — that complete the data model.

These pages are self-contained and may be read in any order by readers already familiar with one or more concepts. New users are advised to proceed from the first page to the last.


## Representational Hierarchy

The diagram below illustrates the standard data flow through a MolPy pipeline. Each node represents a core data structure; each edge represents an explicit transformation.

```text
                    ┌─────────────────────────┐
  SMILES / file     │  Atomistic              │
  ────parser────>   │  (editable molecular    │
                    │   graph: atoms + bonds)  │
                    └───────────┬─────────────┘
                                │
                  typifier + ForceField
                                │
                    ┌───────────▼─────────────┐
                    │  Typed Atomistic         │
                    │  (atoms carry type,      │
                    │   charge, ff parameters) │
                    └───────────┬─────────────┘
                                │
                          .to_frame()
                                │
                    ┌───────────▼─────────────┐
                    │  Frame                   │
                    │  (Block tables +         │
                    │   Box + metadata)        │
                    └───────────┬─────────────┘
                                │
                          io.write_*
                                │
                    ┌───────────▼─────────────┐
                    │  LAMMPS / GROMACS /      │
                    │  PDB / HDF5 files        │
                    └─────────────────────────┘
```

**`Atomistic`** is the primary editing surface. Atom addition and removal, bond formation, reaction execution, and polymer assembly all operate on this structure.

**`Frame`** is the primary numerical surface. Vectorized distance computation, file I/O, and engine export all operate on `Frame` objects and their constituent `Block` tables.

**`ForceField`** is an independent data structure that travels alongside a molecular system. Force field parameters are neither embedded in atoms nor derived implicitly; they are stored in a queryable typed dictionary.

**`Box`** specifies the periodic simulation cell and attaches to a `Frame` as a first-class attribute, not as metadata.

**`Trajectory`** is a time-ordered sequence of `Frame` objects providing lazy access patterns for large datasets.


## Contents

- [Atomistic and Topology](01_atomistic_and_topology.md) — the editable molecular graph and its derived connectivity relations
- [Block and Frame](02_block_and_frame.md) — columnar data tables and system-level numerical snapshots
- [Box and Periodicity](03_box_and_periodicity.md) — simulation cell geometry and minimum-image distance conventions
- [Force Field](04_force_field.md) — the force field data model, potential kernels, and multi-format parameter export
- [Trajectory](05_trajectory.md) — time-ordered frame sequences with lazy loading
- [Selector](06_selector.md) — composable, predicate-based atom filters over `Block` columns
- [Wrapper and Adapter](07_wrapper_and_adapter.md) — subprocess execution boundaries and in-memory representation bridges
- [I/O](io.md) — reading, writing, and extending molecular, trajectory, and force-field formats
- [Tool Layer](tools.md) — packaged multi-step workflows built on top of MolPy's lower-level modules
- [Engine](engine.md) — generation and execution of MD engine input files for LAMMPS, CP2K, and OpenMM
