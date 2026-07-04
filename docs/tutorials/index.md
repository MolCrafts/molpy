# Tutorials

Everything you need to learn MolPy, in order: get it running first, then
understand the data model it rests on. When you have a concrete task in hand
instead — build, typify, pack, export — go straight to
[Guides](../user-guide/index.md), which assume these tutorials.

## Get running

1. **[Installation](../getting-started/installation.md)** — install the package and verify it imports. *~2 min*
2. **[Quickstart](../getting-started/quickstart.md)** — the whole pipeline in six lines, then a TIP3P water box built with full control. *~10 min*
3. **[Example Gallery](../getting-started/examples.md)** — copy-paste workflows: small molecules, packed boxes, polymers, virtual sites.
4. **[FAQ](../getting-started/faq.md)** — why MolPy exists, how it relates to RDKit / ASE / mBuild, and when another tool is the better choice.

If MolPy is installed, this runs as-is — no optional dependencies, not even RDKit:

```python
import molpy as mp

water = mp.Atomistic(name="water")
o  = water.def_atom(element="O", x=0.000, y=0.000, z=0.000)
h1 = water.def_atom(element="H", x=0.957, y=0.000, z=0.000)
h2 = water.def_atom(element="H", x=-0.239, y=0.927, z=0.000)
water.def_bond(o, h1)
water.def_bond(o, h2)

frame = water.to_frame()
mp.io.write_pdb("water.pdb", frame)
print(f"Wrote {frame['atoms'].nrows} atoms to water.pdb")
```

`Wrote 3 atoms to water.pdb` means you are ready — and those few lines already
crossed MolPy's one essential boundary: **edit chemistry on a graph
(`Atomistic`), compute and export from arrays (`Frame`)**. The chapters below
explain that split and everything built on it.

## The data model

MolPy rests on three ideas: **edit on graphs, compute and export on arrays,
keep parameters apart**. These chapters explain the data structures behind
them — what each object *is*, why it exists, and where its boundaries lie.
Read them in order once; afterwards each chapter stands alone as reference.

- **Identity vs data.** Entities (atoms, links) are unique identities; bulk numbers live in columnar arrays. Two atoms with identical properties are still different atoms — they can take part in different bonds, selections, and edits.
- **Graph → arrays.** Build and edit as a graph (`Atomistic`); compute and export from arrays (`Frame`). The conversion is explicit (`Atomistic.to_frame()`).
- **Derived topology.** Angles and dihedrals are *derived* from bonds on demand, not stored and hand-maintained — no stale caches when the bond graph changes.
- **Parameters are separate.** Force-field typing is independent of structure, so a system can be rebuilt and re-typed deterministically.

The typical flow:

> `Atomistic` (edit) → derive topology → `Frame` (arrays) → I/O → simulate → analyze

### Representational hierarchy

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

**`Atomistic`** is the primary editing surface. Atom addition and removal, bond formation, reaction execution, and structure assembly all operate on this representation.

**`Frame`** is the primary numerical surface. Vectorized distance computation, file I/O, and engine export all operate on `Frame` objects and their constituent `Block` tables.

**`ForceField`** is an independent data structure that travels alongside a molecular system. Force field parameters are neither embedded in atoms nor derived implicitly; they are stored in a queryable typed dictionary.

**`Box`** specifies the periodic simulation cell and attaches to a `Frame` as a first-class attribute, not as metadata.

**`Trajectory`** is a time-ordered sequence of `Frame` objects providing lazy access patterns for large datasets.

### Chapter map

| Layer | What it is | In depth |
|-------|-----------|----------|
| **Entity & Link** — `Atom`, `Bond`, `Angle`, `Dihedral` | Identity-first graph model for building and editing | [Atomistic and Topology](01_atomistic_and_topology.md) |
| **Topology** | Angles/dihedrals and k-hop queries *derived* from the bond graph — there is no standalone class; use `get_topo()` / `get_topo_neighbors()` / `get_topo_distances()` | [Atomistic and Topology](01_atomistic_and_topology.md) |
| **Block & Frame** | Columnar tables (`atoms`, `bonds`, …) plus box and metadata — the exchange object that writers and compute operate on | [Block and Frame](02_block_and_frame.md) |
| **Box** | Simulation cell + periodic boundaries (wrapping, minimum-image distances) | [Box and Periodicity](03_box_and_periodicity.md) |
| **ForceField & Typifier** | A parameter catalog (styles + type tables) and the rule engine that assigns types onto a structure | [Force Field](04_force_field.md) |
| **Trajectory** | An ordered sequence of `Frame` objects | [Trajectory](05_trajectory.md) |
| **Selector** | Composable, predicate-based atom filters over `Block` columns | [Selector](06_selector.md) |
| **Wrapper & Adapter** | Subprocess execution boundaries and in-memory representation bridges to external tools | [Wrapper and Adapter](07_wrapper_and_adapter.md) |
| **CoarseGrain** | `Bead` / `CGBond` graph for coarse-grained models | [Coarse-Grained Structure](08_coarsegrain.md) |
| **Units** | Unit-system presets and explicit quantity conversion | [Units](09_units.md) |

## Appendix

- [Naming Conventions](naming-conventions.md) — canonical field names and topology-key rules used across the data model
- [Glossary](glossary.md) — concise definitions for the core data structures and modules

If you remember one sentence: **edit on graphs, compute and export on arrays.**
