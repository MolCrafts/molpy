# Architecture Overview

MolPy is a layered toolkit with explicit data flow and minimal magic. This page is the map that every extension guide assumes: which module owns what, how the three class hierarchies of the data model fit together, and where the boundaries between Python and the molrs Rust backend run. Read it once before touching anything under [Extending MolPy](extending-compute.md).

## Module responsibilities

Each package has one job, and molpy is a thin layer over molrs: every native
name is the molrs object, re-exported by identity, and molpy keeps no parallel
IR, I/O, geometry, units or regions. Every molrs subsystem is mirrored by a
molpy module — `core` for molrs's core (`molrs.core`: the stores, the graph
hierarchy, the box, regions, neighbour search and units in one flat module),
the rest under their molrs names — and the root holds those modules,
the core data classes a user handles directly (promoted from `core` as the
same objects) and the version metadata.

| Package | Purpose |
|---------|---------|
| `molpy` (root) | The subsystems below; the promoted core data classes (`Frame`, `Block`, `Trajectory`, `Box`, `MolGraph`, `Atomistic`, `CoarseGrain`, the entity classes, `Element`, `Topology`); `version` / `release_date` |
| `core` | Mirrors `molrs.core` — `Frame`, `Block`, `Trajectory`, the graph hierarchy (`MolGraph`, `Atomistic`, `CoarseGrain`), `Box`, regions, neighbour search and units, flat, with the vocabularies `keys`, `schema` and `constants` as submodules; plus molpy's column selectors and `TrajectorySplitter` with its strategies |
| `perceive`, `optimize`, `conformer`, `stream` | Mirror the molrs subsystems of the same names: perception, `Lbfgs`, conformers, frame streaming |
| `ff` | Mirrors `molrs.ff`: `forcefield` (the `ForceField` data model), `potential` (compiler and kernels), `typifier` (+ molpy's `AntechamberTypifier` / `TleapTypifier`), `charge`, `ir`, `params`, `clpol_scaling` |
| `io` | Mirrors `molrs.io`: every file reader and writer (structure, trajectory and force-field files, SMILES, `*.mrec` records, frame bytes, LAMMPS logs) as one door per format, `read_<fmt>[_<what>]` / `write_<fmt>[_<what>]`, and one submodule per format that owns classes (`smiles`, `cgsmiles`, `lammps`, `mrec`, `pdb`, `xyz`, `gro`, `dcd`, `trr`, `xtc`); molpy's metric readers, published in the `molcrafts.metric_readers` entry-point group, are `io.lammps.LammpsLogMetricReader`, `io.mrec.MrecMetricReader` and `io.mlp_jsonl.MlpJsonlMetricReader` |
| `builder` | Mirrors `molrs.builder` (assembly, `Coarsener`, graphene and nanotubes); plus polymer planning, crystals, virtual sites, `PackingTemplate`, `AmberPolymerBuilder` |
| `compute`, `signal`, `md`, `op` | Mirror `molrs.compute`, `molrs.signal`, `molrs.md`, `molrs.op` |
| `engine` | External engines: one `generate_inputs` each for LAMMPS, GROMACS and OpenMM, `run`, `Script` |
| `wrapper` | Subprocess boundaries to external CLI tools (antechamber, parmchk2, prepgen, tleap, sander) |
| `adapter` | Optional in-memory bridge (RDKit worked example) |
| `resources` | Bundled package files: force-field files |

The modules behind `engine`, `wrapper`, `adapter`, `builder` and molpy's additions to `core` are private
(`molpy.engine._lammps`, …): each name has one public path, its subpackage
(`mp.engine.LammpsEngine`).

`compute`, `io`, and `engine` operate on the tabular layer (`Frame`/`Block`); `builder` and `ff.typifier` operate on the graph layer (`Atomistic`). `wrapper` and `adapter` sit at the outer edge and never leak external types into the data model.

## The graph layer: live handle views over molrs

The editable graph has one implementation: the molrs world. Python exposes
three cooperating surfaces:

1. **Node refs** — `Atom`, `Bead`, and virtual-site variants are dict-like live
   views identified by a stable native handle.
2. **Relation refs** — `Bond`, `Angle`, `Dihedral`, `Improper`, and `CgBond`
   resolve endpoint handles in the same world.
3. **Worlds** — `Atomistic` and `CoarseGrain` own nodes, relations, columns, and
   graph algorithms. Their `.atoms`, `.bonds`, and related properties are lazy
   handle collections: integer access interns a view (weak-interned per handle);
   string field access (`atoms["x"]`) reads the dense component store without
   materializing every view. There is no mirrored Python property bag.

There is no `Struct`/`TypeBucket` registration layer. Adding a new stored node
or relation kind changes the molrs schema and bindings; it is not a Python
subclassing hook. See [Extending the Data Model](extending-core.md).

## The tabular layer: Block and Frame run on molrs

`Frame` and `Block` belong exclusively to the [molrs](https://github.com/MolCrafts/molrs) Rust column store. Use them from the root (`mp.Frame`, `mp.Block`, also `molpy.core`'s); they are identity re-exports of the molrs types. Columns are typed (float / int / bool / str) and exposed as zero-copy NumPy views; a non-representable column is rejected fail-fast at write. `molcrafts-molrs` is a hard runtime dependency: there is no pure-Python fallback.

The graph → arrays conversion is explicit: `Atomistic.to_frame()` delegates to the molrs world's native `to_frame()`. The box is a first-class attribute (`frame.box`), never metadata. The [molrs Backend](molrs-backend.md) page covers how neighbor lists, `Rdf`, and the analysis catalog surface from Rust.

## Force field: parameters apart, kernels in Rust

`ForceField` is an independent, queryable data structure — parameters are neither embedded in atoms nor derived implicitly. The model has three layers: **Style** (functional form), **ForceFieldType** (parameter set for a type key), and **Potential** (evaluatable kernel). All energy/force kernels live in molrs; a style is named by `ff.def_style(category, name)`, and evaluation always goes through `PotentialCompiler(ff).compile(frame)`. The force-field IR is a protocol, so adding a functional form is a registration from Python (subclass `mp.ff.style_registry.StyleDeclaration`, with an expression or a Python kernel), with nothing rebuilt — the recipe is in [Extending the Force Field](extending-forcefield.md).

## One column vocabulary

Canonical field names (`charge`, not `q`; `mol_id`, not `mol`) are used everywhere inside MolPy. The vocabulary is molrs's: `mp.core.keys` is `molrs.core.keys` (`mp.core.keys.CHARGE.key == "charge"`) and `mp.core.schema` gives each column's dtype. Format-specific names exist only inside the native readers and writers, which map them at the boundary, so no molpy code translates column names. The full canonical-name catalog is in the [Naming Conventions](../tutorials/naming-conventions.md) appendix; the extension recipe is in [Adding an I/O Format](extending-io.md).

## The mutation contract

The core data-model API mutates in place: `def_atom`, `def_bond`, `generate_topology`, `translate`, `rotate`, `scale`, `merge` all modify the structure they are called on. The rigid-body transforms return `self` for chaining, factories return the created entity, and `generate_topology` / `merge` return what they added. `.copy()` is the explicit opt-in for an independent deep copy. Higher-level helpers in `builder` and `op` follow the opposite convention: they must not mutate caller-owned structures unexpectedly — copy first, or build and return a new structure.

## Performance model of assembly

Assembly is one native call and never types anything:

- **Native build** — `Assembler.assemble` runs in molrs and releases the GIL. It copies
  one template per site, poses each copy once (placer, then optional orienter), and links
  every site bond through its two ports. There is no per-bond Python loop and no
  reaction matching: which ports join is decided from the site graph, not searched for
  in the growing world.
- **No retyping during the build** — the world comes back with atoms, bonds and ports
  only. Typing is a separate, whole-graph step the caller runs once afterwards.
- **Explicit topology** — angles and dihedrals are generated once, when needed, by
  `generate_topology(gen_angle=True, gen_dihedral=True)`; a large system stays
  atoms-only until an MD writer needs topology.

## Where extension happens

| I want to add… | Layer | Guide |
|---|---|---|
| an analysis operation | plug-in interface | [Adding a Compute Operation](extending-compute.md) |
| a file format | plug-in interface | [Adding an I/O Format](extending-io.md) |
| an external tool integration | plug-in interface | [Adding a Wrapper or Adapter](extending-integration.md) |
| an entity/link/struct type | core internals — open an issue first | [Extending the Data Model](extending-core.md) |
| a graph typifier / force-field overlay | typifier internals — open an issue first | [Extending Typifiers](extending-typifiers.md) |
| an interaction style / category | plug-in interface (`mp.ff.style_registry.StyleDeclaration`) | [Extending the Force Field](extending-forcefield.md) |
