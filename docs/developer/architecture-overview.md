# Architecture Overview

MolPy is a layered toolkit with explicit data flow and minimal magic. This page is the map that every extension guide assumes: which module owns what, how the three class hierarchies of the data model fit together, and where the boundaries between Python and the Rust core run. Read it once before touching anything under [Extending MolPy](extending-compute.md).

## Module responsibilities

Each package has one job. The native classes are the Rust core's (`crates/molrs`),
compiled into `molpy._core`, and Python keeps no parallel IR, I/O, geometry,
units or regions. Every subsystem of the core has one molpy module of the
same name — `core` (the stores, the graph hierarchy, the box, regions,
neighbour search and units in one flat module), then `io`, `ff`, … — and the
root holds those modules, the core data classes a user handles directly
(promoted from `core` as the same objects) and the version metadata.

| Package | Purpose |
|---------|---------|
| `molpy` (root) | The subsystems below; the promoted core data classes (`Frame`, `Block`, `Trajectory`, `Box`, `MolGraph`, `Atomistic`, `CoarseGrain`, the entity classes, `Element`, `Topology`); `__version__` / `release_date` and the `version` module |
| `core` | The core data model (`molrs::core`) — `Frame`, `Block`, `Trajectory`, the graph hierarchy (`MolGraph`, `Atomistic`, `CoarseGrain`), `Box`, regions, neighbour search and units, flat, with the vocabularies `keys`, `schema` and `constants` as submodules; plus molpy's selectors and `TrajectorySplitter` with its strategies |
| `perceive`, `optimize`, `conformer`, `stream` | The core subsystems of the same names: perception, `Lbfgs`, conformers, frame streaming |
| `ff` | Force fields, one submodule per Rust owner: `forcefield` (the `ForceField` data model), `potential` (the kernels: `Potentials`, `WeightedTerms`, `PairLjCut`), `compile` (`PotentialCompiler`, `ExplicitTerms`: a force field bound to its kernels), `typifier` (GAFF natively as `AtdTypifier` + `GaffTypifier`; + molpy's AmberTools `AntechamberTypifier` / `TleapTypifier`, for frcmod parameters on top of the leaprc and antechamber's other charge methods, which share the private `molpy.tool.ambertools` pipeline with `AmberPolymerBuilder`; a native GAFF force field merges with that builder's prmtop one), `charge` (+ molpy's `Am1BccModel`: sqm's AM1 charges, corrected natively), `ir`, `style_registry` (registering styles and categories from Python), `clpol_scaling` |
| `io` | Every file reader and writer (structure, trajectory and force-field files, SMILES, `*.mrec` records, frame bytes, LAMMPS logs) as one door per format, `read_<fmt>[_<what>]` / `write_<fmt>[_<what>]`, and one submodule per format that owns classes (`smiles`, `cgsmiles`, `lammps`, `mrec`, `pdb`, `xyz`, `gro`, `dcd`, `trr`, `xtc`); molpy's metric readers, published in the `molcrafts.metric_readers` entry-point group, are `io.lammps.LammpsLogMetricReader`, `io.mrec.MrecMetricReader` and `io.mlp_jsonl.MlpJsonlMetricReader` |
| `builder` | The native builders (assembly, `Coarsener`, graphene and nanotubes); plus polymer planning, crystals, virtual sites, `PackingTemplate`, `AmberPolymerBuilder` |
| `compute`, `md`, `op` | Every analysis (one `Correlator` engine under the time correlations, the fits and spectra), in-process MD, the numeric base |
| `engine` | External engines: one `generate_inputs` each for LAMMPS, GROMACS and OpenMM, CP2K running a hand-written input, `run`, `Script` |
| `tool` | The one core every external program shares (`Tool`, `ToolError`, `EnvironmentSpec`); the AmberTools pipeline (antechamber, parmchk2, prepgen, tleap) behind the AmberTools typifiers and `AmberPolymerBuilder` lives under it, in `molpy.tool.ambertools` |
| `resources` | Bundled package files: force-field files |

The modules behind `engine`, `tool`, `builder` and molpy's additions to `core`
(`molpy.engine.lammps`, …) are implementation modules: they are not in their
subpackage's `__all__`, so each name has one public path, its subpackage
(`mp.engine.LammpsEngine`). No module name starts with `_`; `__all__`, not a
name prefix, says what is public.

`compute`, `io`, and `engine` operate on the tabular layer (`Frame`/`Block`); `builder` and `ff.typifier` operate on the graph layer (`Atomistic`). `tool` sits at the outer edge and never leaks external types into the data model.

## The graph layer: live handle views over the Rust core

The editable graph has one implementation: the Rust core's world. Python exposes
three cooperating surfaces:

1. **Node refs** — `Atom`, `Bead`, and virtual-site variants are dict-like live
   views identified by a stable native handle.
2. **Relation refs** — `Bond`, `Angle`, `Dihedral`, `Improper`, and `CgBond`
   resolve endpoint handles in the same world.
3. **Worlds** — `Atomistic` and `CoarseGrain` own nodes, relations, columns, and
   graph algorithms. Their `.atoms`, `.bonds`, and related properties are lazy
   handle collections: integer access interns a view (weak-interned per handle);
   string field access (`atoms["x"]`) reads the dense component columns without
   materializing every view; per-node values live in those columns only.

A new stored node or relation kind is declared in the core's schema and its
Python binding (`molpy/src/`). See [Extending the Data Model](extending-core.md).

## The tabular layer: Block and Frame run on the Rust core

`Frame` and `Block` belong exclusively to the Rust core's column store. Use them from the root (`mp.Frame`, `mp.Block`, also `mp.core`'s). Columns are typed (float / int / uint / bool / str) and exposed as zero-copy NumPy views; a non-representable column is rejected fail-fast at write. There is no pure-Python fallback.

The graph → arrays conversion is explicit: `Atomistic.to_frame()` delegates to the native world's `to_frame()`. The box is a first-class attribute (`frame.box`), never metadata. The [Rust Core](rust-core.md) page covers how neighbor lists, `Rdf`, and the analysis catalog surface from Rust.

## Force field: parameters apart, kernels in Rust

`ForceField` is an independent, queryable data structure — parameters are neither embedded in atoms nor derived implicitly. The model has three layers: **Style** (functional form), **ForceFieldType** (parameter set for a type key), and **Potential** (evaluatable kernel). All energy/force kernels live in the Rust core; a style is named by `ff.def_style(category, name)`, and evaluation always goes through `PotentialCompiler(ff).compile(frame)`. The force-field IR is a protocol, so adding a functional form is a registration from Python (`mp.ff.style_registry.GlobalRegistry().register_style(...)`, with an expression or a Python kernel), with nothing rebuilt — the recipe is in [Extending the Force Field](extending-forcefield.md).

## One column vocabulary

Canonical field names (`charge`, not `q`; `mol_id`, not `mol`) are used everywhere inside MolPy. The vocabulary is the core's: `mp.core.keys` is `molpy.core.keys` (`mp.core.keys.CHARGE.key == "charge"`) and `mp.core.schema` gives each column's dtype. Format-specific names exist only inside the native readers and writers, which map them at the boundary, so no molpy code translates column names. The full canonical-name catalog is in the [Naming Conventions](../tutorials/naming-conventions.md) appendix; the extension recipe is in [Adding an I/O Format](extending-io.md).

## The mutation contract

The core data-model API mutates in place: `add_atom`, `add_bond`, `generate_topology`, `translate`, `rotate`, `scale`, `merge` all modify the structure they are called on. Rigid transforms return `None`; additions return live views, `generate_topology` returns counts, and `merge` returns the added atom views. Use `copy.copy(graph)` for an independent graph and `frame.copy()` for an independent frame. Higher-level helpers in `builder` and `op` follow the opposite convention: they must not mutate caller-owned structures unexpectedly — copy first, or build and return a new structure.

## Performance model of assembly

Assembly is one native call and never types anything:

- **Native build** — `Assembler.assemble` runs in the Rust core and releases the GIL. It copies
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
| an external program (a new engine) | plug-in interface | [Adding a New Engine](../user-guide/12_engine.md#adding-a-new-engine-takes-a-table-and-two-methods) |
| an entity/link/struct type | core internals — open an issue first | [Extending the Data Model](extending-core.md) |
| a graph typifier / force-field overlay | typifier internals — open an issue first | [Extending Typifiers](extending-typifiers.md) |
| an interaction style / category | plug-in interface (`mp.ff.style_registry.GlobalRegistry`) | [Extending the Force Field](extending-forcefield.md) |
