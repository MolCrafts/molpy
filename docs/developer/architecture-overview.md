# Architecture Overview

MolPy is a layered toolkit with explicit data flow and minimal magic. This page is the map that every extension guide assumes: which module owns what, how the three class hierarchies of the data model fit together, and where the boundaries between Python and the molrs Rust backend run. Read it once before touching anything under [Extending MolPy](extending-compute.md).

## Module responsibilities

Each package has one clear responsibility with minimal coupling to its siblings:

| Package | Purpose |
|---------|---------|
| `core` | Graph refs/worlds, `Frame`, `Block`, `Box`, units, force-field surfaces |
| `parser` | SMILES / SMARTS (`SmilesIR`, `SmartsPattern`) |
| `builder` | Polymer planning (sequences, chain-length distributions), nanostructures, crystals, virtual sites, topology finalization; site-graph assembly (`Assembler`, placers, `AxisOrienter`) is native and re-exported on the molpy root |
| `conformer` | 3D conformer generation |
| `typifier` | Graph typification: the molrs `Typifier` base and native typifiers (OPLS-AA, MMFF94, element) re-exported; `AntechamberTypifier` / `TLeapTypifier` for GAFF through AmberTools |
| `io` | File I/O: molecular data, trajectories, force-field formats |
| `compute` | Analysis operators — flat modules under `src/molpy/compute/` (`rdf`, `msd`, `dielectric`, `spectra`, …; molrs kernels) |
| `engine` | MD abstractions: LAMMPS, CP2K, OpenMM input generation and execution |
| `wrapper` | Subprocess boundaries to external CLI tools (antechamber, parmchk2, prepgen, tleap, sander) |
| `adapter` | Optional in-memory bridge (RDKit worked example) |
| `data` | Bundled package data: force-field XML files, parameter tables |

`core` depends on nothing above it; everything else builds on `core`. `compute`, `io`, and `engine` operate on the tabular layer (`Frame`/`Block`); `parser`, `builder`, and `typifier` operate on the graph layer (`Atomistic`). `wrapper` and `adapter` sit at the outer edge and never leak external types into `core`.

## The graph layer: live handle views over molrs

The editable graph has one implementation: the molrs world. Python exposes
three cooperating surfaces:

1. **Node refs** — `Atom`, `Bead`, and virtual-site variants are dict-like live
   views identified by a stable native handle.
2. **Relation refs** — `Bond`, `Angle`, `Dihedral`, `Improper`, and `CGBond`
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

`Frame` and `Block` belong exclusively to the [molrs](https://github.com/MolCrafts/molrs) Rust column store. Import them from molpy (`from molpy import Frame, Block`); they are identity re-exports of the molrs types. Columns are typed (float / int / bool / str) and exposed as zero-copy NumPy views; a non-representable column is rejected fail-fast at write. `molcrafts-molrs` is a hard runtime dependency: there is no pure-Python fallback.

The graph → arrays conversion is explicit: `Atomistic.to_frame()` delegates to the molrs world's native `to_frame()`. The box is a first-class attribute (`frame.box`), never metadata. The [molrs Backend](molrs-backend.md) page covers how neighbor lists, RDF, and the analysis catalog surface from Rust.

## Force field: parameters apart, kernels in Rust

`ForceField` is an independent, queryable data structure — parameters are neither embedded in atoms nor derived implicitly. The model has three layers: **Style** (functional form), **Type** (parameter set for a type key), and **Potential** (evaluatable kernel). All energy/force kernels live in molrs (`molrs-ff`); the Python side exposes thin named `Style` subclasses and evaluation always goes through `ff.to_potentials()`. Adding a functional form therefore means a Rust kernel plus a Python style name plus export formatters — the exact recipe is in [Extending the Force Field](extending-forcefield.md).

## Boundary translation: the formatter hierarchy

Canonical field names (`charge`, not `q`; `mol_id`, not `mol`) are used everywhere inside MolPy; format-specific names exist only at the I/O boundary. The translation machinery lives in `core/fields.py`:

```text
molpy.fields                           — the canonical column names (CHARGE, MOL_ID, …)
    ↓
FieldFormatter                         — data field mapping: {format_key: canonical_key}
    ↓                                     canonicalize() / localize() on Block
ForceFieldFormatter(FieldFormatter)    — adds param formatters: {StyleType: Callable}
```

Readers call `canonicalize()` at exit (format → canonical); writers call `localize_frame()` at entry (canonical → format, on a copy). Per-format subclasses live in their own I/O module, and `__init_subclass__` isolates the registries per subclass. The full canonical-name catalog is in the [Naming Conventions](../tutorials/naming-conventions.md) appendix; the extension recipe is in [Adding an I/O Format](extending-io.md).

## The mutation contract

The core data-model API mutates in place and returns `self` (or the created entity) for chaining: `def_atom`, `def_bond`, `get_topo`, `translate`, `rotate`, `scale`, `merge` all modify the structure they are called on. `.copy()` is the explicit opt-in for an independent deep copy. Higher-level helpers in `builder` and `op` follow the opposite convention: they must not mutate caller-owned structures unexpectedly — copy first, or build and return a new structure.

## Performance model of assembly

Assembly is one native call and never types anything:

- **Native build** — `Assembler.assemble` runs in molrs and releases the GIL. It copies
  one template per site, poses each copy once (placer, then optional orienter), and links
  every site bond through its two ports. There is no per-bond Python loop and no
  reaction matching: which ports join is decided from the site graph, not searched for
  in the growing world.
- **No retyping during the build** — the world comes back with atoms, bonds and ports
  only. Typing is a separate, whole-graph step the caller runs once afterwards.
- **Explicit finalization** — angles and dihedrals are generated once, when needed, by
  `StructureFinalizer(Finalization.TOPOLOGY)`; `Finalization.ATOMS` keeps a large
  system atoms-only until an MD writer needs topology.

## Where extension happens

| I want to add… | Layer | Guide |
|---|---|---|
| an analysis operation | plug-in interface | [Adding a Compute Operation](extending-compute.md) |
| a file format | plug-in interface | [Adding an I/O Format](extending-io.md) |
| an external tool integration | plug-in interface | [Adding a Wrapper or Adapter](extending-integration.md) |
| an entity/link/struct type | core internals — open an issue first | [Extending the Data Model](extending-core.md) |
| a graph typifier / force-field overlay | typifier internals — open an issue first | [Extending Typifiers](extending-typifiers.md) |
| an interaction style / kernel | core internals — open an issue first | [Extending the Force Field](extending-forcefield.md) |
