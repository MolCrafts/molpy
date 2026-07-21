# Architecture Overview

MolPy is a layered toolkit with explicit data flow and minimal magic. This page is the map that every extension guide assumes: which module owns what, how the three class hierarchies of the data model fit together, and where the boundaries between Python and the molrs Rust backend run. Read it once before touching anything under [Extending MolPy](extending-compute.md).

## Module responsibilities

Each package has one clear responsibility with minimal coupling to its siblings:

| Package | Purpose |
|---------|---------|
| `core` | molrs-backed graph refs/worlds, `Frame`, `Block`, `Box`, units, and force-field surfaces |
| `parser` | Grammar-based parsing: SMILES, SMARTS, BigSMILES, G-BigSMILES, CGSmiles |
| `builder` | System assembly: one `GraphAssembler` kernel + a `Selector` family, virtual sites, AmberTools integration |
| `typifier` | Graph typification: one `Typifier` contract (`MolGraph -> MolGraph`), molrs OPLS-AA/MMFF re-exports, and MolPy-side overlays such as CL&P |
| `pack` | Packing workflows: Packmol integration, density targets |
| `io` | File I/O: readers/writers for molecular data, trajectories, and force-field formats |
| `compute` | Analysis operators over `Frame`/`Block` data |
| `engine` | MD abstractions: LAMMPS, CP2K, OpenMM input generation and execution |
| `wrapper` | Subprocess boundaries to external CLI tools (antechamber, packmol, …) |
| `adapter` | In-memory bridges to external object models (RDKit, OpenBabel, …) |
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
   graph algorithms. Their `.atoms`, `.bonds`, and related properties are live
   molrs view collections, not Python buckets or mirrored lists.

There is no `Struct`/`TypeBucket` registration layer. Adding a new stored node
or relation kind changes the molrs schema and bindings; it is not a Python
subclassing hook. See [Extending the Data Model](extending-core.md).

## The tabular layer: Block and Frame run on molrs

`Frame` and `Block` belong exclusively to the [molrs](https://github.com/MolCrafts/molrs) Rust column store. Import them from molpy (`from molpy import Frame, Block`); they are identity re-exports or compatibility module for them. Columns are typed (float / int / bool / str) and exposed as zero-copy NumPy views; a non-representable column is rejected fail-fast at write. `molcrafts-molrs` is a hard runtime dependency: there is no pure-Python fallback.

The graph → arrays conversion is explicit: `Atomistic.to_frame()` delegates to the molrs world's native `to_frame()`. The box is a first-class attribute (`frame.box`), never metadata. The [molrs Backend](molrs-backend.md) page covers how neighbor lists, RDF, and the analysis catalog surface from Rust.

## Force field: parameters apart, kernels in Rust

`ForceField` is an independent, queryable data structure — parameters are neither embedded in atoms nor derived implicitly. The model has three layers: **Style** (functional form), **Type** (parameter set for a type key), and **Potential** (evaluatable kernel). All energy/force kernels live in molrs (`molrs-ff`); the Python side exposes thin named `Style` subclasses and evaluation always goes through `ff.to_potentials()`. Adding a functional form therefore means a Rust kernel plus a Python style name plus export formatters — the exact recipe is in [Extending the Force Field](extending-forcefield.md).

## Boundary translation: the formatter hierarchy

Canonical field names (`charge`, not `q`; `mol_id`, not `mol`) are used everywhere inside MolPy; format-specific names exist only at the I/O boundary. The translation machinery lives in `core/fields.py`:

```text
FieldSpec                              — canonical field definition (key, dtype, shape, doc)
    ↓
FieldFormatter                         — data field mapping: {format_key: FieldSpec}
    ↓                                     canonicalize() / localize() on Block
ForceFieldFormatter(FieldFormatter)    — adds param formatters: {StyleType: Callable}
```

Readers call `canonicalize()` at exit (format → canonical); writers call `localize_frame()` at entry (canonical → format, on a copy). Per-format subclasses live in their own I/O module, and `__init_subclass__` isolates the registries per subclass. The full canonical-name catalog is in the [Naming Conventions](../tutorials/naming-conventions.md) appendix; the extension recipe is in [Adding an I/O Format](extending-io.md).

## The mutation contract

The core data-model API mutates in place and returns `self` (or the created entity) for chaining: `def_atom`, `def_bond`, `get_topo`, `move`, `rotate`, `merge` all modify the structure they are called on. `.copy()` is the explicit opt-in for an independent deep copy. Higher-level helpers in `builder` and `op` follow the opposite convention: they must not mutate caller-owned structures unexpectedly — copy first, or build and return a new structure.

## Performance model of the build loop

Assembly is linear in chain length because the growing graph is never retyped per edit:

- **Compile before execution** — the selector first yields the complete binding set. The
  compiler overlays all planned forming bonds on the intact templates and materializes a
  bounded product motif for every junction. Residue-backed motifs contain whole user-defined
  monomers, so they do not need artificial graph completion.
- **Rooted local cache** — an isomorphism key includes the product motif, its chemical scalar
  labels and the touched root. Identical junctions are typified once, even across builds. A
  cache value contains scalar per-atom annotations only (`type`, `charge`, pair parameters,
  etc.); it never copies local angle/dihedral rows into the world.
- **One batch edit** — `molrs.Reaction.apply_many` resolves every leaving group against the
  intact graph, deletes their union with one relation-table scan, then executes every planned
  transform. There is no “grow once, retype the accumulated polymer, repeat” loop.
- **Explicit finalization** — `Finalization.ATOMS` stops after atom write-back;
  `Finalization.TOPOLOGY` generates angle/dihedral topology once (the default); and
  `Finalization.BONDED` additionally runs `ForceFieldParams` once over that topology. Large
  systems can remain atoms-only until an MD writer needs topology.
- **Matching once** — the kernel matches the reaction's patterns in O(N) and hands the
  occurrences to the `Selector`; pairing (the only O(sites^2) step) belongs to the selector
  that needs it, and `TopologySelector` indexes by residue instead.


Nothing per-connection scales with chain length. The old builder copied the accumulated
structure once per bond and remapped its entities, which made a DP=N chain cost O(N²) in
copying alone. The compile-first kernel performs bounded local work per binding, a single
batch reaction, and at most one requested whole-graph finalization pass.

## Where extension happens

| I want to add… | Layer | Guide |
|---|---|---|
| an analysis operation | plug-in interface | [Adding a Compute Operation](extending-compute.md) |
| a file format | plug-in interface | [Adding an I/O Format](extending-io.md) |
| an external tool integration | plug-in interface | [Adding a Wrapper or Adapter](extending-integration.md) |
| an entity/link/struct type | core internals — open an issue first | [Extending the Data Model](extending-core.md) |
| a graph typifier / force-field overlay | typifier internals — open an issue first | [Extending Typifiers](extending-typifiers.md) |
| an interaction style / kernel | core internals — open an issue first | [Extending the Force Field](extending-forcefield.md) |
