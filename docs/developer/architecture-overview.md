# Architecture Overview

MolPy is a layered toolkit with explicit data flow and minimal magic. This page is the map that every extension guide assumes: which module owns what, how the three class hierarchies of the data model fit together, and where the boundaries between Python and the molrs Rust backend run. Read it once before touching anything under [Extending MolPy](extending-compute.md).

## Module responsibilities

Each package has one clear responsibility with minimal coupling to its siblings:

| Package | Purpose |
|---------|---------|
| `core` | Data structures: `Entity`, `Link`, `Struct`, `Atomistic`, `Frame`, `Block`, `Box`, `ForceField` |
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

## The graph layer: Entity, Link, Struct

The editable data model has three class hierarchies:

1. **Entity** (node) — dict-like base for atoms, beads, and particles, with identity-based hashing (`hash()` is `id()`). Two atoms with identical properties are still different atoms. Subclasses: `Atom`, `Bead`.
2. **Link** (edge) — holds an ordered tuple of `Entity` endpoints. Subclasses: `Bond`, `Angle`, `Dihedral`, `Improper`, `CGBond`.
3. **Struct** (container) — aggregates entities and links in `TypeBucket` collections and manages CRUD. Subclasses: `Atomistic`, `CoarseGrain`.

`TypeBucket` stores items by concrete type: registering `Atom` means `bucket[Atom]` returns all `Atom` instances, with subclasses included in parent queries. New entity or link types must be registered in the struct's `__init__` — see [Extending the Data Model](extending-core.md).

## The tabular layer: Block and Frame run on molrs

`Frame` and `Block` are re-exports of the [molrs](https://github.com/MolCrafts/molrs) Rust column store — `molpy.core.frame.Frame` *is* `molrs.Frame`. Columns are typed (float / int / bool / str) and exposed as zero-copy NumPy views; a non-representable column is rejected fail-fast at write. `molcrafts-molrs` is a hard runtime dependency: there is no pure-Python fallback.

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

Assembly is linear in chain length because nothing per-edit walks the whole system:

- **One paste, in-place edits** — the world is built once; `molrs.Reaction.apply` edits it in
  place and returns the atoms it touched. There is no per-bond copy and no `entity_map`
  remapping, which is where the old builder's four independent O(N)-per-bond terms lived.
- **Local retyping** — only the ball around each new bond is retyped, at the radius
  `AffectedRegion.around` derives from the assembler's `reach`. Identical junctions dedupe
  by structural hash, so the typing cost tracks distinct environments, not bonds.
- **Local topology** — the angles and dihedrals a new bond creates are inserted from the
  region. `generate_topology` is never called on the assembled world.
- **Matching once** — the kernel matches the reaction's patterns in O(N) and hands the
  occurrences to the `Selector`; pairing (the only O(sites^2) step) belongs to the selector
  that needs it, and `TopologySelector` indexes by residue instead.


Nothing per-connection scales with chain length. The old builder copied the accumulated
structure once per bond and remapped its entities, which made a DP=N chain cost O(N²) in
copying alone; the assembly kernel pastes once, edits in place, and never copies again.
The one O(N) step left is the single paste at the start, and the single `.copy()`
`assemble` takes so it does not mutate the world you handed it.

## Where extension happens

| I want to add… | Layer | Guide |
|---|---|---|
| an analysis operation | plug-in interface | [Adding a Compute Operation](extending-compute.md) |
| a file format | plug-in interface | [Adding an I/O Format](extending-io.md) |
| an external tool integration | plug-in interface | [Adding a Wrapper or Adapter](extending-integration.md) |
| an entity/link/struct type | core internals — open an issue first | [Extending the Data Model](extending-core.md) |
| a graph typifier / force-field overlay | typifier internals — open an issue first | [Extending Typifiers](extending-typifiers.md) |
| an interaction style / kernel | core internals — open an issue first | [Extending the Force Field](extending-forcefield.md) |
