# FAQ

Why MolPy exists, how it relates to the tools you already use, and when you
should reach for something else.

## Why does MolPy exist?

Preparing a complex molecular system means editing topology, assigning
force-field parameters, and exporting engine files — steps usually spread
across tools with different data models and hidden conventions. Cross-package
workflows built that way are fragile, engine-dependent, and hard to reproduce.

MolPy keeps the whole preparation layer in one explicit representation, built
on three principles:

1. **One data model.** `Atomistic`, `Frame`, `Block`, and `Box` carry explicit
   topology and typed attributes — the same objects from first parse to final
   export, across polymers, electrolytes, crystals, and mixtures.
2. **Force fields as data.** Parameters live in a queryable structure separate
   from the chemistry, so you can inspect and validate a model *before* an
   expensive simulation, not after it fails.
3. **Explicit editing.** Building, packing, polymerization, and reaction-based
   transformations are programmatic operations with no hidden state between
   them.

MolPy does not replace specialized packages — it is the preparation layer that
keeps them composable. The comparisons below draw the boundaries precisely.

## How does MolPy compare to…?

### …ASE?

ASE excels at representing atomic configurations and interfacing with quantum
and classical engines. Its `Atoms` objects carry coordinates, species, cells,
and calculator data — but bonded topology is not a first-class concept the way
it is in MolPy.

* **Data model**: MolPy structures coordinates, topology, and periodic boxes; ASE focuses on atom lists plus calculator objects.
* **Force fields**: ASE provides calculators, not force-field management; MolPy provides typifiers, bonded terms, and parameter containers.
* **Editing and building**: MolPy's polymer construction, packing, and reactions rely on generative topology; ASE treats bonding as descriptive.

The two are complementary: systems built in MolPy can be exported to engines
managed by ASE.

### …mBuild?

mBuild is a template-based framework for assembling molecules from reusable
building blocks.

* **Data structure**: mBuild uses hierarchical `Compound` objects; MolPy uses flat, typed structures aimed at downstream simulation.
* **Force fields**: mBuild delegates typing to Foyer; MolPy integrates typing as a first-class component.
* **Operations**: MolPy exposes explicit monomer / port / reaction primitives with simulation-oriented semantics.

### …ParmEd?

ParmEd specializes in editing *existing* parameterized topologies — converting,
patching, and inspecting systems produced by other tools.

* **Direction**: ParmEd edits prepared systems; MolPy builds systems from scratch.
* **Data model**: MolPy uses one `Frame` representation rather than separate classes for residues, atoms, and parameters.
* **Typing**: MolPy's typifier framework is force-field-agnostic and rule-based rather than tied to established MD ecosystems.

MolPy-generated systems can be post-processed with ParmEd where its ecosystem
integrations are needed.

### …RDKit?

RDKit is the reference open-source cheminformatics toolkit — SMILES, conformer
generation, substructure search. MolPy interoperates with it (see the
[adapter layer](../api/adapter.md)) and extends beyond molecular graphs:

* **Simulation-centric structures**: periodic boxes, bonded MD topology, multiphase assemblies.
* **System-level operations**: polymers, packing, and reactions are outside RDKit's scope.
* **Interoperability**: RDKit molecules convert to and from MolPy objects.

RDKit manages chemical graphs; MolPy manages molecular *systems* headed for
simulation.

### …MDAnalysis / MDTraj?

They analyze trajectories *after* a simulation; MolPy's center of gravity is
*before* it — building, editing, typing, packing. MolPy's
[Compute](../compute/index.md) layer covers a broad analysis catalog backed by
Rust kernels, but multi-terabyte post-processing pipelines remain MDAnalysis /
MDTraj territory.

### Capability summary

| Task / Capability                | ASE     | mBuild    | ParmEd  | RDKit   | MDAnalysis | MolPy   |
| -------------------------------- | ------- | --------- | ------- | ------- | ---------- | ------- |
| Atomic structure editing         | partial | yes       | partial | limited | no         | **yes** |
| Simulation topology              | minimal | limited   | yes     | no      | no         | **yes** |
| Polymer builder                  | no      | yes       | no      | no      | no         | **yes** |
| Reaction transforms / topology editing | no | partial | no | SMIRKS-style | no | **yes** |
| Force-field typing               | no      | via Foyer | yes     | MMFF/UFF | no        | **yes** |
| System packing / assembly        | limited | yes       | no      | no      | no         | **yes** |
| Trajectory analysis              | basic   | no        | no      | no      | yes        | broad, Rust-backed |
| Periodic system representation   | limited | partial   | yes     | no      | analysis-oriented | **yes** |

## When should I use something else?

MolPy is a system construction and parameterization toolkit. It is not the
right tool for every task:

* **Multi-terabyte trajectory post-processing** — use MDAnalysis or MDTraj; MolPy's compute layer targets per-system analysis, not massive campaign pipelines.
* **Quantum chemistry** — use PySCF, ASE with a QM calculator, or Gaussian/ORCA. MolPy does no electronic structure.
* **Pure cheminformatics** — use RDKit directly. MolPy delegates substructure search and conformer generation to it via the adapter layer.
* **High-throughput screening** — MolPy builds individual systems with full control; screening thousands of candidates belongs in workflow engines and CADD stacks.
* **Running simulation campaigns** — MolPy generates inputs and can launch engines, but execution management belongs to the engines themselves and tools like Signac or AiiDA.
