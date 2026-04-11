# FAQ

This page answers common questions about MolPy's scope, its relation to adjacent packages, and the cases in which another tool is the better choice.

## Tool Comparison

This section compares **MolPy** with widely used molecular modeling and simulation-preparation tools such as **ASE**, **mBuild**, **ParmEd**, **RDKit**, and **MDAnalysis**. The aim is not to rank packages, but to clarify which layer of the workflow MolPy is designed to address.

At its core, MolPy is designed around three principles:

1. **A unified data model** based on `Frame`, `Block`, and `Box`, with explicit topology and typed attributes.
2. **Explicit force-field abstraction**, separating chemical structure from parameter assignment.
3. **Explicit molecular editing**, including building, packing, polymerization, and reaction-based transformations.

The following sections summarize the conceptual differences and complementarity between MolPy and other tools.

---

### MolPy and ASE

ASE excels at representing atomic configurations and interfacing with a wide variety of quantum and classical engines.
However, ASE’s core abstraction is not explicit chemical topology or force-field semantics. `Atoms` objects carry coordinates, species, cells, PBC flags, and calculator-related data, but bonded topology is not a first-class concept in the way it is in MolPy.

MolPy addresses a different layer of the workflow:

* **Data model**: MolPy offers a structured representation of coordinates, topology, and periodic boxes, whereas ASE focuses on atom lists plus calculator objects.
* **Force fields**: ASE provides calculators, not force-field management; MolPy provides typifiers, bonded terms, and parameter containers.
* **Editing and building**: MolPy supports polymer construction, packing, and structural editing that rely on topology; ASE typically treats bonding as descriptive rather than generative.

MolPy is therefore complementary to ASE: models built in MolPy can be exported to engines managed by ASE, but the abstraction layer is different.

---

### MolPy and mBuild

mBuild is a widely used template-based framework for constructing molecules and hierarchical structures.
It is especially effective for assembling reusable building blocks.

MolPy differs in several ways:

* **Data structure philosophy**: mBuild uses hierarchical “Compound” objects; MolPy uses flat, typed `Frame` structures suited to downstream simulation workflows.
* **Force fields**: mBuild delegates typing to packages such as Foyer, whereas MolPy integrates typing as a first-class component.
* **Operations**: MolPy exposes explicit monomer/port/reaction primitives with more explicit data flow and simulation-oriented semantics.

MolPy adopts a different design emphasis: builder logic and force-field semantics are coupled through an explicit typed data model rather than separated across loosely connected layers.

---

### MolPy and ParmEd

ParmEd specializes in editing existing force-field parameter sets and prepared simulation topologies.
It is widely used for converting, patching, or inspecting structures produced by external tools.

MolPy complements ParmEd with a different design focus:

* **Topology + data model**: MolPy uses a unified `Frame` representation rather than multiple classes for residues, atoms, parameters, etc.
* **Generation vs. modification**: ParmEd edits existing systems; MolPy builds systems from scratch.
* **Typing model**: ParmEd is centered on editing prepared topologies and parameterized systems, especially in established MD ecosystems; MolPy’s Typifier framework is force-field-agnostic and rule-based.

MolPy can generate systems for downstream simulation workflows that, if needed, are later processed by ParmEd, but their internal assumptions differ significantly.

---

### MolPy and RDKit

RDKit is one of the most widely used and capable open-source cheminformatics toolkits, covering SMILES, SMARTS, conformer generation, and substructure queries.

MolPy can interoperate with RDKit where cheminformatics functionality is needed, but it extends beyond molecular graphs:

* **Simulation-centric structures**: MolPy supports periodic boxes, bonded topology for MD, and multiphase assemblies.
* **System-level operations**: polymers, lattices, packing, and reactions are outside RDKit’s scope.
* **Interoperability**: RDKit molecules can be converted into MolPy `Frame` objects for downstream modeling.

RDKit manages chemical graphs; MolPy manages molecular systems for simulation-preparation workflows.

---

### MolPy and MDAnalysis / MDTraj

MDAnalysis and MDTraj are trajectory analysis packages designed for post-processing simulation output.

MolPy focuses on construction and manipulation instead:

* **Before simulation**: building, editing, typing, packing, polymerizing
* **After simulation (lightweight)**: a minimal `Compute` abstraction for basic analyses (RDF, MSD, neighbors)
* **Not a replacement**: heavy trajectory analysis is still best done with MDAnalysis/MDTraj.

---

### Summary Table

| Task / Capability                | ASE     | mBuild    | ParmEd  | RDKit   | MDAnalysis | MolPy   |
| -------------------------------- | ------- | --------- | ------- | ------- | ---------- | ------- |
| Atomic structure editing         | partial | yes       | partial | limited | no         | **yes** |
| Simulation topology              | minimal | limited   | yes     | no      | no         | **yes** |
| Polymer builder                  | no      | yes       | no      | no      | no         | **yes** |
| Reaction transforms / topology editing | no | partial | no | SMIRKS-style | no | **yes** |
| Force-field typing               | no      | via Foyer | yes     | MMFF/UFF | no        | **yes** |
| System packing / assembly        | limited | yes       | no      | no      | no         | **yes** |
| Trajectory analysis              | basic   | no        | no      | no      | yes        | minimal |
| Periodic system representation   | limited | partial   | yes     | no      | analysis-oriented | **yes** |

---

### When NOT to use MolPy

MolPy is a system construction and parameterization toolkit. It is not the right tool for every molecular modeling task:

* **Heavy trajectory analysis** — use MDAnalysis or MDTraj. MolPy provides minimal analysis (MSD, basic correlations), but it is not designed for large-scale post-processing of multi-gigabyte trajectories.
* **Quantum chemistry** — use PySCF, ASE with a QM calculator, or Gaussian/ORCA directly. MolPy does not perform electronic structure calculations.
* **Cheminformatics** — use RDKit. MolPy delegates conformer generation, substructure search, and chemical informatics to RDKit via its adapter layer.
* **High-throughput screening** — MolPy is designed for constructing individual systems with full control, not for screening thousands of molecules in a pipeline. Use workflow engines, screening platforms, or broader CADD stacks for that kind of campaign.
* **Running simulations** — MolPy can generate input files and launch engines, but it is not a simulation runner. Simulation execution and campaign management should be handled by the target engines and external workflow managers such as Signac or AiiDA.

---

### Why MolPy uses a distinct architecture

MolPy was developed for workflows in which topology editing, force-field assignment, and simulation export must remain explicit across multiple preparation steps. Existing packages provide strong functionality in individual domains, but they often rely on different data models, assumptions, and extension patterns. As a result, cross-package workflows can become fragile, engine-dependent, and difficult to automate reproducibly.

MolPy therefore uses a common representation that extends across polymers, electrolytes, crystals, and mixtures without requiring a separate internal model for each workflow. The emphasis is on explicit transformations, inspectable parameters, and stable programmatic interfaces rather than hidden conventions.

This architecture does not replace specialized packages. It provides a consistent preparation layer for workflows that need direct control over reactive topology editing and downstream simulation export.
