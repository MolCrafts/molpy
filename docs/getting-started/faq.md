## Tool Comparison

This section provides an overview of how **MolPy** differs from widely used molecular modeling and simulation preparation tools such as **ASE**, **mBuild**, **ParmEd**, **RDKit**, and **MDAnalysis**.
Rather than competing directly with any single package, MolPy combines the strengths of several approaches into a **modern, coherent, and strongly-typed framework** for molecular system construction and analysis.

At its core, MolPy is designed around three principles:

1. **A unified data model** based on `Frame`, `Block`, and `Box`, with explicit topology and typed attributes.
2. **Explicit force-field abstraction**, separating chemical structure from parameter assignment.
3. **Composable molecular editing**, including building, packing, polymerization, and reaction-based transformations.

The following sections summarize the conceptual differences and complementarity between MolPy and other tools.

---

### MolPy and ASE

ASE excels at representing atomic configurations and interfacing with a wide variety of quantum and classical engines.
However, ASE’s data model is intentionally minimal: atoms are represented primarily by positions and atomic numbers, with limited support for chemical topology.

MolPy addresses a different layer of the workflow:

* **Data model**: MolPy offers a structured representation of coordinates, topology, and periodic boxes, whereas ASE focuses on atom lists plus calculator objects.
* **Force fields**: ASE provides calculators, not force-field management; MolPy provides typifiers, bonded terms, and parameter containers.
* **Editing and building**: MolPy supports polymer construction, packing, and structural editing that rely on topology; ASE typically treats bonding as descriptive rather than generative.

MolPy is therefore complementary to ASE: models built in MolPy can be exported to engines managed by ASE, but the abstraction layer is different.

---

### MolPy and mBuild

mBuild is a powerful template-based framework for constructing molecules and hierarchical structures.
It is excellent for assembling reusable building blocks.

MolPy differs in several ways:

* **Data structure philosophy**: mBuild uses hierarchical “Compound” objects; MolPy uses flat, typed `Frame` structures suitable for direct simulation.
* **Force fields**: mBuild delegates typing to packages such as Foyer, whereas MolPy integrates typing as a first-class component.
* **Operations**: MolPy exposes explicit monomer/port/reaction primitives, designed to be predictable, less magical, and easier to automate.

Conceptually, MolPy can be seen as a next-generation modeling toolkit that unifies builder logic with force-field semantics in a more explicit and type-safe manner.

---

### MolPy and ParmEd

ParmEd specializes in editing existing force-field parameter sets and prepared simulation topologies.
It is widely used for converting, patching, or inspecting structures produced by external tools.

MolPy complements ParmEd with a different design focus:

* **Topology + data model**: MolPy uses a unified `Frame` representation rather than multiple classes for residues, atoms, parameters, etc.
* **Generation vs. modification**: ParmEd edits existing systems; MolPy builds systems from scratch.
* **Typing model**: ParmEd encodes Amber/CHARMM conventions; MolPy’s Typifier framework is force-field-agnostic and rule-based.

MolPy can generate simulation-ready systems that, if needed, are later processed by ParmEd, but their internal assumptions differ significantly.

---

### MolPy and RDKit

RDKit is unmatched in cheminformatics: SMILES, SMARTS, conformer generation, and substructure queries.

MolPy builds on RDKit but expands beyond molecular graphs:

* **Simulation-centric structures**: MolPy supports periodic boxes, bonded topology for MD, and multiphase assemblies.
* **System-level operations**: polymers, lattices, packing, and reactions are outside RDKit’s scope.
* **Interoperability**: RDKit molecules can be converted into MolPy `Frame` objects for downstream modeling.

RDKit manages chemical graphs; MolPy manages **simulation-ready molecular systems**.

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
| Reaction modeling                | no      | partial   | no      | SMIRKS  | no         | **yes** |
| Force-field typing               | no      | via Foyer | yes     | limited | no         | **yes** |
| System packing / assembly        | limited | yes       | no      | no      | no         | **yes** |
| Trajectory analysis              | yes     | no        | no      | no      | yes        | minimal |
| Periodic MD-ready representation | limited | partial   | yes     | no      | yes        | **yes** |

---

### Why a New Ecosystem?

MolPy is built on the idea that modern molecular modeling needs more than patched-together tools. Existing packages provide excellent functionality in isolated domains, but they rarely share data models, assumptions, or extension patterns. This makes workflows fragile, engine-dependent, and difficult to automate.

MolPy introduces a modern, flexible, and extensible data structure that can adapt to diverse molecular systems—polymers, electrolytes, crystals, mixtures—without rewriting core logic. It minimizes external dependencies, avoids hidden conventions, and uses explicit, strongly typed APIs that are AI-friendly and reproducible.

Instead of forcing users to juggle multiple incompatible libraries, MolPy offers a clean foundation that can grow with the complexity of the system and the workflow.
