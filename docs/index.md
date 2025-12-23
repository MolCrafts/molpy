---
template: home.html
hide:
  - navigation
  - toc
---

# MolPy

<div class="badges" markdown>
  [![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/license-BSD-green.svg)](https://github.com/MolCrafts/molpy/blob/master/LICENSE)
  [![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://molcrafts.github.io/molpy)
</div>

<p class="lead" markdown>a LLM-ready toolkit for building molecular systems.</p>

<div class="button-group" markdown>
  [Quick Start](getting-started/quickstart.ipynb){ .md-button .md-button--primary }
  [GitHub](https://github.com/MolCrafts/molpy){ .md-button }
</div>

---

## Features

<div class="grid cards" markdown>

- :robot:{ .lg .middle } __LLM-Friendly__

    ---

    Stateless functions, explicit types, zero global state. Every parameter is documented and visible, making it trivial for AI agents to read, understand, and generate correct code.

- :building_construction:{ .lg .middle } __Extensible__

    ---

    Build custom workflows by composing functions and data structures. No inheritance hierarchies to navigate, just pure functions you can mix and match.

- :link:{ .lg .middle } __Interoperable__

    ---

    Native adapters for industry-standard tools: AmberTools, LAMMPS, OpenMM, Packmol, RDKit. One-function export to simulation engines or analysis pipelines.

- :gear:{ .lg .middle } __Modular Architecture__

    ---

    Organized into domain-specific modules (builder, parser, typifier, adapter) around a unified `Frame`/`Block` data model. Use only what you need.

- :bar_chart:{ .lg .middle } __Flexible Structures__

    ---

    Lightweight `Frame` for single configurations, hierarchical `Block` for composed systems, memory-mapped `Trajectory` for multi-million atom simulations.

- :wrench:{ .lg .middle } __Force-Field System__

    ---

    Rule-based atom typing with SMARTS/SMIRKS patterns. Store, validate, and export force field parameters to any simulation engine with full provenance tracking.


</div>

---

## Why MolPy?

MolPy is designed for researchers and engineers who need **reliable**, **transparent**, and **composable** molecular modeling workflows. Whether you're building simulation systems, designing force fields, or integrating ML/AI pipelines, MolPy provides the building blocks you need.

<div class="grid cards" markdown>

- :mag:{ .lg .middle } __Typical Use Cases__

    ---

    - **System preparation** – build polymer melts, solvated proteins, or crystal interfaces
    - **Force field development** – define custom atom types with SMARTS/SMIRKS patterns
    - **Simulation workflows** – export to LAMMPS, OpenMM, or GROMACS with one function call
    - **LLM-driven research** – let AI agents write and run molecular modeling scripts

- :bulb:{ .lg .middle } __Design Highlights__

    ---

    - **No global state** – all functions are pure and testable
    - **Explicit over implicit** – every parameter is visible and documented
    - **Composition over inheritance** – mix and match components freely
    - **Memory efficiency** – lazy loading and memory-mapped trajectories for large systems

- :rocket:{ .lg .middle } __When to Choose MolPy__

    ---

    - You need **type-safe APIs** for LLM integration or code generation
    - You want **full control** without framework lock-in
    - You're building **custom workflows** that don't fit traditional GUI tools
    - You need **interoperability** between multiple simulation engines

</div>

---

## Quick Links

<div class="grid" markdown>

!!! info "Documentation"
    New to MolPy? Start with our [Quick Start](getting-started/quickstart.ipynb). Explore comprehensive guides in our [User Guide](user-guide/index.md) and [Tutorials](tutorials/index.md) with runnable notebooks and end-to-end examples.

!!! example "API Reference"
    Detailed API documentation is available in the [API Reference](api/index.md) section.

!!! question "Need Help?"
    Check out our [FAQ](getting-started/faq.md) or open an issue on [GitHub](https://github.com/MolCrafts/molpy/issues).

</div>

---

## Comprehensive Documentation

MolPy provides a growing collection of runnable Jupyter notebooks alongside generated API documentation.

<div class="grid cards" markdown>

- :books:{ .lg .middle } __Runnable Notebooks__

    ---

    Practical examples across the core modules: Parser, Reacter, Builder, Typifier, IO, Adapter, Potential, and more.

- :wrapped_gift:{ .lg .middle } __Extensive API Coverage__

    ---

    Public APIs are documented with parameters, return values, and usage examples.

- :school:{ .lg .middle } __Architecture Guides__

    ---

    Developer documentation covering recipe system, design patterns, IR principles, and contribution guidelines.

</div>

[Explore the User Guide →](user-guide/index.md){ .md-button .md-button--primary }

---

## Roadmap

<div class="roadmap-grid" markdown>

<div markdown>

### Core Foundations
- [x] Stabilize core data structures (`Frame`, `Block`, `Box`, `Trajectory`)
- [x] Unify topology representation (bonds/angles/dihedrals)
- [x] Improve I/O layer (XYZ, PDB, LAMMPS DATA)
- [x] Strengthen typing, doctests, and basic documentation

</div>

<div markdown>

### Modeling & Construction
- [x] General lattice + crystal builder (SC/BCC/FCC/HCP + regions)
- [x] Basic molecular builders (monomers, ports, polymer assembly)
- [x] System assembly utilities (packing, placement, initialization)
- [x] Core editing tools (add/remove atoms, merge fragments, simple reactions)

</div>

<div markdown>

### Force Fields & Typing
- [x] Force field container (atom types + bonded parameters)
- [x] Typifier system (rule-based SMARTS-style typing)
- [x] Parameter assignment and export for external engines
- [x] Validation / consistency tools for typed systems

</div>

<div markdown>

### Compute & Analysis
- [x] Unified `Compute` + `Result` abstraction
- [x] Common analysis functions (RDF, MSD, basic time series)
- [ ] Optional shared context (neighbor lists, box info)
- [ ] Export analysis results to arrays / DataFrames

</div>

<div markdown>

### Performance & User Experience
- [ ] Rust backend for performance-critical operations
- [ ] Visualization integration with molvis

</div>

</div>

---

## Ecosystem

- **[MolVis](https://github.com/MolCrafts/molvis)** — Production-level visualization with WebGL acceleration and real-time manipulation
- **[MolRS](https://github.com/MolCrafts/molrs)** — Rust backend for performance-critical operations

MolPy is the **core Python library** in the MolCrafts ecosystem. MolVis provides high-performance 3D visualization, while MolRS offers compiled speed for compute-intensive tasks. All three projects share a unified data model and can be used independently or together.

---

## Community & Contributing

We welcome contributions from researchers, developers, and users. There are three main ways to get involved:

<div class="grid" markdown>

!!! abstract "Contribute Code"
    Found a bug or have an idea for improvement? Read the [Contributing Guide](developer/contributing.md) and submit a pull request on
    [GitHub](https://github.com/MolCrafts/molpy).

!!! note "Share & Discuss"
    Built something with MolPy or want to discuss workflows, design choices, or publications?
    Join the conversation in [Discussions](https://github.com/MolCrafts/molpy/discussions).

!!! question "Get Help & Request Features"
    Need help or missing a feature? Start with the [FAQ](getting-started/faq.md), then open an issue or proposal on
    [GitHub Issues](https://github.com/MolCrafts/molpy/issues) or [Discussions](https://github.com/MolCrafts/molpy/discussions).

</div>


---

## Dependencies

- [numpy](https://github.com/numpy/numpy) — Numerical computing
- [python-igraph](https://github.com/igraph/python-igraph) — Graph analysis
- [lark](https://github.com/lark-parser/lark) — SMARTS/SMILES parsing
- [h5py](https://www.h5py.org/) - Serialization

---

## License

This project is licensed under the BSD-3-Clause License. See [LICENSE](https://github.com/MolCrafts/molpy/blob/master/LICENSE) for details.
