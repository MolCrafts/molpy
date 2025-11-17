---
template: home.html
hide:
  - navigation
  - toc
---

# MolPy

<div class="badges" markdown>
  [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/license-BSD-green.svg)](https://github.com/MolCrafts/molpy/blob/master/LICENSE)
  [![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://molcrafts.github.io/molpy)
</div>

<p class="lead">A fast, clean, and composable toolkit for molecular modeling</p>

<div class="button-group" markdown>
  [Quick Start](getting-started/quickstart.ipynb){ .md-button .md-button--primary }
  [GitHub](https://github.com/MolCrafts/molpy){ .md-button }
</div>

---

## Features

<div class="grid cards" markdown>

- :robot:{ .lg .middle } __LLM-Friendly__

    ---

    Clean APIs, explicit typing, no hidden magic

- :building_construction:{ .lg .middle } __Extensible__

    ---

    Composition-based design for easy customization

- :link:{ .lg .middle } __Interoperable__

    ---

    Works smoothly with AmberTools, LAMMPS, OpenMM, Packmol, RDKit

- :gear:{ .lg .middle } __Modular Architecture__

    ---

    Clear, composable modules built around a unified data model

- :bar_chart:{ .lg .middle } __Flexible Structures__

    ---

    Frame/Block system and memory-mapped Trajectory support

- :wrench:{ .lg .middle } __Force-Field System__

    ---

    SMARTS/SMIRKS typing and structured parameter management


</div>

---

## Quick Links

<div class="grid" markdown>

!!! tip "Getting Started"
    New to MolPy? Start with our [Quick Start Guide](getting-started/quickstart.ipynb) to learn the basics.

!!! info "Documentation"
    Explore comprehensive guides in our [User Guide](user-guide/index.md) and [Tutorials](tutorials/index.md).

!!! example "API Reference"
    Detailed API documentation is available in the [API Reference](api/index.md) section.

!!! question "Need Help?"
    Check out our [FAQ](getting-started/faq.md) or open an issue on [GitHub](https://github.com/MolCrafts/molpy/issues).

</div>

---

## Roadmap

<div class="roadmap-grid" markdown>

### Core Foundations
- [x] Stabilize core data structures (`Frame`, `Block`, `Box`, `Trajectory`)
- [x] Unify topology representation (bonds/angles/dihedrals)
- [x] Improve I/O layer (XYZ, PDB, LAMMPS DATA)
- [x] Strengthen typing, doctests, and basic documentation

### Modeling & Construction
- [x] General lattice + crystal builder (SC/BCC/FCC/HCP + regions)
- [x] Basic molecular builders (monomers, ports, polymer assembly)
- [x] System assembly utilities (packing, placement, initialization)
- [x] Core editing tools (add/remove atoms, merge fragments, simple reactions)

### Force Fields & Typing
- [x] Force field container (atom types + bonded parameters)
- [x] Typifier system (rule-based SMARTS-style typing)
- [x] Parameter assignment and export for external engines
- [x] Validation / consistency tools for typed systems

### Compute & Analysis
- [ ] Unified `Compute` + `Result` abstraction
- [ ] Common analysis functions (RDF, MSD, basic time series)
- [ ] Optional shared context (neighbor lists, box info)
- [ ] Export analysis results to arrays / DataFrames

### Performance & User Experience
- [ ] Rust backend for performance-critical operations
- [ ] Visualization integration with molvis

</div>

---

## Ecosystem

**[MolVis](https://github.com/MolCrafts/molvis)** — Production-level visualization with WebGL acceleration and real-time manipulation
**[MolRS](https://github.com/MolCrafts/molrs)** — Rust backend for performance-critical operations

---

## Dependencies

- [numpy](https://github.com/numpy/numpy) — Numerical computing
- [python-igraph](https://github.com/igraph/python-igraph) — Graph analysis
- [lark](https://github.com/lark-parser/lark) — SMARTS/SMILES parsing

---

## License

This project is licensed under the BSD-3-Clause License. See [LICENSE](https://github.com/MolCrafts/molpy/blob/master/LICENSE) for details.
