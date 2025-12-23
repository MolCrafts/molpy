# Getting Started

Welcome to MolPy! This section will help you get up and running quickly. Whether you're new to molecular modeling or an experienced researcher, we'll guide you through the essentials.

## Learning Path

We recommend following these guides in order:

<div class="grid cards" markdown>

- :one:{ .lg .middle } __[Installation](installation.ipynb)__

    ---

    Get MolPy installed and verify your setup. Learn about dependencies, optional packages, and development installation.

- :two:{ .lg .middle } __[Quickstart](quickstart.ipynb)__

    ---

    Build your first molecular system in minutes. This hands-on tutorial covers the core workflow from parsing to export.

- :three:{ .lg .middle } __[Core Concepts](core-concepts.ipynb)__

    ---

    Understand MolPy's fundamental data structures: `Atomistic`, `Frame`, `Block`, and `Box`. Learn how topology and coordinates work together.

- :question:{ .lg .middle } __[FAQ](faq.md)__

    ---

    Common questions and answers. Troubleshooting tips and comparisons with other tools.

</div>

---

## What You'll Learn

### Installation

- Installing MolPy via pip
- Setting up a development environment
- Understanding dependencies and optional packages
- Verifying your installation

### Quickstart

- Parsing chemical notation (SMILES, BigSMILES)
- Building simple molecular structures
- Working with topology and coordinates
- Exporting to common formats (LAMMPS, PDB)

### Core Concepts

- **Atomistic**: The fundamental molecular structure representation
- **Frame**: Tabular data structure for coordinates and properties
- **Block**: Hierarchical composition of multiple structures
- **Box**: Periodic boundary conditions
- **Topology**: Bonds, angles, dihedrals, and their relationships

---

## Next Steps

Once you've completed the Getting Started guides, explore:

<div class="grid" markdown>

!!! tip "Tutorials"
    Deep dive into specific topics with runnable Jupyter notebooks in the [Tutorials](../tutorials/index.md) section.

!!! example "User Guide"
    Follow complete workflows for real-world tasks in the [User Guide](../user-guide/index.md):
    - Parsing chemistry notation
    - Building polymers step-by-step
    - Creating crosslinked networks
    - Working with polydisperse systems

!!! info "API Reference"
    Browse detailed API documentation in the [API Reference](../api/index.md) section.

</div>

---

## Prerequisites

Before you begin, make sure you have:

- **Python 3.12+** installed
- Basic familiarity with Python
- (Optional) Experience with molecular modeling or chemistry

No prior knowledge of MolPy is required—we'll teach you everything you need!

---

## Quick Links

<div class="grid" markdown>

!!! success "Ready to Start?"
    Begin with [Installation](installation.ipynb) to get MolPy set up on your system.

!!! question "Need Help?"
    Check the [FAQ](faq.md) for answers to common questions, or open an issue on [GitHub](https://github.com/MolCrafts/molpy/issues).

!!! info "Want Examples?"
    Jump to the [Quickstart](quickstart.ipynb) for hands-on examples you can run immediately.

</div>

---

## What Makes MolPy Different?

MolPy is designed with these principles in mind:

- **Type Safety**: Strong typing throughout for better IDE support and fewer runtime errors
- **Composability**: Build complex workflows by combining simple, reusable components
- **Explicit APIs**: No hidden magic—every operation is clear and documented
- **Interoperability**: Seamless integration with RDKit, LAMMPS, OpenMM, and more
- **AI-Friendly**: Predictable APIs that work well with LLM code generation

Ready to get started? Let's begin with [Installation](installation.ipynb)!
