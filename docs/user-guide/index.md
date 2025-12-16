# User Guide

Welcome to the MolPy User Guide! This section is organized around **tasks and workflows**. Each guide provides comprehensive examples to help you accomplish specific goals.

## Guides by Topic

### 🧪 [Parsing Chemistry](01_parsing_chemistry.ipynb)
**Goal:** Define molecules using syntax instead of manual construction.
-   **BigSMILES**: Define monomers with reactive ports.
-   **CGSmiles**: Define coarse-grained topology (blocks, branches).
-   **GBigSMILES**: Define systems with distributions.

### 🏗️ [Building Polymers](02_polymer_stepwise.ipynb)
**Goal:** Assemble complex polymer architectures.
-   **Linear Polymers**: Standard chain growth.
-   **Copolymers**: Block and random sequences.
-   **Complex Architectures**: Graft, star, and cross-linked polymers.

### 🔗 [Polymer SMILES](03_polymer_smiles.ipynb)
**Goal:** Program chemical reactivity.
-   **Manual Reactions**: Connecting specific molecules.
-   **Custom Mechanisms**: Defining new reaction rules with Selectors.
-   **Templates**: Generating reaction templates for MD engines (LAMMPS).

### 🔗 [Polymer Crosslinking](04_polymer_crosslinking.ipynb)
**Goal:** Create cross-linked polymer networks.
-   **Reaction Templates**: Defining crosslinking reactions.
-   **Network Formation**: Building 3D polymer networks.
-   **LAMMPS Integration**: Generating reaction templates for MD.

### 📊 [Polydisperse Systems](05_polymer_polydisperse.ipynb)
**Goal:** Model realistic material distributions.
-   **Distributions**: Schulz-Zimm, Poisson, Flory-Schulz.
-   **Ensembles**: Generating representative populations.
-   **Analysis**: Calculating molecular weight moments ($M_n$, $M_w$).

### ⚙️ [Simulation Preparation](06_simulation_preparation.ipynb)
**Goal:** Prepare a system for molecular dynamics.
-   **Packing**: Creating dense simulation boxes (solvation).
-   **Typifier**: Assigning force field parameters (OPLS-AA, GAFF).
-   **Optimization**: Minimizing energy to remove overlaps.
-   **Export**: Writing ready-to-run LAMMPS data files.



---

## Need API Basics?

If you are looking for detailed explanations of core classes (like `Atomistic`, `Frame`, `Topology`), check out the **[Tutorials](../tutorials/index.md)** section.
