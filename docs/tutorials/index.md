# Tutorials

Welcome to the MolPy Tutorials! These **hands-on, example-driven guides** help you learn MolPy by doing. Each tutorial is a Jupyter notebook you can run interactively to master specific concepts and workflows.

## How to Use These Tutorials

**Learning path:**
* 📘 **Foundational** – Start here if you're new to MolPy
* 🔧 **Practical** – Learn by running real examples
* 🎯 **Task-focused** – Each tutorial solves a specific problem
* 🔗 **Progressive** – Tutorials build on each other

**Prerequisites:** Basic Python knowledge and familiarity with molecular modeling concepts.

---

## Getting Started

If you haven't already:
1. 📦 **[Install MolPy](../getting-started/installation.ipynb)** – Set up your environment
2. 🚀 **[Quickstart Guide](../getting-started/quickstart.ipynb)** – 5-minute introduction
3. 📚 **[Core Concepts](../getting-started/core-concepts.ipynb)** – Understand the data model

---

## Tutorial Categories

### 🏗️ Core Data Structures

Master MolPy's fundamental building blocks before diving into advanced features.

#### [Frame & Block](frame-block.ipynb)
**Learn MolPy's tabular data system**

- Creating and manipulating `Frame` objects
- Working with `Block` columns (coordinates, velocities, properties)
- Converting between data formats
- Understanding the Frame/Block architecture

**Start here if:** You're new to MolPy and want to understand how data is organized.

**What you'll learn:**
```python
frame = Frame(...)
frame.block.xyz  # Access coordinates
frame.block["mass"]  # Access properties
```

---

#### [Box](box.ipynb)
**Work with simulation cells and periodic boundaries**

- Creating simulation boxes (orthorhombic, triclinic)
- Setting periodic boundary conditions (PBC)
- Box transformations and resizing
- Understanding box vectors and lattice parameters

**Start here if:** You need to define simulation boundaries or work with periodic systems.

**Key concepts:** Cell dimensions, PBC wrapping, lattice vectors

---

#### [Topology](topology.ipynb)
**Understand molecular connectivity and bonding**

- Building molecular graphs (atoms, bonds)
- Topology representation in MolPy
- Bond orders and molecular structure
- Integration with `Atomistic` objects

**Start here if:** You're working with molecular graphs or need to understand bonding information.

**Applications:** Force field assignment, reaction modeling, structure analysis

---

#### [Molecular Graph Model](molecular-graph-model.ipynb)
**Deep dive into graph-based molecular representation**

- Graph theory applied to molecules
- Node and edge attributes
- Traversing molecular graphs
- Substructure matching

**Start here if:** You need advanced graph operations or are implementing custom algorithms.

**Advanced topics:** Graph algorithms, pattern matching, molecular descriptors

---

### 🛠️ Building & Construction

Learn to construct molecular systems from scratch.

#### [Crystal Builder](crystal-builder.ipynb)
**Generate crystal structures and lattices**

- Building crystal unit cells
- Common lattice types (FCC, BCC, HCP, diamond)
- Supercell generation
- Custom lattice parameters

**Start here if:** You're working with crystalline materials or periodic systems.

**Example use cases:**
- Metal nanoparticles
- Mineral surfaces
- Semiconductor structures

**Quick example:**
```python
from molpy.builder.crystal import fcc
crystal = fcc(element="Cu", a=3.61, n_cells=(3, 3, 3))
```

---

### ⚗️ Chemistry & Reactions

Work with chemical structures and transformations.

#### [Selector](selector.ipynb)
**Select and filter molecular substructures**

- Pattern-based selection (SMARTS)
- Atom and bond selectors
- Combining selection criteria
- Integration with reactions

**Start here if:** You need to identify specific atoms or groups for reactions or analysis.

**Use cases:**
- Finding reactive sites
- Selecting functional groups
- Filtering by properties

**Pattern:**
```python
selector = SMARTSSelector("[OH]")  # Find hydroxyl groups
selected = selector(atomistic)
```

---

#### [Force Field](force-field.ipynb)
**Assign force field parameters and atom types**

- Force field typification workflow
- OPLS-AA, AMBER typing
- Custom typing rules
- Validation and troubleshooting

**Start here if:** You're preparing structures for molecular dynamics simulations.

**Complete workflow:**
1. Build or import structure
2. Assign atom types with typifier
3. Validate assignments
4. Export to simulation format

**Integration:** Works with Builder, Reacter, IO modules

---

### 📊 Analysis & Trajectories

Analyze molecular dynamics results and time-series data.

#### [Trajectory](trajectory.ipynb)
**Work with MD trajectory files**

- Reading trajectory formats (LAMMPS, GROMACS, XYZ)
- Memory-efficient streaming
- Frame-by-frame analysis
- Trajectory manipulation and export

**Start here if:** You're analyzing simulation results or processing large trajectory files.

**Key features:**
- Lazy loading (memory efficient)
- Format conversion
- Property calculation over time
- Integration with Compute module

**Example workflow:**
```python
for frame in trajectory:
    rg = compute_radius_of_gyration(frame)
    results.append(rg)
```

---

### 🔧 Advanced Features

Extend MolPy with custom functionality.

#### [Wrappers](wrappers.ipynb)
**Understand MolPy's wrapper system**

- Wrapper pattern for external libraries
- RDKit integration example
- Creating custom wrappers
- Bidirectional data conversion

**Start here if:** You're integrating external tools or extending MolPy's capabilities.

**Key concept:** Wrappers provide a consistent interface to external libraries while preserving MolPy's data model.

**Example integrations:**
- RDKit for cheminformatics
- OpenBabel for format conversion
- Custom molecular mechanics engines

---

## Learning Paths

### Path 1: Complete Beginner
**Goal:** Learn MolPy from scratch

```
1. Frame & Block → Understand data structures
2. Box → Learn about simulation cells
3. Topology → Master molecular connectivity
4. Crystal Builder → Create first structure
5. Force Field → Prepare for simulation
```

**Time:** ~3-4 hours | **Level:** Beginner

---

### Path 2: Simulation Setup
**Goal:** Prepare systems for MD simulations

```
1. Box → Define simulation boundaries
2. Crystal Builder OR import structure
3. Force Field → Assign types
4. Export to LAMMPS/GROMACS (see User Guide: IO)
```

**Time:** ~2 hours | **Level:** Intermediate

---

### Path 3: Trajectory Analysis
**Goal:** Analyze simulation results

```
1. Frame & Block → Understand data format
2. Trajectory → Learn trajectory handling
3. Analysis with Compute module (see User Guide)
```

**Time:** ~2 hours | **Level:** Intermediate

---

### Path 4: Advanced Customization
**Goal:** Extend MolPy for custom needs

```
1. Molecular Graph Model → Graph operations
2. Selector → Pattern matching
3. Wrappers → External integrations
4. Custom modules (see Developer Guide)
```

**Time:** ~4-5 hours | **Level:** Advanced

---

## Tutorial Quick Reference

| Tutorial | Duration | Difficulty | Prerequisites |
|----------|----------|------------|---------------|
| [Frame & Block](frame-block.ipynb) | 30 min | ⭐ Beginner | None |
| [Box](box.ipynb) | 20 min | ⭐ Beginner | Frame & Block |
| [Topology](topology.ipynb) | 30 min | ⭐⭐ Intermediate | Frame & Block |
| [Molecular Graph Model](molecular-graph-model.ipynb) | 45 min | ⭐⭐⭐ Advanced | Topology |
| [Crystal Builder](crystal-builder.ipynb) | 30 min | ⭐⭐ Intermediate | Frame & Block, Box |
| [Selector](selector.ipynb) | 25 min | ⭐⭐ Intermediate | Topology |
| [Force Field](force-field.ipynb) | 40 min | ⭐⭐ Intermediate | Topology |
| [Trajectory](trajectory.ipynb) | 35 min | ⭐⭐ Intermediate | Frame & Block |
| [Wrappers](wrappers.ipynb) | 30 min | ⭐⭐⭐ Advanced | Core concepts |

---

## Running the Tutorials

### Interactive (Recommended)

**Local:**
```bash
git clone https://github.com/MolCrafts/molpy.git
cd molpy/docs/tutorials
jupyter notebook
```

**Google Colab:**
Click the "Open in Colab" badge at the top of each tutorial.

### Read-Only

Browse tutorials directly on the [documentation site](https://molcrafts.github.io/molpy/tutorials/) – great for quick reference!

---

## What's Next?

**After completing tutorials:**

1. 📖 **[User Guide](../user-guide/index.md)** – Comprehensive module documentation
   - Deeper coverage of each module
   - Production workflow patterns
   - Best practices and optimization tips

2. 🔬 **[API Reference](../api/index.md)** – Complete function documentation
   - Full API specifications
   - Parameter details
   - Return value documentation

3. 🛠️ **[Developer Guide](../developer/index.md)** – Contribute to MolPy
   - Development setup
   - Coding standards
   - Creating custom modules

---

## Tips for Learning

💡 **Best practices:**
- Run notebooks interactively – don't just read!
- Experiment with parameters and see what happens
- Check the User Guide for deeper explanations
- Join our [Discussions](https://github.com/MolCrafts/molpy/discussions) to ask questions

⚠️ **Common beginner mistakes:**
- Skipping Frame & Block (start here!)
- Not understanding Frame vs Atomistic distinction
- Forgetting to typify before simulation export
- Mixing data structures without conversion

---

## Need Help?

- 💬 **[GitHub Discussions](https://github.com/MolCrafts/molpy/discussions)** – Ask questions
- 🐛 **[Issues](https://github.com/MolCrafts/molpy/issues)** – Report problems
- 📧 **Email:** support@molcrafts.org
- ❓ **[FAQ](../getting-started/faq.md)** – Common questions

---

Happy learning! 🧪✨
