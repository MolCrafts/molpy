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

**Start here if:** You need to access data from chemical files and analyze results.

**What you'll learn:**
```python
frame = mp.io.read_lammps_data("data.lmp")
frame["atoms"]["xyz"]
frame["bonds"]["type"]
```

---

#### [Molecular Graph](molecular-graph.ipynb)

**Start here if:** You need create or edit molecule.

**What you'll learn:**

```python
atoms = mp.Atomistic()
C = atoms.def_atom(symbol="C", xyz=[0, 0, 0])
O = atoms.def_atom(symbol="O", xyz=[1.2, 0, 0])
atoms.def_bond(C, O, order=2)
```

---

#### [Box](box.ipynb)

**Start here if:** You need to define simulation boundaries or work with periodic systems.

**Key concepts:** Cell dimensions, PBC wrapping, lattice vectors

```python
box = mp.Box.cubic(length=10.0, origin=[0, 0, 0], pbc=[True, True, False])
```

---

#### [Topology](topology.ipynb)

**Start here if:** You're working with abstract molecular graph or graph algorithms.

```
from igraph import Graph
topo: Graph = mp.Atomistic().get_topology()
```
---

#### [Force Field](force-field.ipynb)

**Start here if:** You're manupulating force field parameters

```python
ff = mp.ForceField()
atype = ff.def_style(mp.AtomStyle("full"))
atype.def_type("C", mass=12.01, charge=0.0)

```

#### [Trajectory](trajectory.ipynb)

**Start here if:** You need to analyze simulation results or process large trajectory files.

```python
traj = mp.Trajectory.read_lammps_trajectory("traj.lammpstrj")
for frame in traj:
    print(frame.metadata["temperature"])
```

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

### 🔧 Advanced Features

Extend MolPy with custom functionality.

#### [Wrappers](wrappers.ipynb)

**Understand MolPy's wrapper system**

- Wrapper pattern for external libraries
- RDKit integration example
- Creating custom wrappers

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
1. Frame & Block -> Understand data structures
2. Box -> Learn about simulation cells
3. Topology -> Master molecular connectivity
4. Force Field -> Prepare for simulation
```

---

### Path 2: Simulation Setup
**Goal:** Prepare systems for MD simulations

```
1. Box -> Define simulation boundaries
2. Crystal Builder OR import structure
3. Molecular Graph -> Edit molecule manually
4. Force Field -> Assign types
5. IO modules -> Export for simulation
```

---

### Path 3: Trajectory Analysis
**Goal:** Analyze simulation results

```
1. Frame & Block -> Understand data format
2. Trajectory -> Learn trajectory handling
3. Analysis with Compute module
```

---

## Running the Tutorials

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

## Need Help?

- 💬 **[GitHub Discussions](https://github.com/MolCrafts/molpy/discussions)** – Ask questions
- 🐛 **[Issues](https://github.com/MolCrafts/molpy/issues)** – Report problems
- 📧 **Email:** support@molcrafts.org
- ❓ **[FAQ](../getting-started/faq.md)** – Common questions

---

Happy learning! 🧪✨
