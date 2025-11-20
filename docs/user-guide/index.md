# User Guide

Welcome to the MolPy User Guide! Whether you're building polymers, setting up simulations, or analyzing molecular dynamics trajectories, this guide will help you master MolPy's modular toolkit.

## What You'll Find Here

This guide provides **in-depth documentation** for each of MolPy's modules. Unlike the quick-start tutorials, these pages dive deep into capabilities, design patterns, and best practices for production workflows.

**How to use this guide:**
- 📖 **Reference-style** – Look up specific modules when you need them
- 🔗 **Interconnected** – Each module links to related concepts
- 💻 **Code-first** – Every page includes working examples
- 🎯 **Practical** – Focused on real-world molecular modeling tasks

---

## Core Data Structures

Before diving into specific modules, familiarize yourself with MolPy's core data model:

| Structure | Purpose | When to Use |
|-----------|---------|-------------|
| **`Frame` + `Block`** | Tabular data (coordinates, properties) | File I/O, simulation export, analysis |
| **`Atomistic`** | Graph-based molecules (atoms, bonds) | Building, reactions, force field typing |
| **`Box`** | Simulation cell + PBC | Defining system boundaries |
| **`Trajectory`** | Time series of frames | MD analysis, streaming large files |

> 💡 **Key insight**: MolPy uses `Atomistic` for *chemistry* (bonds, reactions) and `Frame` for *geometry* (coordinates, simulation setup). Most workflows involve converting between them.

---

## Module Overview

### 🏗️ Building & Construction

#### [Builder](builder.ipynb)
**Build polymers and assemble molecular systems**

- Linear polymer assembly from monomer sequences
- Crystal structure generation (FCC, BCC, HCP lattices)
- Integration with reaction-based workflows

**Start here if:** You're constructing polymers, crystals, or composite systems from scratch.

**Quick taste:**
```python
from molpy.builder.polymer.linear import linear
poly = linear(sequence="ABAB", library=monomers, connector=ReacterConnector())
```

---

#### [Pack](pack.ipynb)
**Distribute molecules spatially without overlaps**

- Packing algorithms (Packmol integration)
- Constraint-based placement (boxes, spheres, regions)
- Multi-component systems (solvation, polymer melts)

**Start here if:** You need to create simulation boxes with realistic initial configurations.

**Typical use:** Creating a box with 500 water molecules + 1 protein.

---

#### [Reacter](reacter.ipynb)
**Program chemical reactions like code**

- Define reactions with selectors + transformers
- Esterification, amide formation, C-C coupling
- Automatic retypification after bond changes
- Full audit trail in product metadata

**Start here if:** You're modeling polymerization, small molecule reactions, or bond manipulation.

**Example reaction:**
```python
esterification = Reacter(
    anchor_left=port_anchor_selector,
    leaving_left=remove_OH,
    anchor_right=port_anchor_selector,
    leaving_right=remove_one_H,
    bond_maker=make_single_bond
)
product = esterification.run(acid, alcohol, port_L="1", port_R="1")
```

---

### 🔧 Force Fields & Typing

#### [Typifier](typifier.ipynb)
**Assign force field atom types automatically**

- OPLS-AA, AMBER, custom force fields
- SMARTS pattern-based typing rules
- Retypification after reactions
- Validation and consistency checks

**Start here if:** You need to prepare structures for MD simulations.

**Workflow:**
```python
typifier = OplsAtomisticTypifier()
typifier.typify(atomistic)  # Assigns OPLS types
```

---

#### [Potential](potential.ipynb)
**Force field energy functions**

- Bond, angle, dihedral potentials
- Pair interactions (Lennard-Jones, Coulomb)
- Custom potential definitions

**Start here if:** You're implementing custom force fields or understanding energy calculations.

**Note:** Most users won't call these directly – they're used internally by simulation engines.

---

### 📁 File I/O & Interoperability

#### [IO](io.ipynb)
**Read/write molecular file formats**

- **Data files**: PDB, LAMMPS data, GRO, XYZ
- **Force fields**: LAMMPS scripts, XML, AMBER prmtop
- **Trajectories**: LAMMPS dump, XYZ multi-frame
- **Streaming**: Memory-efficient trajectory iteration

**Start here if:** You're importing structures or exporting to simulation engines.

**Supported formats:** PDB, LAMMPS, GROMACS, AMBER, OpenMM XML, XYZ, XSF, MOL2

---

#### [Adapter](adapter.ipynb)
**Bridge to external libraries (RDKit)**

- Bidirectional `Mol` ↔ `Atomistic` conversion
- SMILES to 3D coordinates
- 2D molecular drawings (SVG export)
- Integration with RDKit cheminformatics

**Start here if:** You're working with SMILES, need 3D coordinate generation, or using RDKit features.

**Common pattern:**
```python
wrapper = RDKitWrapper.from_mol(mol)
wrapper.generate_3d(optimize=True)
atomistic = wrapper.inner  # Now has 3D coordinates
```

---

### 📊 Analysis & Computation

#### [Compute](compute.ipynb)
**Unified framework for calculations**

- Standardized `Compute` → `Result` pattern
- Shareable context for expensive intermediates
- Type-safe calculation pipelines
- Trajectory analysis workflows

**Start here if:** You're analyzing MD trajectories or implementing custom analysis functions.

**Example:**
```python
compute = RadiusOfGyrationCompute()
for frame in trajectory:
    result = compute(frame)
    print(f"Rg = {result.rg:.2f} Å")
```

---

### 🧮 Geometric Operations

#### [Op](op.ipynb)
**Low-level coordinate manipulation**

- Rotation (Rodrigues, quaternions)
- Translation and alignment
- Matrix transformations

**Start here if:** You're implementing custom placement algorithms or molecular alignment.

**Typical use:** Orienting monomers during polymer assembly.

---

### 🔤 Parsing

#### [Parser](parser.ipynb)
**SMILES/SMARTS pattern parsing**

- SMILES molecules → `Atomistic`
- SMARTS patterns for atom typing
- BigSMILES notation (in development)

**Start here if:** You need native Python SMILES parsing (no RDKit) or are defining typing rules.

**Note:** For production SMILES work, prefer the RDKit adapter.

---

## Typical Workflows

### Workflow 1: Polymer Building → Simulation

```
1. [Builder] → Create polymer from monomers
2. [Reacter] → Polymerization reactions (if needed)
3. [Typifier] → Assign OPLS atom types
4. [Pack] → Create simulation box with multiple chains
5. [IO] → Export to LAMMPS data + force field files
```

**Relevant guides:** Builder, Reacter, Typifier, Pack, IO

---

### Workflow 2: SMILES → 3D → MD Setup

```
1. [Adapter] → SMILES → Atomistic with 3D coords
2. [Typifier] → Assign force field types
3. [IO] → Export to simulation format
```

**Relevant guides:** Adapter, Typifier, IO

---

### Workflow 3: Trajectory Analysis

```
1. [IO] → Stream trajectory frames
2. [Compute] → Calculate properties (RDF, MSD, Rg, etc.)
3. Post-process results
```

**Relevant guides:** IO, Compute

---

## Module Dependencies

Understanding how modules relate helps plan your workflow:

```
Core Data (Frame, Atomistic, Box)
        ↓
    ┌───┴────────┬─────────────┬──────────┐
Parser/Adapter  Builder    IO/Readers    Op
        ↓           ↓
    Reacter ←── Typifier
        ↓           ↓
    Builder     IO/Writers ──→ LAMMPS/GROMACS/OpenMM
        ↓
    Pack ──────────┘
        ↓
    IO/Writers
```

**Key takeaway:** Most workflows involve: Build/Import → Type → Pack → Export

---

## Best Practices

### 🎯 When to Use Which Module

| Goal | Start With | Then Use |
|------|-----------|----------|
| Build polymer from scratch | Builder | Typifier → IO |
| Import PDB, add solvent | IO → Pack | Typifier → IO |
| SMILES to simulation | Adapter | Typifier → IO |
| Analyze MD trajectory | IO (trajectory) | Compute |
| Custom force field | Typifier + Potential | IO |

### 💡 Common Patterns

**Pattern 1: Chemistry-first (Atomistic)**
```python
# Build using Atomistic (bonds matter)
mono = Monomer(...)
poly = linear(monomers)

# Convert to Frame for simulation
frame = atomistic_to_frame(poly)
```

**Pattern 2: Geometry-first (Frame)**
```python
# Start with coordinates (from file or generation)
frame = read_pdb("structure.pdb")

# Pack into larger system
packed = Molpack.pack([frame], n=100)
```

### ⚠️ Common Pitfalls

1. **Forgetting to typify before export** → Simulation engines need atom types
2. **Mixing Atomistic and Frame operations** → Convert explicitly when needed
3. **Not retypifying after reactions** → Bond changes invalidate old types
4. **Using separate x/y/z instead of xyz array** → PDB/IO expects `xyz` field

---

## Quick Links

### Getting Started
- 🚀 **[Quickstart Guide](../getting-started/quickstart.ipynb)** – 5-minute intro
- 📚 **[Core Concepts](../getting-started/core-concepts.ipynb)** – Data model deep dive
- ❓ **[FAQ](../getting-started/faq.md)** – Common questions

### Hands-On Learning
- 📖 **[Tutorials](../tutorials/index.md)** – Step-by-step examples
- 🔬 **[API Reference](../api/index.md)** – Complete function documentation

### Community
- 💬 **[Discussions](https://github.com/MolCrafts/molpy/discussions)** – Ask questions
- 🐛 **[Issues](https://github.com/MolCrafts/molpy/issues)** – Report bugs

---

## Next Steps

**New to MolPy?** Start with the [Quickstart](../getting-started/quickstart.ipynb) to get familiar with `Frame` and `Atomistic`.

**Ready to dive in?** Pick a module above that matches your task and explore the detailed guide.

**Building something specific?** Check the workflow diagrams above to plan your module sequence.

Happy modeling! 🧬
