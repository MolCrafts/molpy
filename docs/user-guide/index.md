# User Guide

Welcome to the MolPy User Guide! Here you will find practical workflows to help you accomplish your research tasks.

## 🧪 [Parsing Chemistry](01_parsing_chemistry.ipynb)
**Goal:** Define molecules using strict syntax instead of manual construction.
- Use **BigSMILES** for monomers with reactive ports.
- Use **CGSmiles** for coarse-grained topology.
- Use **GBigSMILES** for systems with distributions.

```python
from molpy.parser.smiles import parse_bigsmiles
# Parse a stochastic object directly from string
molecule_ir = parse_bigsmiles("{[<]CC[>]}[$]CO[$]")
```

## 🏗️ [Building Polymers](02_polymer_stepwise.ipynb)
**Goal:** Assemble complex polymer architectures from monomer libraries.
- Build linear, block, or star polymers.
- Automatically connect reactive ports using graph rules.

```python
# Simple linear polymer construction
builder = PolymerBuilder(library=my_monomers, connector=my_connector)
result = builder.build("{[#EO2][#PS]}")
polymer = result.polymer
```

## 🔗 [Polymer SMILES](03_polymer_smiles.ipynb)
**Goal:** Program chemical reactivity and mechanisms.
- Define custom reactions with **Selectors**.
- Execute reactions on specific ports or atoms.

```python
# Run a specific reaction on two fragments
cc_coupling = Reacter(
    name="C-C",
    bond_former=form_single_bond,
    # ... selectors ...
)
result = cc_coupling.run(fragments[0], fragments[1], port_atom_L=p1, port_atom_R=p2)
```

## 🕸️ [Polymer Crosslinking](04_polymer_crosslinking.ipynb)
**Goal:** Create 3D cross-linked networks and simulation templates.
- Generate LAMMPS `fix bond/react` templates.
- Build explicitly cross-linked networks in memory.

## 📊 [Polydisperse Systems](05_polymer_polydisperse.ipynb)
**Goal:** Model realistic material distributions and ensembles.
- Generate chains following Poisson or Schulz-Zimm distributions.
- Plan systems with target total mass and compositions.

---

## Need API Basics?

If you are looking for detailed explanations of core classes (like `Atomistic`, `Frame`, `Topology`), check out the **[Tutorials](../tutorials/index.md)** section.
