# Guides

Each guide in this section addresses a concrete modeling task from input specification to simulation-ready output. The notebooks are self-contained and executable; all required dependencies are specified within each notebook. Readers are assumed to be familiar with MolPy's core data model, as described in [Concepts](../tutorials/index.md).


## Foundational Subsystems

- [Tool Layer](tools.md) — packaged multi-step recipes (`PrepareMonomer`, `polymer`, `polymer_system`) for common preparation workflows
- [I/O Subsystem](io.md) — reading, writing, and extending file format support for molecular data, trajectories, and force fields
- [Chemistry Notation Parsing](01_parsing_chemistry.ipynb) — conversion of SMILES, SMARTS, BigSMILES, and CGSmiles strings into `Atomistic` structures


## Polymer Construction Workflows

- [Stepwise Chain Construction](02_polymer_stepwise.ipynb) — explicit reaction-based monomer coupling, the `PolymerBuilder` interface, and high-level facade functions
- [Topology-Driven Assembly](03_polymer_topology.ipynb) — specification of linear, cyclic, and branched polymer architectures via CGSmiles expressions
- [Crosslinked Network Generation](04_crosslinking.ipynb) — template-based network formation and pre/post topology generation for LAMMPS `fix bond/react`
- [Polydisperse System Construction](05_polydisperse_systems.ipynb) — molecular-weight distribution sampling, atomistic chain construction, and box packing


## Parameterization and External Tool Integration

- [Force Field Typification](06_typifier.ipynb) — SMARTS-based atom type assignment and force field parameter resolution
- [PEO–LiTFSI Electrolyte via AmberTools](07_ambertools_integration.ipynb) — a complete polymer electrolyte system preparation workflow using the AmberTools integration
- [Running MD Engines](engine.md) — generating input scripts and running LAMMPS, CP2K, and OpenMM directly from Python
