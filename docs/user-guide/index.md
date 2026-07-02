# Guides

Each guide in this section addresses a concrete modeling task, from input specification to simulation-ready output. Readers are assumed to be familiar with MolPy's core data model, as described in [Concepts](../tutorials/index.md).

Several guides use polymers as the working example. That is a choice of demonstration domain, not a statement of scope: chain growth, crosslinking, and polydispersity exercise every part of MolPy's molecular editing machinery — reaction-based coupling, topology regeneration, statistical ensembles. The same operations apply to any complex molecular system.


## Foundations

- [Parsing Chemistry](01_parsing_chemistry.md) — conversion of SMILES, SMARTS, BigSMILES, and CGSmiles strings into `Atomistic` structures


## Chain & Network Construction

- [Stepwise Construction](02_polymer_stepwise.md) — explicit reaction-based monomer coupling, the `PolymerBuilder` interface, and high-level facade functions
- [Topology-Driven Assembly](03_polymer_topology.md) — specification of linear, cyclic, and branched architectures via CGSmiles expressions
- [Crosslinked Networks](04_crosslinking.md) — template-based network formation and pre/post topology generation for LAMMPS `fix bond/react`
- [Polydisperse Systems](05_polydisperse_systems.md) — molecular-weight distribution sampling, atomistic chain construction, and box packing


## Parameterization

- [Force Field Typification](06_typifier.md) — SMARTS-based atom type assignment and force field parameter resolution


## Geometry & Packing

- [3D Conformer Generation](07_conformers.md) — embedding chemically valid 3D coordinates for a parsed or constructed structure
- [Geometry Optimization](08_geometry_optimization.md) — force-field-driven structure minimization and how to read the optimization report
- [Packing Systems](09_packing.md) — filling a simulation cell with molecules under geometric constraints via the Packmol backend
- [Polarizable & Virtual-Site Models](10_polarizable.md) — Drude shells and TIP4P M-sites through the virtual-site builder protocol


## Export & Engines

- [File I/O](11_io.md) — reading and writing molecular data, trajectories, log files, and force-field formats
- [Simulation Engines](12_engine.md) — generating input decks for LAMMPS, CP2K, and OpenMM, and running them from Python


## Tools & Ecosystem

- [AmberTools Integration](13_ambertools_integration.md) — a complete electrolyte preparation workflow driving antechamber, parmchk2, and tleap
- [Moltemplate CLI](14_moltemplate_cli.md) — converting moltemplate `.lt` files to MolPy systems and back
- [MCP Suite](15_mcp.md) — exposing MolPy symbols and docs to Model Context Protocol agents
