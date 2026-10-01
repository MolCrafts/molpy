# Guides

Each guide takes one concrete modeling task from input to simulation-ready output — the *how-to* layer of the manual. The [Example Gallery](../getting-started/examples.md) holds copy-paste short forms of several of these; the guides are the full story. When a term is unfamiliar, the [data-model tutorials](../tutorials/index.md) are where it is defined.

Several guides use polymers as the working example — a demonstration domain, not a statement of scope. Chain growth, crosslinking, and polydispersity exercise every part of MolPy's editing machinery; the same operations apply to any complex molecular system.

## Foundations

- [Parsing Chemistry](01_parsing_chemistry.md) — conversion of SMILES and SMARTS strings into `Atomistic` structures (BigSMILES is no longer parsed; CGsmiles is covered in Assembly)

## Chain & Network Construction

- [Assembly](02_assembly.md) — units with ports joined along a site graph by `Assembler`: grown from a CGsmiles topology, or backmapped onto a coarse-grained model
- [Polydisperse Systems](05_polydisperse_systems.md) — molecular-weight distribution sampling, atomistic chain construction, and box packing

## Polymer Topologies

From one ethylene-oxide kit to every architecture the site-graph assembler supports. Each page pairs with `examples/topology/<same-name>.py`.

- [**Section home**](topology/index.md) — units, topology, assembly; port rules; the kit
- [Linear](topology/01_linear.md) · [Block](topology/02_block.md) · [Ring](topology/03_ring.md) · [Star](topology/04_star.md) · [Comb](topology/05_comb.md) · [Telechelic](topology/06_telechelic.md)

## Parameterization

- [Force Field Typification](06_typifier.md) — SMARTS-based atom type assignment and force field parameter resolution

## Geometry & Packing

- [Nanostructures](04_nanostructures.md) — zigzag, armchair, and chiral carbon nanotubes with open or periodic axial topology
- [3D Conformer Generation](07_conformers.md) — embedding chemically valid 3D coordinates for a parsed or constructed structure
- [Geometry Optimization](08_geometry_optimization.md) — force-field-driven structure minimization and how to read the optimization report
- [Packing Systems](09_packing.md) — filling a simulation cell with molecules under geometric restraints via molpack
- [Polarizable & Virtual-Site Models](10_polarizable.md) — Drude shells and TIP4P M-sites through the virtual-site builder protocol

## Export & Engines

- [File I/O](11_io.md) — reading and writing molecular data, trajectories, log files, and force-field formats
- [Simulation Engines](12_engine.md) — generating input decks for LAMMPS, CP2K, and OpenMM, and running them from Python

## Tools & Ecosystem

- [AmberTools Integration](13_ambertools_integration.md) — a complete electrolyte preparation workflow driving antechamber, parmchk2, and tleap
