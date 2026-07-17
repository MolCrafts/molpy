# Guides

Each guide takes one concrete modeling task from input to simulation-ready output — the *how-to* layer of the manual. The [Example Gallery](../getting-started/examples.md) holds copy-paste short forms of several of these; the guides are the full story. When a term is unfamiliar, the [data-model tutorials](../tutorials/index.md) are where it is defined.

Several guides use polymers as the working example — a demonstration domain, not a statement of scope. Chain growth, crosslinking, and polydispersity exercise every part of MolPy's editing machinery; the same operations apply to any complex molecular system.


## Foundations

- [Parsing Chemistry](01_parsing_chemistry.md) — conversion of SMILES, SMARTS, BigSMILES, and CGSmiles strings into `Atomistic` structures


## Chain & Network Construction

- [Assembly](02_assembly.md) — one `GraphAssembler` grows chains, crosslinks melts, and closes rings; only the `Selector` that pairs the reaction sites differs
- [Building a Crosslinked Gel](16_crosslinked_gel.md) — offline end-to-end workflow: GAFF chain → grid melt → crosslink → LAMMPS equilibration → network connectivity analysis
- [Polydisperse Systems](05_polydisperse_systems.md) — molecular-weight distribution sampling, atomistic chain construction, and box packing


## Polymer Topologies

From one ethylene-glycol kit to every architecture the assembly stack supports. Each page pairs with `examples/topology/<same-name>.py`.

- [**Section home**](topology/index.md) — three decisions, SMARTS, kit, ruled vs statistical
- [Linear](topology/01_linear.md) · [Block](topology/02_block.md) · [Ring](topology/03_ring.md) · [Star](topology/04_star.md) · [Comb](topology/05_comb.md) · [Telechelic](topology/06_telechelic.md)
- [Exhaustive gel](topology/07_gel_exhaustive.md) · [Random gel](topology/08_gel_random.md) · [End-linked](topology/09_end_linked.md) · [Dual network](topology/10_dual_network.md) · [Prepolymer + agent](topology/11_prepolymer_agent.md)

```bash
cd examples && python topology/run_all.py
```


## Parameterization

- [Force Field Typification](06_typifier.md) — SMARTS-based atom type assignment and force field parameter resolution


## Geometry & Packing

- [Nanostructures](04_nanostructures.md) — zigzag, armchair, and chiral carbon nanotubes with open or periodic axial topology
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
