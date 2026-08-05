# API Reference

Auto-generated reference for every public symbol, with typed signatures throughout. Start from the tables below: find your task, get the symbol and its package.

## Index of Operations and Symbols

| Operation | Primary symbols | Package |
|-----------|----------------|---------|
| Construct a molecule from atoms and bonds | `Atomistic`, `def_atom`, `def_bond` | [Core](core.md) |
| Store tabular molecular data | `Block`, `Frame` | [Core](core.md) |
| Define a periodic simulation cell | `Box` | [Core](core.md) |
| Represent a time-ordered frame sequence | `Trajectory` | [Core](core.md) |
| Perceive angles/dihedrals in place; k-hop bond-graph queries | `get_topo`, `get_topo_neighbors`, `get_topo_distances` | [Core](core.md) |
| Define and query force field parameters | `ForceField`, `Style`, `Type` | [Core](core.md) |
| Parse SMILES / SMARTS | `mp.io.read_smiles`, `SmilesIR`, `SmartsPattern` | [Parser](parser.md) |
| Perceive hydrogens / aromaticity / rings | `Perceive`, `RingInfo` | [Core](core.md) |
| Execute a chemical reaction (bond formation / removal) | `Reaction`, `GraphAssembler`, `Selector` | [Builder](builder.md) |
| Generate `fix bond/react` pre/post topology templates | `BondReactTemplate`, `write_bond_react_map` | [IO](io.md) |
| Assemble polymer chains | `MonomerLibrary`, `PolymerBuilder`, `Selector`, `Placer` | [Builder](builder.md) |
| Pack molecules into a simulation box | `molpack.Molpack`, `Target`, restraints | [Pack](pack.md) |
| Generate 3D conformers from a molecular graph | `Conformer` | [Conformer](conformer.md) |
| Assign force field atom types via SMARTS matching | `OPLSAATypifier`, `ClpTypifier`, `MMFFTypifier` | [Typifier](typifier.md) |
| Evaluate bond, angle, and pair potentials | `BondHarmonicStyle`, `LJ126Style`, `Potentials` | [Potential](potential.md) |
| Read and write molecular files (PDB, LAMMPS, GRO, …) | `read_pdb`, `write_lammps_data`, `read_xml_forcefield` | [I/O](io.md) |
| Bridge to a third-party library (in-memory) | `Adapter`, `RDKitAdapter` (optional example) | [Adapter](adapter.md) |
| Invoke external CLI tools (antechamber, tleap) | `Wrapper`, `AntechamberWrapper` | [Wrapper](wrapper.md) |
| Build polymer chains and crosslinked networks | `PolymerBuilder`, `GraphAssembler`, `Conformer` | [Builder](builder.md) |
| Compute mean-squared displacement, correlations, RDF, clustering | `MSD`, `Onsager`, `RDF` | [Compute](compute.md) |
| Locate bundled data files and built-in force fields | `get_forcefield_path`, `get_path` | [Data](data.md) |
| Generate LAMMPS, CP2K, or OpenMM input decks | `LAMMPSEngine`, `CP2KEngine`, `OpenMMEngine` | [Engine](engine.md) |

## Package Responsibilities

| Package | Responsibility |
|---------|---------------|
| [Core](core.md) | Foundational data structures: `Atomistic`, `Frame`, `Block`, `Box`, `Trajectory`, `Entity`/`Link`, `Region`, `UnitSystem`, `ForceField` |
| [Parser](parser.md) | SMILES / SMARTS (`SmilesIR`, `SmartsPattern`); moltemplate |
| [Builder](builder.md) | Polymer system construction: builders, port connectors, geometric placers |
| [Pack](pack.md) | Spatial packing via molpack (`molcrafts-molpack`) |
| [Conformer](conformer.md) | 3D conformer generation from molecular graphs |
| [Typifier](typifier.md) | Atom typing for OPLS-AA, CL&P, and MMFF (GAFF via AmberTools wrappers) |
| [Potential](potential.md) | Numerical potential kernels for bonds, angles, dihedrals, and non-bonded interactions |
| [I/O](io.md) | Format-specific readers and writers for molecular data, force fields, and trajectories |
| [Adapter](adapter.md) | Optional in-memory bridge to RDKit (worked example) |
| [Wrapper](wrapper.md) | Subprocess interfaces for AmberTools command-line executables |
| [Engine](engine.md) | Simulation engine abstractions for LAMMPS, CP2K, OpenMM |
| [Optimization](optimize.md) | Potential wrappers for geometry optimization workflows |
| [Compute](compute.md) | Trajectory analysis: MSD, Onsager, transport, dielectric, RDF, clustering, … |
| [Data](data.md) | Locators for bundled data files and built-in force fields |
