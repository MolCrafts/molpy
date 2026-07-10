# API Reference

Auto-generated reference for every public symbol, with typed signatures throughout. Start from the tables below: find your task, get the symbol and its package.


## Index of Operations and Symbols

| Operation | Primary symbols | Package |
|-----------|----------------|---------|
| Construct a molecule from atoms and bonds | `Atomistic`, `def_atom`, `def_bond` | [Core](core.md) |
| Store tabular molecular data | `Block`, `Frame` | [Core](core.md) |
| Define a periodic simulation cell | `Box` | [Core](core.md) |
| Represent a time-ordered frame sequence | `Trajectory` | [Core](core.md) |
| Query bond-graph relations (angles, paths, rings) | `get_topo`, `get_topo_neighbors`, `get_topo_distances` | [Core](core.md) |
| Define and query force field parameters | `AtomisticForcefield`, `Style`, `Type` | [Core](core.md) |
| Parse SMILES / BigSMILES / SMARTS / CGSmiles | `parse_molecule`, `parse_monomer`, `parse_smarts`, `parse_cgsmiles` | [Parser](parser.md) |
| Execute a chemical reaction (bond formation / removal) | `Reaction`, `GraphAssembler`, `Selector` | [Builder](builder.md) |
| Generate `fix bond/react` pre/post topology templates | `BondReactTemplate`, `write_bond_react_map` | [IO](io.md) |
| Assemble polymer chains from CGSmiles | `PolymerBuilder`, `Selector`, `Placer` | [Builder](builder.md) |
| Pack molecules into a periodic simulation box | `Packmol`, `InsideBoxConstraint` | [Pack](pack.md) |
| Generate 3D conformers from a molecular graph | `Conformer` | [Conformer](conformer.md) |
| Assign force field atom types via SMARTS matching | `OPLSAATypifier`, `ClpTypifier`, `MMFFTypifier` | [Typifier](typifier.md) |
| Evaluate bond, angle, and pair potentials | `BondHarmonicStyle`, `LJ126Style`, `Potentials` | [Potential](potential.md) |
| Read and write molecular files (PDB, LAMMPS, GRO, …) | `read_pdb`, `write_lammps_data`, `read_xml_forcefield` | [I/O](io.md) |
| Convert between MolPy and RDKit / OpenBabel objects | `RDKitAdapter`, `OpenBabelAdapter` | [Adapter](adapter.md) |
| Invoke external CLI tools (antechamber, tleap) | `Wrapper`, `AntechamberWrapper` | [Wrapper](wrapper.md) |
| Build polymer chains and crosslinked networks | `PolymerBuilder`, `GraphAssembler`, `Conformer` | [Builder](builder.md) |
| Compute mean-squared displacement, correlations, RDF, clustering | `MSD`, `MCDCompute`, `RDF` | [Compute](compute.md) |
| Locate bundled data files and built-in force fields | `get_forcefield_path`, `get_path` | [Data](data.md) |
| Generate LAMMPS, CP2K, or OpenMM input decks | `LAMMPSEngine`, `CP2KEngine`, `OpenMMEngine` | [Engine](engine.md) |


## Package Responsibilities

| Package | Responsibility |
|---------|---------------|
| [Core](core.md) | Foundational data structures: `Atomistic`, `Frame`, `Block`, `Box`, `Trajectory`, `Entity`/`Link`, `Region`, `UnitSystem`, `ForceField` |
| [Parser](parser.md) | Grammar-based notation parsing: SMILES, SMARTS, BigSMILES, CGSmiles, G-BigSMILES |
| [Builder](builder.md) | Polymer system construction: builders, port connectors, geometric placers |
| [Pack](pack.md) | Spatial packing of molecular ensembles via the Packmol executable |
| [Conformer](conformer.md) | 3D conformer generation from molecular graphs (molrs backend) |
| [Typifier](typifier.md) | SMARTS-based atom typing for OPLS-AA, CL&P, and MMFF force fields (GAFF atom types come from the AmberTools wrappers) |
| [Potential](potential.md) | Numerical potential kernels for bonds, angles, dihedrals, and non-bonded interactions |
| [I/O](io.md) | Format-specific readers and writers for molecular data, force fields, and trajectories |
| [Adapter](adapter.md) | In-memory representation bridges to RDKit and OpenBabel |
| [Wrapper](wrapper.md) | Subprocess interfaces for AmberTools command-line executables |
| [Engine](engine.md) | Simulation engine abstractions for LAMMPS and CP2K |
| [Optimization](optimize.md) | Potential wrappers for geometry optimization workflows |
| [Compute](compute.md) | Trajectory analysis: MSD, displacement correlation, RDF, clustering, and custom compute operators |
| [Data](data.md) | Locators for bundled data files and built-in force fields |
