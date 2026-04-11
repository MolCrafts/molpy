# API Reference

All public symbols are documented on the package pages listed below. Docstrings are auto-generated from source annotations; signatures include type hints throughout. The tables on this page provide a task-oriented index for locating the appropriate symbol and package without prior knowledge of MolPy's internal organization.


## Index of Operations and Symbols

| Operation | Primary symbols | Package |
|-----------|----------------|---------|
| Construct a molecule from atoms and bonds | `Atomistic`, `def_atom`, `def_bond` | [Core](core.md) |
| Store tabular molecular data | `Block`, `Frame` | [Core](core.md) |
| Define a periodic simulation cell | `Box` | [Core](core.md) |
| Represent a time-ordered frame sequence | `Trajectory` | [Core](core.md) |
| Query bond-graph relations (angles, paths, rings) | `Topology` | [Core](core.md) |
| Define and query force field parameters | `AtomisticForcefield`, `Style`, `Type` | [Core](core.md) |
| Parse SMILES / BigSMILES / SMARTS / CGSmiles | `parse_molecule`, `parse_monomer`, `parse_smarts`, `parse_cgsmiles` | [Parser](parser.md) |
| Execute a chemical reaction (bond formation / removal) | `Reacter`, `find_port`, `select_neighbor` | [Reacter](reacter.md) |
| Generate `fix bond/react` pre/post topology templates | `TemplateReacter` | [Reacter](reacter.md) |
| Assemble polymer chains from CGSmiles | `PolymerBuilder`, `Connector`, `Placer` | [Builder](builder.md) |
| Pack molecules into a periodic simulation box | `Molpack`, `InsideBoxConstraint` | [Pack](pack.md) |
| Assign force field atom types via SMARTS matching | `OplsAtomisticTypifier`, `GaffTypifier` | [Typifier](typifier.md) |
| Evaluate bond, angle, and pair potentials | `BondHarmonic`, `LJ126` | [Potential](potential.md) |
| Read and write molecular files (PDB, LAMMPS, GRO, …) | `read_pdb`, `write_lammps_data`, `read_xml_forcefield` | [I/O](io.md) |
| Convert between MolPy and RDKit / OpenBabel objects | `RDKitAdapter`, `OpenBabelAdapter` | [Adapter](adapter.md) |
| Invoke external CLI tools (antechamber, tleap) | `Wrapper`, `AntechamberWrapper` | [Wrapper](wrapper.md) |
| Execute packaged multi-step preparation workflows | `PrepareMonomer`, `polymer`, `generate_3d` | [Tool](tool.md) |
| Compute mean-squared displacement and correlations | `MSD`, `DisplacementCorrelation` | [Tool](tool.md) |
| Generate LAMMPS or CP2K input decks | `LAMMPSEngine`, `CP2KEngine` | [Engine](engine.md) |


## Package Responsibilities

| Package | Responsibility |
|---------|---------------|
| [Core](core.md) | Foundational data structures: `Atomistic`, `Frame`, `Block`, `Box`, `Trajectory`, `Topology`, `ForceField` |
| [Parser](parser.md) | Grammar-based notation parsing: SMILES, SMARTS, BigSMILES, CGSmiles, G-BigSMILES |
| [Reacter](reacter.md) | Chemical reaction framework: site and leaving-group selectors, bond formers, reactive MD templates |
| [Builder](builder.md) | Polymer system construction: builders, port connectors, geometric placers |
| [Pack](pack.md) | Spatial packing of molecular ensembles via the Packmol executable |
| [Typifier](typifier.md) | SMARTS-based atom typing for OPLS-AA and GAFF force fields |
| [Potential](potential.md) | Numerical potential kernels for bonds, angles, dihedrals, and non-bonded interactions |
| [I/O](io.md) | Format-specific readers and writers for molecular data, force fields, and trajectories |
| [Adapter](adapter.md) | In-memory representation bridges to RDKit and OpenBabel |
| [Wrapper](wrapper.md) | Subprocess interfaces for AmberTools command-line executables |
| [Tool](tool.md) | High-level multi-step preparation workflows and analysis operators |
| [Engine](engine.md) | Simulation engine abstractions for LAMMPS and CP2K |
| [Optimization](optimize.md) | Potential wrappers for geometry optimization workflows |
