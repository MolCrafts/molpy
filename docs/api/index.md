# API Reference

Auto-generated reference for every public symbol, with typed signatures throughout. Start from the tables below: find your task, get the symbol and its package.

## Public surface: one path per name

Write `import molpy as mp` and reach every name through `mp`. Each name has exactly
one public path — there is no `molpy.core.Atomistic`, `molpy.parser` or
`molpy.optimize` spelling. Two layers own the names:

| Layer | What it means | Names |
|-------|---------------|-------|
| **Re-exported from the native core** (molrs) | Identity re-export: `mp.Atomistic is molrs.Atomistic`. Behaviour and docs come from the native core; molpy adds nothing. | Graphs and views: `Atomistic`, `CoarseGrain`, `Atom`, `Bond`, `Angle`, `Dihedral`, `Improper`, `VirtualSite`, `DrudeParticle`, `MasslessSite`, `Bead`, `CGBond`, `NodeRef`, `RelationRef`, `Refs`, `Port`, `Graph`, `Topology`, `Trace`, `ExtractedSubgraph` · Data: `Frame`, `Block`, `FrameMeta`, `MetaDocument`, `MetaValue`, `Element`, `keys`, `schema` · Force field: `ForceField`, `Style` / `Type` and their `Atom`/`Bond`/`Angle`/`Dihedral`/`Improper`/`Pair` subclasses, `PotentialCompiler`, `Potentials`, `FragmentScaling`, `BccModel`, `GasteigerModel`, `MullikenModel` · Notation and perception: `SmilesIR`, `CGSmilesIR`, `SmilesError`, `SmartsPattern`, `SmartsMatch`, `Perceive`, `RingInfo`, `Reaction`, `Coarsener`, `SubgraphMatcher` · Assembly: `Assembler`, `GrowthPlacer`, `SitePlacer`, `AxisOrienter` · Shapes: `Cuboid`, `Sphere`, `Parallelepiped`, `HalfSpace` · Units: `Unit`, `Quantity`, `UnitRegistry`, `UnitPreset`, `UnitsError` · Neighbours: `NeighborQuery`, `NeighborList`, `Neighbors`, `VerletSkin` · Optimisation and conformer reports: `LBFGS`, `OptReport`, `ConformerReport`, `ConformerStageReport` · Namespaces `mp.md`, `mp.op` |
| **molpy's own** | Defined in molpy; *sub* marks a subclass of a native type that adds molpy behaviour. | `Box` (sub), `Trajectory` (sub), `TrajectorySplitter` and its strategies, `Region`, `BoxRegion` / `SphereRegion` / `Cube` / `AndRegion` / `OrRegion` / `NotRegion` (sub), the selectors, `UnitSystem` (sub), `Conformer` (sub), `Config`, `Script`, `fields`, `FrameCollection` |
| **Subpackage namespaces** | molpy code plus the native names that belong to that subsystem, exported there and nowhere else. | `mp.io` (readers / writers), `mp.builder` (planning, crystals, nanostructures, `PackingTemplate`), `mp.compute` (every analysis, e.g. `mp.compute.RDF`, `mp.compute.signal`), `mp.typifier` (e.g. `mp.typifier.OPLSAATypifier`), `mp.engine`, `mp.adapter`, `mp.data`, and `molpy.wrapper` (imported explicitly) |

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
| Apply a reaction SMARTS to a graph (bond formation / removal) | `Reaction` | [Parser](parser.md) |
| Generate `fix bond/react` pre/post topology templates | `BondReactTemplate`, `write_bond_react_map` | [IO](io.md) |
| Assemble units along a site graph (chains, rings, stars, combs, backmapping) | `Assembler`, `GrowthPlacer`, `SitePlacer`, `AxisOrienter`, `CGSmilesIR` | [Builder](builder.md) |
| Pack molecules into a simulation box | `molpack.GenCanPack`, `Target`, molrs regions as restraints | [Pack](pack.md) |
| Generate 3D conformers from a molecular graph | `Conformer` | [Conformer](conformer.md) |
| Assign force field types (OPLS-AA, MMFF94, GAFF via AmberTools) | `OPLSAATypifier`, `MMFF94Typifier`, `AntechamberTypifier`, `TLeapTypifier` | [Typifier](typifier.md) |
| Evaluate bond, angle, and pair potentials | `BondHarmonicStyle`, `LJ126Style`, `Potentials` | [Potential](potential.md) |
| Read and write molecular files (PDB, LAMMPS, GRO, …) | `read_pdb`, `write_lammps_data`, `read_xml_forcefield` | [I/O](io.md) |
| Bridge to a third-party library (in-memory) | `Adapter`, `RDKitAdapter` (optional example) | [Adapter](adapter.md) |
| Invoke external CLI tools (antechamber, tleap) | `Wrapper`, `AntechamberWrapper` | [Wrapper](wrapper.md) |
| Plan polydisperse polymer systems | `SystemPlanner`, `PolydisperseChainGenerator`, `SchulzZimmPolydisperse` | [Builder](builder.md) |
| Compute mean-squared displacement, correlations, RDF, clustering | `MSD`, `Onsager`, `RDF` | [Compute](compute.md) |
| Locate bundled data files and built-in force fields | `get_forcefield_path`, `get_path` | [Data](data.md) |
| Generate LAMMPS, CP2K, or OpenMM input decks | `LAMMPSEngine`, `CP2KEngine`, `OpenMMEngine` | [Engine](engine.md) |

## Package Responsibilities

| Package | Responsibility |
|---------|---------------|
| [Core](core.md) | Foundational data structures: `Atomistic`, `Frame`, `Block`, `Box`, `Trajectory`, `NodeRef`/`RelationRef`, `Region`, `UnitSystem`, `ForceField` |
| [Parser](parser.md) | SMILES / SMARTS / CGsmiles types at the root (`mp.SmilesIR`, `mp.SmartsPattern`, `mp.CGSmilesIR`) |
| [Builder](builder.md) | System construction: site-graph assembly and placers, polydisperse planning, nanostructures, crystals, virtual sites |
| [Pack](pack.md) | Spatial packing via molpack (`molcrafts-molpack`) |
| [Conformer](conformer.md) | 3D conformer generation from molecular graphs |
| [Typifier](typifier.md) | Force-field typing: OPLS-AA and MMFF94 (native), GAFF / GAFF2 through AmberTools |
| [Potential](potential.md) | Numerical potential kernels for bonds, angles, dihedrals, and non-bonded interactions |
| [I/O](io.md) | Format-specific readers and writers for molecular data, force fields, and trajectories |
| [Adapter](adapter.md) | Optional in-memory bridge to RDKit (worked example) |
| [Wrapper](wrapper.md) | Subprocess interfaces for AmberTools command-line executables |
| [Engine](engine.md) | Simulation engine abstractions for LAMMPS, CP2K, OpenMM |
| [Optimization](optimize.md) | Native L-BFGS minimizer at the root (`mp.LBFGS`, `mp.OptReport`) |
| [Compute](compute.md) | Trajectory analysis: MSD, Onsager, transport, dielectric, RDF, clustering, … |
| [Data](data.md) | Locators for bundled data files and built-in force fields |
