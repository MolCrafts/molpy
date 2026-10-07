# API Reference

Auto-generated reference for every public symbol, with typed signatures throughout. Start from the tables below: find your task, get the symbol and its package.

## Public surface: one path per name

Write `import molpy as mp` and reach every name through `mp`. Each name has exactly
one public path — there is no `molpy.core.Atomistic`, `molpy.parser`,
`molpy.optimize` or `molpy.engine.lammps.LAMMPSEngine` spelling: the modules
behind molpy's own subpackages are private (`_`-prefixed). molpy is a thin layer over molrs: every native name
is the molrs object (`mp.Atomistic is molrs.system.Atomistic`), placed by one
rule. A molrs subsystem molpy has a namespace for is mirrored there under the
same name (`mp.ff` with its submodules, `mp.io`, `mp.compute`, `mp.signal`,
`mp.md`, `mp.op`, `mp.builder`); the data-model subsystems (`store`,
`system`, `spatial`, `units`, `perceive`, `optimize`, `conformer`) are
flattened onto the root.

| Layer | What it means | Names |
|-------|---------------|-------|
| **Flattened onto the root** (molrs) | Identity re-export: `mp.Atomistic is molrs.system.Atomistic`. Behaviour and docs come from the native core; molpy adds nothing. | `molrs.system` — `Atomistic`, `CoarseGrain`, `Graph`, `Atom`, `Bond`, `Angle`, `Dihedral`, `Improper`, `VirtualSite`, `DrudeParticle`, `MasslessSite`, `Bead`, `CGBond`, `NodeRef`, `RelationRef`, `Refs`, `RelationBuckets`, `Port`, `Topology`, `ExtractedSubgraph`, `Element` · `molrs.store` — `Frame`, `Block`, `BlockDtypeError`, `FrameMeta`, `MetaDocument`, `MetaValue`, `ScalarObservable`, `VectorObservable`, `Trajectory`, `keys`, `schema` · `molrs.spatial` — `Box`, `Cuboid`, `Sphere`, `HalfSpace`, `Parallelepiped`, `Cylinder`, `Ellipsoid`, `Polyhedron`, `SphereUnion`, `Region`, `TriMesh`, `Trace`, `NeighborQuery`, `NeighborList`, `Neighbors`, `VerletSkin` · `molrs.units` — `Unit`, `Quantity`, `UnitRegistry`, `UnitPreset`, `UnitsError`, `AMBER_COULOMB` · `molrs.perceive` — `Perceive`, `RingInfo`, `SmartsPattern`, `SmartsMatch`, `Reaction`, `SubgraphMatcher` · `molrs.optimize` — `LBFGS`, `OptReport` · `molrs.conformer` — `Conformer`, `ConformerReport`, `ConformerStageReport` |
| **Mirrored subpackages** (molrs) | The molrs subsystem under its own name, every native name the molrs object, plus molpy's additions. | `mp.ff` (`forcefield`, `potential`, `typifier` + the AmberTools typifiers, `charge`, `ir`, `params`, `scale_lj`), `mp.io`, `mp.compute`, `mp.signal`, `mp.md`, `mp.op`, `mp.builder` (+ crystals, polymers, virtual sites, `PackingTemplate`) |
| **molpy's own** | Defined in molpy. | `TrajectorySplitter` and its strategies (over the native `Trajectory`), the column selectors (`ElementSelector`, `AtomTypeSelector`, `AtomIndexSelector`, `MaskPredicate`), `FrameCollection`; the subpackages `mp.engine` (with `Script`), `mp.adapter`, `mp.data`, and `molpy.wrapper` / `molpy.integrations` (imported explicitly) |

## Index of Operations and Symbols

| Operation | Primary symbols | Package |
|-----------|----------------|---------|
| Construct a molecule from atoms and bonds | `Atomistic`, `def_atom`, `def_bond` | [Core](core.md) |
| Store tabular molecular data | `Block`, `Frame` | [Core](core.md) |
| Define a periodic simulation cell | `Box` | [Core](core.md) |
| Represent a time-ordered frame sequence | `Trajectory` | [Core](core.md) |
| Perceive angles/dihedrals in place; bond-graph distances; molecule ids of a frame | `generate_topology`, `topo_distances`, `Topology.from_frame` | [Core](core.md) |
| Define and query force field parameters | `mp.ff.forcefield.ForceField`, `Style`, `Type` | [Core](core.md) |
| Parse SMILES / SMARTS | `mp.io.SmilesIR` (`.to_atomistic()`), `SmartsPattern` | [Parser](parser.md) |
| Perceive hydrogens / aromaticity / rings | `Perceive`, `RingInfo` | [Core](core.md) |
| Apply a reaction SMARTS to a graph (bond formation / removal) | `Reaction` | [Parser](parser.md) |
| Generate `fix bond/react` pre/post topology templates | `mp.io.BondReactTemplate`, `write_lammps_bond_react_system` | [IO](io.md) |
| Assemble units along a site graph (chains, rings, stars, combs, backmapping) | `mp.builder.Assembler`, `GrowthPlacer`, `SitePlacer`, `AxisOrienter`, `mp.io.CGSmilesIR` | [Builder](builder.md) |
| Pack molecules into a simulation box | `molpack.GenCanPack`, `Target`, molrs regions as restraints | [Pack](pack.md) |
| Generate 3D conformers from a molecular graph | `Conformer` | [Conformer](conformer.md) |
| Assign force field types (OPLS-AA, MMFF94, GAFF via AmberTools) | `mp.ff.typifier.OPLSAATypifier`, `MMFF94Typifier`, `AtdTypifier`, `AntechamberTypifier`, `TLeapTypifier` | [Typifier](typifier.md) |
| Evaluate bond, angle, and pair potentials | `mp.ff.potential.PotentialCompiler`, `Potentials`, `kernel` | [Potential](potential.md) |
| Register a force-field style or category from Python | `mp.ff.ir.StyleSpec`, `register_style`, `register_category` | [Potential](potential.md) |
| Read and write molecular files (PDB, LAMMPS, GRO, …) | `mp.io.read_pdb`, `mp.io.write_lammps_data`, `mp.ff.forcefield.read_forcefield_xml` | [I/O](io.md) |
| Store a frame, trajectory or force field as a `*.mrec` record | `mp.io.mrec.write`, `mp.io.mrec.read_trajectory`, `mp.io.mrec.read_forcefield` | [I/O](io.md) |
| Bridge to a third-party library (in-memory) | `Adapter`, `RDKitAdapter` (optional example) | [Adapter](adapter.md) |
| Invoke external CLI tools (antechamber, tleap) | `Wrapper`, `AntechamberWrapper` | [Wrapper](wrapper.md) |
| Plan polydisperse polymer systems | `SystemPlanner`, `PolydisperseChainGenerator`, `SchulzZimmPolydisperse` | [Builder](builder.md) |
| Compute mean-squared displacement, correlations, RDF, clustering | `MSD`, `Onsager`, `RDF` | [Compute](compute.md) |
| Locate bundled data files and built-in force fields | `get_forcefield_path`, `get_path` | [Data](data.md) |
| Generate LAMMPS, GROMACS, or OpenMM input decks | `LAMMPSEngine.generate_inputs`, `GROMACSEngine.generate_inputs`, `OpenMMEngine.generate_inputs` | [Engine](engine.md) |

## Package Responsibilities

| Package | Responsibility |
|---------|---------------|
| [Core](core.md) | Foundational data structures: `Atomistic`, `Frame`, `Block`, `Box`, `Trajectory`, `NodeRef`/`RelationRef`, regions, `UnitRegistry` / `UnitPreset`, and `mp.ff.forcefield.ForceField` |
| [Parser](parser.md) | SMILES / CGsmiles text on `mp.io` (`mp.io.SmilesIR`, `mp.io.CGSmilesIR`), SMARTS patterns at the root (`mp.SmartsPattern`) |
| [Builder](builder.md) | System construction: site-graph assembly and placers, polydisperse planning, nanostructures, crystals, virtual sites |
| [Pack](pack.md) | Spatial packing via molpack (`molcrafts-molpack`) |
| [Conformer](conformer.md) | 3D conformer generation from molecular graphs |
| [Typifier](typifier.md) | Force-field typing: OPLS-AA and MMFF94 (native), GAFF / GAFF2 through AmberTools |
| [Potential](potential.md) | `mp.ff.potential` kernels and compiler, `mp.ff.ir` style registration |
| [I/O](io.md) | `mp.io`: structure and trajectory formats; force-field formats on `mp.ff.forcefield` |
| [Adapter](adapter.md) | Optional in-memory bridge to RDKit (worked example) |
| [Wrapper](wrapper.md) | Subprocess interfaces for AmberTools command-line executables |
| [Engine](engine.md) | Simulation engines (LAMMPS, GROMACS, OpenMM, CP2K): one `generate_inputs` each, `run`, `Script` |
| [Optimization](optimize.md) | Native L-BFGS minimizer at the root (`mp.LBFGS`, `mp.OptReport`) |
| [Compute](compute.md) | Trajectory analysis: MSD, Onsager, transport, dielectric, RDF, clustering, … |
| [Data](data.md) | Locators for bundled data files and built-in force fields |
