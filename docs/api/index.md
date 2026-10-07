# API Reference

Auto-generated reference for every public symbol, with typed signatures throughout. Start from the tables below: find your task, get the symbol and its package.

## Public surface: one path per name

Write `import molpy as mp`. **The root holds the subsystem modules, the core
data classes a user handles directly (promoted from `mp.core`), and the
version metadata (`version`, `release_date`), nothing else** — so there is
`mp.Frame` and `mp.Atomistic`, but no `mp.read_pdb` or `mp.LBFGS`: those are
`mp.io.read_pdb` and `mp.optimize.LBFGS`. Each name has one public module (a
promoted class is also `mp.core`'s, as the same object); there is no
`molpy.parser` or `molpy.engine.lammps.LAMMPSEngine` spelling, and the modules
behind molpy's own additions are private (`_`-prefixed). molpy is a thin
layer over molrs: every native name is the molrs object
(`mp.Atomistic is molrs.system.Atomistic`), in the molpy module that mirrors
its molrs subsystem.

| Layer | What it means | Names |
|-------|---------------|-------|
| **Promoted core data classes** | The data classes a user handles directly, on the root as the `mp.core` objects (`mp.Frame is mp.core.Frame is molrs.store.Frame`). No function, algorithm or unit preset is promoted. | `Frame`, `Block`, `Trajectory`, `Box`, `Graph`, `Atomistic`, `CoarseGrain`, `Atom`, `Bond`, `Angle`, `Dihedral`, `Improper`, `Bead`, `CGBond`, `Port`, `VirtualSite`, `DrudeParticle`, `MasslessSite`, `Element`, `Topology` |
| **Mirrors of molrs** | One molpy module per molrs subsystem, holding the same names by identity. Behaviour and docs come from the native core. | `mp.core` — molrs's core (`molrs.store`, `molrs.system`, `molrs.spatial`, `molrs.units`) in one module: the classes above plus `FrameMeta`, `MetaValue`, `keys`, `schema`, `NodeRef`, `Refs`, regions (`Cuboid`, `Sphere`, `HalfSpace`, `Region`, …), `NeighborList`, `VerletSkin`, `UnitRegistry`, `UnitPreset`, `Quantity`, … · `mp.perceive` (`Perceive`, `RingInfo`, `SmartsPattern`, `Reaction`, `SubgraphMatcher`, …) · `mp.optimize` (`LBFGS`, `OptReport`) · `mp.conformer` (`Conformer`, …) · `mp.io` (every `read_*` / `write_*`, with the per-format submodules `mp.io.smiles`, `mp.io.log`, `mp.io.mrec`, `mp.io.trajectory`, `mp.io.lammps_bond_react`) · `mp.ff` (`forcefield`, `potential`, `typifier`, `charge`, `ir`, `params`, `scale_lj`) · `mp.compute` · `mp.signal` · `mp.md` · `mp.op` · `mp.builder` · `mp.stream` |
| **molpy's additions** | Defined in molpy, placed in the subsystem whose types they act on. | `mp.core`: the column selectors (`ElementSelector`, `AtomTypeSelector`, `AtomIndexSelector`, `MaskPredicate`) and `TrajectorySplitter` with its strategies · `mp.io.log`: `LammpsLogMetricReader`, `MlpJsonlMetricReader` · `mp.io.mrec`: `MrecMetricReader` · `mp.ff.typifier`: the AmberTools typifiers · `mp.builder`: crystals, polymers, virtual sites, `PackingTemplate` |
| **molpy's own subpackages** | No molrs counterpart. | `mp.engine` (with `Script`), `mp.adapter`, `mp.data`, and `molpy.wrapper` (imported explicitly) |

## Index of Operations and Symbols

| Operation | Primary symbols | Package |
|-----------|----------------|---------|
| Construct a molecule from atoms and bonds | `Atomistic`, `def_atom`, `def_bond` | [Core](core.md) |
| Store tabular molecular data | `Block`, `Frame` | [Core](core.md) |
| Define a periodic simulation cell | `Box` | [Core](core.md) |
| Represent a time-ordered frame sequence | `Trajectory` | [Core](core.md) |
| Perceive angles/dihedrals in place; bond-graph distances; molecule ids of a frame | `generate_topology`, `topo_distances`, `Topology.from_frame` | [Core](core.md) |
| Define and query force field parameters | `mp.ff.forcefield.ForceField`, `Style`, `Type` | [Core](core.md) |
| Parse SMILES / SMARTS | `mp.io.read_smiles`, `mp.io.smiles.SmilesIR`, `mp.perceive.SmartsPattern` | [Parser](parser.md) |
| Perceive hydrogens / aromaticity / rings | `Perceive`, `RingInfo` | [Core](core.md) |
| Apply a reaction SMARTS to a graph (bond formation / removal) | `Reaction` | [Parser](parser.md) |
| Generate `fix bond/react` pre/post topology templates | `mp.io.lammps_bond_react.BondReactTemplate`, `write_lammps_bond_react_system` | [IO](io.md) |
| Assemble units along a site graph (chains, rings, stars, combs, backmapping) | `mp.builder.Assembler`, `GrowthPlacer`, `SitePlacer`, `AxisOrienter`, `mp.io.smiles.CGSmilesIR` | [Builder](builder.md) |
| Pack molecules into a simulation box | `molpack.GenCanPack`, `Target`, molrs regions as restraints | [Pack](pack.md) |
| Generate 3D conformers from a molecular graph | `Conformer` | [Conformer](conformer.md) |
| Assign force field types (OPLS-AA, MMFF94, GAFF via AmberTools) | `mp.ff.typifier.OPLSAATypifier`, `MMFF94Typifier`, `AtdTypifier`, `AntechamberTypifier`, `TLeapTypifier` | [Typifier](typifier.md) |
| Evaluate bond, angle, and pair potentials | `mp.ff.potential.PotentialCompiler`, `Potentials`, `kernel` | [Potential](potential.md) |
| Register a force-field style or category from Python | `mp.ff.ir.StyleSpec`, `register_style`, `register_category` | [Potential](potential.md) |
| Read and write molecular files (PDB, LAMMPS, GRO, …) | `mp.io.read_pdb`, `mp.io.write_lammps_data`, `mp.io.read_forcefield_xml` | [I/O](io.md) |
| Store a frame, trajectory or force field as a `*.mrec` record | `mp.io.write_mrec`, `mp.io.read_mrec_trajectory`, `mp.io.read_mrec_forcefield` | [I/O](io.md) |
| Bridge to a third-party library (in-memory) | `Adapter`, `RDKitAdapter` (optional example) | [Adapter](adapter.md) |
| Invoke external CLI tools (antechamber, tleap) | `Wrapper`, `AntechamberWrapper` | [Wrapper](wrapper.md) |
| Plan polydisperse polymer systems | `SystemPlanner`, `PolydisperseChainGenerator`, `SchulzZimmPolydisperse` | [Builder](builder.md) |
| Compute mean-squared displacement, correlations, RDF, clustering | `MSD`, `Onsager`, `RDF` | [Compute](compute.md) |
| Locate bundled data files and built-in force fields | `get_path`, `list_files` | [Data](data.md) |
| Generate LAMMPS, GROMACS, or OpenMM input decks | `LAMMPSEngine.generate_inputs`, `GROMACSEngine.generate_inputs`, `OpenMMEngine.generate_inputs` | [Engine](engine.md) |

## Package Responsibilities

| Package | Responsibility |
|---------|---------------|
| [Core](core.md) | Foundational data structures: `Atomistic`, `Frame`, `Block`, `Box`, `Trajectory`, `NodeRef`/`RelationRef`, regions, `UnitRegistry` / `UnitPreset`, and `mp.ff.forcefield.ForceField` |
| [Parser](parser.md) | SMILES / CGsmiles text (`mp.io.read_smiles`, `mp.io.smiles.SmilesIR`, `mp.io.smiles.CGSmilesIR`), SMARTS patterns (`mp.perceive.SmartsPattern`) |
| [Builder](builder.md) | System construction: site-graph assembly and placers, polydisperse planning, nanostructures, crystals, virtual sites |
| [Pack](pack.md) | Spatial packing via molpack (`molcrafts-molpack`) |
| [Conformer](conformer.md) | 3D conformer generation from molecular graphs |
| [Typifier](typifier.md) | Force-field typing: OPLS-AA and MMFF94 (native), GAFF / GAFF2 through AmberTools |
| [Potential](potential.md) | `mp.ff.potential` kernels and compiler, `mp.ff.ir` style registration |
| [I/O](io.md) | `mp.io`: every file reader and writer — structure, trajectory and force-field files, SMILES, `*.mrec` records, LAMMPS logs — and molpy's metric readers |
| [Adapter](adapter.md) | Optional in-memory bridge to RDKit (worked example) |
| [Wrapper](wrapper.md) | Subprocess interfaces for AmberTools command-line executables |
| [Engine](engine.md) | Simulation engines (LAMMPS, GROMACS, OpenMM, CP2K): one `generate_inputs` each, `run`, `Script` |
| [Optimization](optimize.md) | Native L-BFGS minimizer (`mp.optimize.LBFGS`, `mp.optimize.OptReport`) |
| [Compute](compute.md) | Trajectory analysis: MSD, Onsager, transport, dielectric, RDF, clustering, … |
| [Data](data.md) | Locators for bundled data files and built-in force fields |
