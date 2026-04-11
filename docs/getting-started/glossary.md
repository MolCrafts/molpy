## Glossary

Quick definitions for MolPy's core terminology. Each entry links to the page that covers the concept in depth.

### Data structures

**Atomistic**
:   An editable molecular graph where atoms are nodes and bonds are edges. Use it when the structure is still under construction — adding atoms, removing leaving groups, querying neighbors. See [Atomistic and Topology](../tutorials/01_atomistic_and_topology.md).

**Entity**
:   Base class for atoms and beads. Dictionary-like: read and write properties with bracket notation. Uses identity-based hashing (`id(self)`), not value-based equality.

**Atom**
:   An `Entity` subclass representing one atom. Carries arbitrary key-value properties (`element`, `charge`, `type`, etc.).

**Bead**
:   An `Entity` subclass representing one coarse-grained site.

**Link**
:   Base class for topology connections. Holds an ordered tuple of `Entity` endpoints. Subclasses: `Bond`, `Angle`, `Dihedral`, `Improper`.

**Struct**
:   Base class that aggregates entities and links into a container. Subclasses: `Atomistic`, `CoarseGrain`.

**Topology**
:   A thin wrapper around an igraph graph, derived from the bond list of an `Atomistic`. Provides graph algorithms (shortest paths, connected components, ring detection). See [Atomistic and Topology](../tutorials/01_atomistic_and_topology.md).

**Block**
:   A columnar table mapping string keys to NumPy arrays. All columns share the same row count. Used inside `Frame` to store atoms, bonds, angles, etc. See [Block and Frame](../tutorials/02_block_and_frame.md).

**Frame**
:   A named collection of `Block` objects plus free-form metadata. Represents one complete system snapshot. The universal exchange object for I/O. See [Block and Frame](../tutorials/02_block_and_frame.md).

**Box**
:   A simulation cell defined by a 3x3 lattice matrix and periodic boundary conditions. Provides wrapping, minimum-image distances, and coordinate conversion. See [Box and Periodicity](../tutorials/03_box_and_periodicity.md).

**Trajectory**
:   An ordered sequence of `Frame` objects. Supports lazy access via generators and `map` transforms. See [Trajectory](../tutorials/05_trajectory.md).


### Force field

**ForceField**
:   A container that holds all styles, types, and parameters for a molecular system. Created manually or loaded from XML/LAMMPS/AMBER files.

**Style**
:   An interaction family within a force field — for example, "harmonic" bonds or "lj126/cut" pairs. Defines which parameters are expected. Subclasses: `BondStyle`, `AngleStyle`, `DihedralStyle`, `PairStyle`.

**Type**
:   One concrete parameter record within a style. For example, a bond type "CT-OH" with `k0=320.0` and `r0=1.41`. Subclasses: `AtomType`, `BondType`, `AngleType`, `DihedralType`, `PairType`.

**Potential**
:   The numerical realization of a style's types — arrays of parameters ready for energy/force computation. Produced by `style.to_potential()`. See [Force Field](../tutorials/04_force_field.md).


### Modules

**Parser**
:   Converts string notations (SMILES, SMARTS, BigSMILES, CGSmiles) into MolPy structures. See [Parsing Chemistry](../user-guide/01_parsing_chemistry.ipynb).

**Reacter**
:   Executes a chemical reaction by connecting two `Atomistic` objects at designated port atoms, removing leaving groups, and forming new bonds. See [Stepwise Polymer Construction](../user-guide/02_polymer_stepwise.ipynb).

**Port**
:   A marker on an atom (`<`, `>`, or `$`) indicating that it is a reactive connection point for polymerization.

**Typifier**
:   Assigns force field types to atoms, bonds, angles, and dihedrals via SMARTS pattern matching. Subclasses: `OplsAtomisticTypifier`, `GaffTypifier`. See [Force Field Typification](../user-guide/06_typifier.ipynb).

**Selector**
:   A composable predicate that filters atoms in a `Block` by element, type, coordinate range, or distance. Combinable with `&`, `|`, `~`. See [Selector](../tutorials/06_selector.md).

**Tool**
:   A high-level workflow interface that wires multiple MolPy modules into a single callable — for example, `PrepareMonomer`, `polymer()`. See [Tool Layer](../tutorials/tools.md).

**Wrapper**
:   Runs an external executable (antechamber, tleap, Packmol) as a subprocess and captures its results. Crosses an execution boundary. See [Wrapper and Adapter](../tutorials/07_wrapper_and_adapter.md).

**Adapter**
:   Translates between MolPy objects and another library's in-memory objects (RDKit, OpenBabel). Crosses a representation boundary. See [Wrapper and Adapter](../tutorials/07_wrapper_and_adapter.md).


### Naming conventions

**atomi / atomj / atomk / atoml**
:   Integer atom indices used in `Frame` and `Block` (the data-interchange layer). Always 0-based. Never store object references.

**itom / jtom / ktom / ltom**
:   Atom object references used in `Entity`-level topology (Bond, Angle, Dihedral). Never store integers. See [Naming Conventions](naming-conventions.md).
