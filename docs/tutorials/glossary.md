## Glossary

Quick definitions for MolPy's core terminology. Each entry links to the page that covers the concept in depth.

### Data structures

**Atomistic**
:   An editable molecular graph where atoms are nodes and bonds are edges. Use it when the structure is still under construction — adding atoms, removing leaving groups, querying neighbors. See [Atomistic and Topology](01_atomistic_and_topology.md).

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
:   The derived view of an `Atomistic`'s bond graph, computed by the molrs Rust kernels. `get_topo()` returns a new `Atomistic` with angles/dihedrals perceived from the bonds; `get_topo_neighbors()` / `get_topo_distances()` answer k-hop graph queries. There is no standalone topology class. See [Atomistic and Topology](01_atomistic_and_topology.md).

**Block**
:   A columnar table mapping string keys to NumPy arrays. All columns share the same row count. Used inside `Frame` to store atoms, bonds, angles, etc. See [Block and Frame](02_block_and_frame.md).

**Frame**
:   A named collection of `Block` objects plus free-form metadata. Represents one complete system snapshot. The universal exchange object for I/O. See [Block and Frame](02_block_and_frame.md).

**Box**
:   A simulation cell defined by a 3x3 lattice matrix and periodic boundary conditions. Provides wrapping, minimum-image distances, and coordinate conversion. See [Box and Periodicity](03_box_and_periodicity.md).

**Trajectory**
:   An ordered sequence of `Frame` objects. Supports lazy access via generators and `map` transforms. See [Trajectory](05_trajectory.md).


### Force field

**ForceField**
:   A container that holds all styles, types, and parameters for a molecular system. Created manually or loaded from XML/LAMMPS/AMBER files.

**Style**
:   An interaction family within a force field — for example, "harmonic" bonds or "lj126/cut" pairs. Defines which parameters are expected. Subclasses: `BondStyle`, `AngleStyle`, `DihedralStyle`, `PairStyle`.

**Type**
:   One concrete parameter record within a style. For example, a bond type "CT-OH" with `k=320.0` and `r0=1.41`. Subclasses: `AtomType`, `BondType`, `AngleType`, `DihedralType`, `PairType`.

**Potential**
:   The numerical realization of a force field's styles and types, ready for energy/force computation. Produced by `ff.to_potentials()` (a deferred `Potentials`) and evaluated against a typed `Frame` via `pots.calc_energy(frame)` / `pots.calc_forces(frame)`; the kernels run in the molrs Rust extension. See [Force Field](04_force_field.md).


### Modules

**Parser**
:   Converts string notations (SMILES, SMARTS, BigSMILES, CGSmiles) into MolPy structures. See [Parsing Chemistry](../user-guide/01_parsing_chemistry.md).

**Reaction**
:   A reaction SMARTS. It matches the reactant patterns, forms and breaks bonds, and deletes the atoms that appear on the left and not on the right (the leaving groups). All the chemistry lives here. See [Assembly](../user-guide/02_assembly.md).

**GraphAssembler**
:   Pastes molecules into one world, applies a `Reaction` wherever its `Selector` says, and repairs the force-field types near each new bond. `PolymerBuilder` is a `GraphAssembler` that also owns a monomer library and speaks CGSmiles.

**Site**
:   A name (`fields.SITE`) on an atom that may react. Sites have no direction and no role — a linear chain, a branch point and a ring closure differ only in how many sites a monomer carries and how the topology pairs them.

**Typifier**
:   Assigns force field types to atoms, bonds, angles, and dihedrals via SMARTS pattern matching. Subclasses: `OplsTypifier`, `ClpTypifier`, `MMFFTypifier`, `PairTypifier`. (GAFF atom types are *not* a Typifier — they come from AmberTools/antechamber; see [AmberTools Integration](../user-guide/13_ambertools_integration.md).) See [Force Field Typification](../user-guide/06_typifier.md).

**Selector**
:   A composable predicate that filters atoms in a `Block` by element, type, coordinate range, or distance. Combinable with `&`, `|`, `~`. See [Selector](06_selector.md).

**Wrapper**
:   Runs an external executable (antechamber, tleap, Packmol) as a subprocess and captures its results. Crosses an execution boundary. See [Wrapper and Adapter](07_wrapper_and_adapter.md).

**Adapter**
:   Translates between MolPy objects and another library's in-memory objects (RDKit, OpenBabel). Crosses a representation boundary. See [Wrapper and Adapter](07_wrapper_and_adapter.md).


### Naming conventions

**atomi / atomj / atomk / atoml**
:   Integer atom indices used in `Frame` and `Block` (the data-interchange layer). Always 0-based. Never store object references.

**itom / jtom / ktom / ltom**
:   Atom object references used in `Entity`-level topology (Bond, Angle, Dihedral). Never store integers. See [Naming Conventions](naming-conventions.md).


### Compute terminology

Acronyms used across the [Compute](../compute/index.md) analyses.

**RDF** — radial distribution function `g(r)`
:   Probability of finding a neighbour at distance `r` relative to an ideal gas. See [Structural Analysis](../compute/structure.md).

**MSD** — mean-squared displacement
:   `⟨|r(t) − r(0)|²⟩`; its slope gives the self-diffusion coefficient. See [Diffusion & Ionic Transport](../compute/transport.md).

**VACF** — velocity autocorrelation function
:   `⟨v(0)·v(t)⟩`; its integral (Green–Kubo) gives diffusion, its FFT gives the VDOS. See [Velocity Autocorrelation & VDOS](../compute/vacf.md).

**VDOS** — vibrational density of states
:   Spectral density of atomic motion, `∝ FFT[VACF]`. See [Velocity Autocorrelation & VDOS](../compute/vacf.md).

**MCD** — mean-displacement correlation (distinct diffusion)
:   Cross-correlated displacements between different species — the *distinct* part of diffusion, beyond the single-particle MSD. See [Diffusion & Ionic Transport](../compute/transport.md).

**PMSD** — polarization mean-squared displacement
:   MSD of the collective charge dipole; the Einstein route to ionic conductivity. See [Dielectric Spectroscopy](../compute/dielectric.md).

**SDF** — spatial distribution function
:   Three-dimensional density of neighbours around a reference frame (angular structure, not just radial). See [Distribution Functions](../compute/distributions.md).

**CDF** — combined distribution function
:   A joint histogram over two geometric observables (e.g. distance × angle). See [Distribution Functions](../compute/distributions.md).

**PMFT** — potential of mean force and torque
:   Free energy `−k_BT ln g` over relative position/orientation coordinates. See [Distribution Functions](../compute/distributions.md).

**ROA / VCD** — Raman optical activity / vibrational circular dichroism
:   Chiroptical vibrational spectra derived from correlation functions of the polarizability / magnetic-dipole responses. See [Vibrational Spectra](../compute/spectra.md).
