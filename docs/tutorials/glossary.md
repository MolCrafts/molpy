## Glossary

Quick definitions for MolPy's core terminology. Each entry links to the page that covers the concept in depth.

### Data structures

**Atomistic**
: An editable molecular graph where atoms are nodes and bonds are edges. Use it when the structure is still under construction — adding atoms, removing leaving groups, querying neighbors. See [Atomistic and Topology](01_atomistic_and_topology.md).

**NodeRef**
: A live handle onto one graph node (`mp.NodeRef`); `Atom` and `Bead` are node views. Dictionary-like: read and write properties with bracket notation. Hashing is by identity, not by value.

**Atom**
: A node view representing one atom. Carries arbitrary key-value properties (`element`, `charge`, `type`, etc.).

**Bead**
: A node view representing one coarse-grained site.

**RelationRef**
: A live handle onto one topology connection (`mp.RelationRef`). Holds an ordered tuple of node endpoints. Relation views: `Bond`, `Angle`, `Dihedral`, `Improper`.

**Graph**
: Native base class (`mp.Graph`) holding nodes, relations and ports. `Atomistic` and `CoarseGrain` derive from it; molpy re-exports all three by identity.

**Topology**
: Bonded terms derived from an `Atomistic`'s bond graph by the Rust kernels. `get_topo()` perceives angles/dihedrals **in place** and returns the same `Atomistic` (use `.copy().get_topo(...)` when you need an independent graph); `get_topo_neighbors()` / `get_topo_distances()` answer k-hop graph queries. There is no standalone topology class. See [Atomistic and Topology](01_atomistic_and_topology.md).

**Block**
: A columnar table mapping string keys to NumPy arrays. All columns share the same row count. Used inside `Frame` to store atoms, bonds, angles, etc. See [Block and Frame](02_block_and_frame.md).

**Frame**
: A named collection of `Block` objects plus an optional simulation box and dict-like `meta`. Represents one complete system snapshot. The universal exchange object for I/O. See [Block and Frame](02_block_and_frame.md).

**Box**
: A simulation cell defined by a 3x3 lattice matrix and periodic boundary conditions. Provides wrapping, minimum-image distances, and coordinate conversion. See [Box and Periodicity](03_box_and_periodicity.md).

**Trajectory**
: An ordered sequence of `Frame` objects. Supports lazy access via generators and `map` transforms. See [Trajectory](05_trajectory.md).

### Force field

**ForceField**
: A container that holds all styles, types, and parameters for a molecular system. Created manually or loaded from XML/LAMMPS/AMBER files.

**Style**
: An interaction family within a force field — for example, "harmonic" bonds or "lj126/cut" pairs. Defines which parameters are expected. Subclasses: `BondStyle`, `AngleStyle`, `DihedralStyle`, `PairStyle`.

**Type**
: One concrete parameter record within a style. For example, a bond type "CT-OH" with `k=320.0` and `r0=1.41`. Subclasses: `AtomType`, `BondType`, `AngleType`, `DihedralType`, `PairType`.

**Potential**
: The numerical realization of a force field's styles and types, ready for energy/force computation. Produced by `mp.PotentialCompiler(ff).compile(frame)` (or `.defer()` for a `Potentials` bound later) and evaluated against a typed `Frame` via `pots.calc_energy(frame)` / `pots.calc_forces(frame)`; the kernels run in the high-performance backend. See [Force Field](04_force_field.md).

### Modules

**Parsing**
: `mp.SmilesIR` and `mp.SmartsPattern` convert SMILES and SMARTS strings into MolPy structures; `CGSmilesIR` parses CGsmiles into ported units and site graphs. BigSMILES is not parsed. See [Parsing Chemistry](../user-guide/01_parsing_chemistry.md).

**Reaction**
: A reaction SMARTS. It matches the reactant patterns, forms and breaks bonds, and deletes the atoms that appear on the left and not on the right (the leaving groups). See [Parser](../api/parser.md).

**Assembler**
: `mp.Assembler(library, placer, orienter=None)` builds one world from a site graph: one copy of `library[bead_type]` per site, each site bond joining one accepting port of each end. Every atom gets `frag_id` (its site's ordinal) and `mol_id` (its connected component, from 1). See [Assembly](../user-guide/02_assembly.md).

**Port**
: A place where a unit may bond: an *(anchor, handle)* pair, where the handle is a real atom bonded to the anchor (usually the capping hydrogen) that leaves when the bond forms. Its kind decides its partner — `<` joins `>`, `$` joins `$`, `!` joins `!` — and a label and bond order must match. In CGsmiles, ports are the bonding descriptors (`[<]OCC[>]`).

**Site graph**
: The topology of an assembly: an `mp.CoarseGrain` whose beads name library units (`bead_type`) and whose bonds say which units join. It comes from CGsmiles notation (`CGSmilesIR(...).to_coarsegrain()`) or from coarsening a CG model (`Coarsener`), in which case each site also has a position and an axis.

**Placer**
: The assembler's rule for each copy's pose. `GrowthPlacer` grows each molecule onto its parents' ports and needs no coordinates; `SitePlacer` puts each copy's centre of mass on its site.

**Orienter**
: The optional rule that turns each copy before it is placed. `AxisOrienter` aligns a unit with its site's axis and bonds; it needs site positions.

**Typifier**
: Assigns force field types to atoms, bonds, angles, and dihedrals: implements `match`, and the base's `typify` returns a typed copy while `forcefield()` accumulates the assigned types. Native: `OPLSAATypifier`, `MMFF94Typifier`, `MMFF94STypifier`, `ElementTypifier`. GAFF / GAFF2 through AmberTools: `AntechamberTypifier`, `TLeapTypifier` (see [AmberTools Integration](../user-guide/13_ambertools_integration.md)). See [Force Field Typification](../user-guide/06_typifier.md).

**Selector**
: A composable predicate that filters atoms in a `Block` by element, type, coordinate range, or distance. Combinable with `&`, `|`, `~`. See [Selector](06_selector.md).

**Wrapper**
: Runs an external executable (antechamber, tleap, …) as a subprocess and captures its results. Crosses an execution boundary. Packing uses molpack in-process, not a wrapper. See [Wrapper and Adapter](07_wrapper_and_adapter.md).

**Adapter**
: Translates between MolPy objects and another library's in-memory objects (RDKit, OpenBabel). Crosses a representation boundary. See [Wrapper and Adapter](07_wrapper_and_adapter.md).

### Naming conventions

**atomi / atomj / atomk / atoml**
: Integer atom indices used in `Frame` and `Block` (the data-interchange layer). Always 0-based. Never store object references.

**itom / jtom / ktom / ltom**
: Atom object references used in graph-level topology (Bond, Angle, Dihedral). Never store integers. See [Naming Conventions](naming-conventions.md).

### Compute terminology

Acronyms used across the [Compute](../compute/index.md) analyses.

**RDF** — radial distribution function `g(r)`
: Probability of finding a neighbour at distance `r` relative to an ideal gas. See [RDF](../compute/rdf.md).

**MSD** — mean-squared displacement
: `⟨|r(t) − r(0)|²⟩`; its slope gives the self-diffusion coefficient. See [MSD](../compute/msd.md).

**VACF** — velocity autocorrelation function
: `⟨v(0)·v(t)⟩`; its integral (Green–Kubo) gives diffusion, its FFT gives the VDOS. See [VACF](../compute/vacf.md).

**VDOS** — vibrational density of states
: Spectral density of atomic motion, `∝ FFT[VACF]`. See [VACF](../compute/vacf.md).

**MCD** — mean-displacement correlation (distinct diffusion)
: Cross-correlated displacements between different species — the *distinct* part of diffusion, beyond the single-particle MSD. See [MSD](../compute/msd.md).

**PMSD** — polarization / charge-dipole mean-squared displacement
: MSD of $\mathbf{M}(t)=\sum q_a\mathbf{r}_a$ (unwrapped); raw curve from
 `EinsteinConductivity`. Fit $\sigma$ with `LinearFit` + SI scale. See
 [MSD](../compute/msd.md).

**Current ACF** — $\langle\mathbf{J}(0)\cdot\mathbf{J}(t)\rangle$
: From `GreenKuboConductivity` with $\mathbf{J}=\sum q\mathbf{v}$. Integrate
 with `CumulativeTrapezoid` then SI-scale for $\sigma$.

**SDF** — spatial distribution function
: Three-dimensional density of neighbours around a reference frame (angular structure, not just radial). See [Distribution](../compute/distribution.md).

**CDF** — combined distribution function
: A joint histogram over two geometric observables (e.g. distance × angle). See [Distribution](../compute/distribution.md).

**PMFT** — potential of mean force and torque
: Free energy `−k_BT ln g` over relative position/orientation coordinates. See [Distribution](../compute/distribution.md).

**ROA / VCD** — Raman optical activity / vibrational circular dichroism
: Chiroptical vibrational spectra derived from correlation functions of the polarizability / magnetic-dipole responses. See [Spectra](../compute/spectra.md).
