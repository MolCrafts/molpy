## Glossary

Quick definitions for MolPy's core terminology. Each entry links to the page that covers the concept in depth.

### Data structures

**Atomistic**
: An editable molecular graph where atoms are nodes and bonds are edges. Use it when the structure is still under construction — adding atoms, removing leaving groups, querying neighbors. See [Atomistic and Topology](01_atomistic_and_topology.md).

**NodeRef**
: A live handle onto one graph node (`mp.core.NodeRef`); `Atom` and `Bead` are node views. Dictionary-like: read and write properties with bracket notation. Hashing is by identity, not by value.

**Atom**
: A node view representing one atom. Carries arbitrary key-value properties (`element`, `charge`, `type`, etc.).

**Bead**
: A node view representing one coarse-grained site.

**RelationRef**
: A live handle onto one topology connection (`mp.core.RelationRef`). Holds an ordered tuple of node endpoints. Relation views: `Bond`, `Angle`, `Dihedral`, `Improper`.

**MolGraph**
: Native base class (`mp.MolGraph`) holding nodes, relations and ports. `Atomistic` and `CoarseGrain` derive from it; molpy re-exports all three by identity.

**Topology**
: Bonded terms derived from an `Atomistic`'s bond graph by the native kernels. `generate_topology(gen_angle=..., gen_dihedral=...)` perceives angles and dihedrals **in place** and returns the counts it added (use `.copy()` first when you need an independent graph); `topo_distances(handle, max_hops=...)` answers bond-graph distance queries. On the tabular side, `mp.Topology.from_frame(frame)` reads a `Frame`'s bonds, for example `connected_components()` to number its molecules. See [Atomistic and Topology](01_atomistic_and_topology.md).

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
: An interaction family within a force field — for example, `harmonic` bonds or `lj/cut` pairs — defined with `ff.def_style(category, name)`. Subclasses: `AtomStyle`, `BondStyle`, `AngleStyle`, `DihedralStyle`, `ImproperStyle`, `PairStyle`.

**Type**
: One concrete parameter record within a style, defined with `style.def_type(name, *endpoints, **params)`. For example, a bond type `CT-OH` between the atom types `CT` and `OH` with `k=320.0` and `r0=1.41`. Subclasses: `AtomType`, `BondType`, `AngleType`, `DihedralType`, `ImproperType`, `PairType`.

**Potential**
: The numerical realization of a force field's styles and types, ready for energy/force computation. Produced by `mp.ff.compile.PotentialCompiler(ff).compile(frame)` (or `.defer()` for a `Potentials` bound later) and evaluated against a typed `Frame` via `pots.calc_energy(frame)` / `pots.calc_forces(frame)`; the kernels run in the high-performance backend. See [Force Field](04_force_field.md).

### Modules

**Parsing**
: `mp.io.smiles.SmilesIr` and `mp.perceive.SmartsPattern` convert SMILES and SMARTS strings into MolPy structures; `mp.io.cgsmiles.CgSmilesIr` parses CGsmiles into ported units and site graphs. BigSMILES is not parsed. See [Parsing Chemistry](../user-guide/01_parsing_chemistry.md).

**Reaction**
: A reaction SMARTS. It matches the reactant patterns, forms and breaks bonds, and deletes the atoms that appear on the left and not on the right (the leaving groups). See [Notation](../api/notation.md).

**Assembler**
: `mp.builder.Assembler(library, placer, orienter=None)` builds one world from a site graph: one copy of `library[bead_type]` per site, each site bond joining one accepting port of each end. Every atom gets `frag_id` (its site's ordinal) and `mol_id` (its connected component, from 1). See [Assembly](../user-guide/02_assembly.md).

**Port**
: A place where a unit may bond: an *(anchor, handle)* pair, where the handle is a real atom bonded to the anchor (usually the capping hydrogen) that leaves when the bond forms. Its kind decides its partner — `<` joins `>`, `$` joins `$`, `!` joins `!` — and a label and bond order must match. In CGsmiles, ports are the bonding descriptors (`[<]OCC[>]`).

**Site graph**
: The topology of an assembly: an `mp.CoarseGrain` whose beads name library units (`bead_type`) and whose bonds say which units join. It comes from CGsmiles notation (`mp.io.cgsmiles.read_cgsmiles(...).to_coarsegrain()`) or from coarsening a CG model (`Coarsener`), in which case each site also has a position and an axis.

**Placer**
: The assembler's rule for each copy's pose. `GrowthPlacer` grows each molecule onto its parents' ports and needs no coordinates; `SitePlacer` puts each copy's centre of mass on its site.

**Orienter**
: The optional rule that turns each copy before it is placed. `AxisOrienter` aligns a unit with its site's axis and bonds; it needs site positions.

**Typifier**
: Assigns force field types to atoms, bonds, angles, and dihedrals: implements `assign` (returning a `TypeAssignment`), and the base's `typify` returns a typed copy while `forcefield()` accumulates the assigned types. Native: `OplsAaTypifier`, `Mmff94Typifier`, `Mmff94sTypifier`, `ElementTypifier`, and GAFF / GAFF2 as `AtdTypifier` then `GaffTypifier`. AM1-BCC charges for the native route: `mp.ff.charge.Am1BccModel` (sqm via AmberTools). A native GAFF force field merges with an `AmberPolymerBuilder` chain's. The whole AmberTools pipeline, for parameters of your own on top of the leaprc and antechamber's other charge methods: `AntechamberTypifier`, `TleapTypifier` (see [AmberTools Integration](../user-guide/13_ambertools_integration.md)). See [Force Field Typification](../user-guide/06_typifier.md).

**Selector**
: A composable predicate that filters atoms in a `Block` by element, type, coordinate range, or distance. Combinable with `&`, `|`, `~`. See [Selector](06_selector.md).

**Tool**
: An external program molpy runs as a subprocess (an engine, or an AmberTools program such as antechamber or tleap), resolved from molpy's configuration; every failure is an `mp.tool.ToolError`. Packing uses molpack in-process, not a tool. See [Tool](../api/tool.md).

### Naming conventions

**atomi / atomj / atomk / atoml**
: Integer atom indices used in `Frame` and `Block` (the data-interchange layer). Always 0-based. Never store object references.

**itom / jtom, endpoints**
: Atom views on graph-level topology: `bond.itom` / `bond.jtom` are a bond's two atoms, and every relation (`Bond`, `Angle`, `Dihedral`, `Improper`) lists its atoms in order as `.endpoints`. They are views, never integers. See [Naming Conventions](naming-conventions.md).

### Compute terminology

Acronyms used across the [Compute](../compute/index.md) analyses.

**RDF** — radial distribution function `g(r)`
: Probability of finding a neighbour at distance `r` relative to an ideal gas. See [RDF](../compute/rdf.md).

**MSD** — mean-squared displacement
: `⟨|r(t) − r(0)|²⟩`; its slope gives the self-diffusion coefficient. See [MSD](../compute/msd.md).

**VACF** — velocity autocorrelation function
: `⟨v(0)·v(t)⟩`; its integral (Green–Kubo) gives diffusion, its FFT gives the VDOS. See [VACF](../compute/vacf.md).

**VDOS** — vibrational density of states
: Spectral density of atomic motion, `∝ FFT[VACF]` (`Vacf` then `Spectrum`). See [Spectra](../compute/spectra.md).

**OACF** — orientational autocorrelation function
: `C_ℓ(t) = ⟨P_ℓ(û(0)·û(t))⟩` of a molecular unit vector, computed by `ReorientationAcf`; `ℓ = 1` is what dielectric relaxation sees, `ℓ = 2` what NMR and Raman see. See [Reorientation](../compute/reorientation.md).

**Correlator**
: The one time-correlation estimator. `Msd`, `Vacf`, `OnsagerCorrelation`, `DipoleAcf`, `CurrentAcf`, `ReorientationAcf`, `RotationalAutocorrelation` and `Survival` are configurations of it; a correlation with no named class is a `Correlator` built directly. See [Spectra](../compute/spectra.md).

**TimeCorrelation**
: The one object every correlation returns: the curve `values[lag, channel]` on its lag axis `lag_times`, and nothing derived. Fits (`LinearFit`, `Plateau`, `DebyeFit`, `CumulativeTrapezoid`) are computes that read it. See [VACF](../compute/vacf.md).

**MCD** — mean-displacement correlation (distinct diffusion)
: Cross-correlated displacements between different species — the *distinct* part of diffusion, beyond the single-particle MSD. See [MSD](../compute/msd.md).

**PMSD** — polarization / charge-dipole mean-squared displacement
: MSD of $\mathbf{M}(t)=\sum q_a\mathbf{r}_a$ (unwrapped): `DipoleMoment`
 builds the series, `Msd` gives the raw curve, `LinearFit` + SI scale give
 $\sigma$. See [PMSD](../compute/pmsd.md).

**Current ACF** — $\langle\mathbf{J}(0)\cdot\mathbf{J}(t)\rangle$
: `CurrentAcf` of $\mathbf{J}=\sum q\mathbf{v}$. Integrate with
 `CumulativeTrapezoid` then SI-scale for $\sigma$. See [JACF](../compute/jacf.md).

**SDF** — spatial distribution function
: Three-dimensional density of neighbours around a reference frame (angular structure, not just radial). See [Spatial](../compute/spatial.md).

**CDF** — combined distribution function
: A joint histogram over two geometric observables (e.g. distance × angle). See [Distribution](../compute/distribution.md).

**PMFT** — potential of mean force and torque
: Free energy `−k_BT ln g` over relative position/orientation coordinates. See [PMFT](../compute/pmft.md).

**ROA / VCD** — Raman optical activity / vibrational circular dichroism
: Chiroptical vibrational spectra derived from correlation functions of the polarizability / magnetic-dipole responses. See [Spectra](../compute/spectra.md).
