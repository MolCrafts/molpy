# Python Reference

```python
import molrs
```

The top level of `molrs` is its subsystems, plus the core data classes a user
handles directly. Those classes are the same objects as `molrs.core`'s
(`molrs.Frame is molrs.core.Frame`). No function,
algorithm, region or unit preset is promoted; everything else has one path,
the Python module named after its Rust owner. Import either way:

```python
from molrs import Frame, Block, Trajectory, Topology
from molrs.core import Box, NeighborList
from molrs.ff.compile import PotentialCompiler
```

This page is rendered from the installed `molrs` package by
`mkdocstrings-python`. The type stub `molrs-python/python/molrs/_native.pyi` is
the committed companion artifact that keeps signatures visible to static tools
and the docs build.

| Module | Rust owner | Holds |
|---|---|---|
| `molrs.core` | `molrs::core` | `Block`, `Frame`, `FrameMeta`, `MetaValue`, `MetaDocument`, `Trajectory`, `ObservableRecord`, `BlockDtypeError`; `Box` (Rust `SimBox`) and its `Cell`, `NeighborList`, `Neighbors`, `NeighborQuery`, `VerletSkin`, the regions, `TriMesh`, `Trace`; `MolGraph`, `Atomistic`, `CoarseGrain`, the node / relation views, `ExtractedSubgraph`, `Element`, `BondOrder`, `BondNumber`, `Topology`; `Unit`, `Quantity`, `UnitRegistry`, `UnitPreset`, `UnitsError` |
| `molrs.core.keys` | `molrs::core::keys` | the canonical column, frame-meta and graph keys |
| `molrs.core.schema` | `molrs::core::schema` | `ColumnSpec`, `BlockSpec`, the block names, `relation_endpoints` |
| `molrs.core.constants` | `molrs::core::constants` | every physical and engine constant (`AVOGADRO`, `COULOMB_REAL`, `AMBER_COULOMB`, `AMBER_SCEE`, …) |
| `molrs.op` | `molrs::op` | `superpose`, `centroid`, `place_from_internal_coords`, `Superposition`, `DEFAULT_GAP_TOL` |
| `molrs.perceive` | `molrs::perceive` | `perceive_rings` → `RingSet`, the `perceive_*` / `assign_*` functions, `add_hydrogens`, `SmartsPattern`, `SmartsMatch`, `Reaction`, `SubgraphMatcher` |
| `molrs.io` | `molrs::io` | every file reader and writer, one door per format and direction: `read_<fmt>(source)` / `write_<fmt>(sink, frame)` (a source is an `os.PathLike`, `bytes` or `Text`; a sink an `os.PathLike` or a `bytearray`) — structure and trajectory files, `write_mrec`, wire-encoded frames (`read_wire_frame` / `write_wire_frame`), the LAMMPS log, CSV blocks; `Text`, `CubeReader`, `XmlForcefieldReader` / `XmlForcefieldWriter` |
| `molrs.io.pdb`, `.xyz`, `.gro`, `.mol2`, `.cif`, `.sdf`, `.dcd`, `.trr`, `.xtc` | `molrs::io::{pdb, xyz, gro, mol2, cif, sdf, dcd, trr, xtc}` | each format's multi-frame classes: `<Fmt>Reader(source)` and (all but SDF) `<Fmt>Writer(sink)` |
| `molrs.io.lammps` | `molrs::io::lammps` | `LammpsDumpReader` / `LammpsDumpWriter`, `LammpsForcefieldReader` / `LammpsForcefieldWriter`, `BondReactTemplate`, the `Lammps*` log records, `is_lammps_log` |
| `molrs.io.gromacs`, `.amber`, `.openmm_xml` | `molrs::io::{gromacs, amber, openmm_xml}` | the force-field classes: `GromacsTopForcefieldReader` / `Writer`, `AmberPrmtopForcefieldReader`, `AmberFrcmodWriter`, `OpenmmXmlReader` / `Writer` |
| `molrs.io.smiles` | `molrs::io::smiles` | `SmilesIr`, `FragmentIr`, `SmilesError`, `BondingDescriptor` |
| `molrs.io.cgsmiles` | `molrs::io::cgsmiles` | `CgSmilesIr` and the CGsmiles records |
| `molrs.io.mrec` | `molrs::io::mrec` | `MrecFile`, `MrecReader`, `MrecWriter`, `SequenceSchema`, `ForceFieldSection`, `pack_mrec_zip`, `validation` |
| `molrs.ff.forcefield` | `molrs::ff::forcefield` | `ForceField`, the `Style` / `ForceFieldType` handles (the data model; its files are `molrs.io`'s) |
| `molrs.ff.potential` | `molrs::ff::potential` | `Potentials`, `WeightedTerms`, `PairLjCut`, `intramolecular_pairs`, `Potential` |
| `molrs.ff.compile` | `molrs::ff::compile` | `PotentialCompiler`, `ExplicitTerms` |
| `molrs.ff.typifier` | `molrs::ff::typifier` | `Typifier`, `TypeAssignment`, the built-in typifiers, `assign_cmaps` |
| `molrs.ff.charge` | `molrs::ff::charge` | `BccModel`, `MullikenModel`, `GasteigerModel` |
| `molrs.ff.ir` | `molrs::ff::ir` | the force-field IR's vocabulary (`ParamSpec`, `StyleSpec`, `CategorySpec`) and its `IrError` family |
| `molrs.ff.style_registry` | `molrs::ff::style_registry` | `register_category`, `register_style`, `register_engine_form`, `unregister_style`, `StyleDeclaration`, `styles`, `categories`, `evaluate` |
| `molrs.ff.params` | `molrs::ff::params` | `clpol_polarizability` |
| `molrs.ff.clpol_scaling` | `molrs::ff::clpol_scaling` | `FragmentScaling`, `compute_k_ij`, `scale_lj`, `fragment_table` (the shipped table) |
| `molrs.optimize` | `molrs::optimize` | `Lbfgs`, `OptimizationReport` |
| `molrs.md` | `molrs::md` | `MD`, `Phase`, `VelocityVerlet`, `FixLangevin`, `NeighborList`, `MaxwellBoltzmann` |
| `molrs.conformer` | `molrs::conformer` | `Conformer`, `ConformerReport`, `ConformerStageReport` |
| `molrs.builder` | `molrs::builder` | `GrapheneBuilder`, `CarbonTubeBuilder`, `Assembler` and its placers / orienter, `Coarsener` |
| `molrs.compute` | `molrs::compute` | one class per compute type, flat; their results; `ComputeSpec`; the `Compute` protocol |
| `molrs.stream` | `molrs::stream` | `Publisher`, `PublisherConfig`, `ControlCommand` |

### Where Python's shape differs from Rust's

These binding classes have no one-to-one Rust type. Each is deliberate:
it makes the Python object behave the way Python code uses it, and the
Rust names it maps onto are listed here.

| Python | Rust | Why |
|---|---|---|
| `molrs.core.RelationRef` and its per-kind subclasses `Bond` (`itom`, `jtom`), `Angle`, `Dihedral`, `Improper`, `CgBond`, `Port` (`anchor`, `handle_atom`) | one `molrs::core::Relation` (a kind name, endpoint handles and fields) | A view's class is its relation kind, so `isinstance(r, Bond)` and the named endpoints work; every view reads and writes the same graph row a `Relation` does. A user-registered kind is a plain `RelationRef`. |
| `molrs.ff.typifier.Typifier` and the built-in typifiers (`Mmff94Typifier`, `GaffTypifier`, …): `assign`, `typify`, `forcefield` | the trait `molrs::ff::typifier::Typifier` (`assign`, `source_forcefield`) plus the driver `Typing<T>` (`typify`, `forcefield`) | Rust splits them because a trait cannot hold the output force field that typing accumulates; a Python class can, so one object is both. `Typifier.assign` is the trait's method; `typify` and `forcefield` are `Typing`'s. |
| `molrs.compute.CombinedDistribution` and its `CombinedDistributionResult` | `CombinedDistribution<N>` / `CombinedDistributionResult<N>` for `N` = 2, 3 | A Python class cannot be generic over a constant, so the number of `observables` picks `N`; `spec()` is the one registered `CombinedDistribution` descriptor. |
| `molrs.compute.PmftResult`, `molrs.compute.Histogram` | `PmftResult<C>` for each `Pmft<C>` (`PmftR12`, `PmftXy`, `PmftXyt`, `PmftXyz`); `Histogram<N>` | One result class per generic type: the arrays carry their own dimension. |

## `molrs.core`

::: molrs.core.Block

::: molrs.core.Frame

::: molrs.core.FrameMeta

::: molrs.core.MetaValue

Every door of `frame.meta` hands back a frozen value: a fixed-length vector
is a `tuple`, and a JSON object is a `MetaDocument`. Nested arrays are
tuples. `json.dumps` accepts a tuple and rejects a document — use
`json.dumps(frame.meta["run"].copy())`. Order inside a nested document is
unspecified.

::: molrs.core.MetaDocument

::: molrs.core.Trajectory

::: molrs.core.ObservableRecord

::: molrs.core.Atomistic

::: molrs.core.CoarseGrain

::: molrs.core.MolGraph

::: molrs.core.BondOrder

::: molrs.core.BondNumber

`Atomistic` and `CoarseGrain` are not `MolGraph` subclasses: each holds its
own typed graph (Rust `molrs::core::Atomistic` / `CoarseGrain`) and speaks
only its vocabulary. Every entity they hand out or take is a live view
(`Atom`, `Bond`, `Bead`, …), and a view's mapping is the one door to its
fields: `atom["x"]`, `atom["x"] = 1.0`, `del atom["charge"]`. Removal is
`del_atom(atom)` for one atom and `del_atoms(atoms)` for several (atomic),
and likewise for bonds, angles, dihedrals, impropers, ports, beads and terms
(`add_term(block, atoms, **props)` for any other relation block). A bond's
class is written only by `set_bond_class(bond, BondOrder, BondNumber)`.
Copies are `copy.copy(mol)` / `copy.deepcopy(mol)`.

Rigid-body moves are methods of `Atomistic` and `CoarseGrain`, with the same
signatures on both: `translate(delta)`, `rotate(axis, angle, about=None)`
and `scale(factor, about=None)` move every node that has coordinates in
place, and `center(nodes)` is the mass-weighted centre of the given nodes.

::: molrs.core.NodeRef

::: molrs.core.RelationRef

::: molrs.core.Refs

::: molrs.core.Cell

::: molrs.core.Box

A region is a solid with a signed distance to its boundary: every class
answers `contains`, `distance` (negative inside) and `bounds`, and composes
with `&`, `|` and `~`. Outside a shape is `~shape`; a shell is
`outer & ~inner`. `TriMesh` is the surface a `Polyhedron` is bounded by and
what `molrs.io.read_stl` reads (from a path or from the file's `bytes`).

::: molrs.core.Sphere

::: molrs.core.Cuboid

::: molrs.core.Parallelepiped

::: molrs.core.HalfSpace

::: molrs.core.Cylinder

::: molrs.core.Ellipsoid

::: molrs.core.TriMesh

::: molrs.core.Polyhedron

::: molrs.core.SphereUnion

::: molrs.core.Region

::: molrs.core.NeighborList

::: molrs.core.Neighbors

::: molrs.core.NeighborQuery

## `molrs.perceive`

Two verbs: `perceive_<fact>` is a query — it reports a side table and leaves
the graph alone; `assign_<fact>` is a mutation — it writes the fact onto the
molecule it is given, in place, and returns `None` (`assign_kekule_bond_orders`
returns whether every aromatic bond got a legal number). Copy first
(`copy.copy(mol)`) to keep the input. `add_hydrogens` is a graph edit and
returns a new graph. Atoms in results are the molecule's own views.

::: molrs.perceive.perceive_rings

::: molrs.perceive.perceive_rotatable_bonds

::: molrs.perceive.RotatableBond

::: molrs.perceive.assign_rings

::: molrs.perceive.assign_aromaticity

::: molrs.perceive.add_hydrogens

::: molrs.perceive.assign_stereo

::: molrs.perceive.assign_rotatable_bonds

::: molrs.perceive.assign_bond_orders

::: molrs.perceive.assign_kekule_bond_orders

::: molrs.perceive.assign_bcc_bond_types

::: molrs.perceive.BondOrderSource

::: molrs.perceive.UnknownBondPolicy

::: molrs.perceive.assign_equivalence_classes

::: molrs.perceive.RingSet

::: molrs.perceive.Reaction

::: molrs.perceive.ReactionBatch

## `molrs.io`

One module per file format. Every format has one door per direction:
`read_X(source)` reads a file's first (or only) frame and `write_X(sink,
frame)` writes one frame as a whole file. A source is an `os.PathLike`, the
`bytes` of a whole file, or a `molrs.io.Text`; a plain `str` is refused
(ambiguous between a path and text). A sink is an `os.PathLike` or a
`bytearray` (cleared, then holding exactly the file's bytes). A multi-frame
format's classes live in its submodule: `molrs.io.X.<X>Reader(source)`
(`len`, `reader[i]` / `reader[a:b]`, iteration) and
`molrs.io.X.<X>Writer(sink)` (`write(frame)`, `close()`, context manager).
Family formats carry the family name (`read_lammps_data`, `read_prmtop`,
`read_poscar`). No door picks the format for the caller. Every reader emits
the canonical column names (`molrs.core.keys`).

::: molrs.io.Text

### Structure files

::: molrs.io.read_pdb

::: molrs.io.write_pdb

::: molrs.io.read_xyz

::: molrs.io.write_xyz

::: molrs.io.read_gro

::: molrs.io.write_gro

::: molrs.io.read_sdf

::: molrs.io.read_mol2

::: molrs.io.write_mol2

::: molrs.io.read_cif

::: molrs.io.write_cif

::: molrs.io.read_xsf

::: molrs.io.write_xsf

::: molrs.io.read_cube

::: molrs.io.write_cube

::: molrs.io.read_poscar

::: molrs.io.write_poscar

::: molrs.io.read_chgcar

::: molrs.io.read_stl

### Multi-frame readers and writers

::: molrs.io.pdb.PdbReader

::: molrs.io.pdb.PdbWriter

::: molrs.io.xyz.XyzReader

::: molrs.io.xyz.XyzWriter

::: molrs.io.gro.GroReader

::: molrs.io.gro.GroWriter

::: molrs.io.mol2.Mol2Reader

::: molrs.io.mol2.Mol2Writer

::: molrs.io.cif.CifReader

::: molrs.io.cif.CifWriter

::: molrs.io.sdf.SdfReader

::: molrs.io.cube.CubeReader

::: molrs.io.read_dcd

::: molrs.io.write_dcd

::: molrs.io.dcd.DcdReader

::: molrs.io.dcd.DcdWriter

::: molrs.io.read_trr

::: molrs.io.write_trr

::: molrs.io.trr.TrrReader

::: molrs.io.trr.TrrWriter

::: molrs.io.read_xtc

::: molrs.io.write_xtc

::: molrs.io.xtc.XtcReader

::: molrs.io.xtc.XtcWriter

### LAMMPS (`molrs.io.lammps`)

::: molrs.io.read_lammps_data

::: molrs.io.write_lammps_data

::: molrs.io.read_lammps_molecule

::: molrs.io.write_lammps_molecule

::: molrs.io.read_lammps_molecule_json

::: molrs.io.write_lammps_molecule_json

::: molrs.io.read_lammps_dump

::: molrs.io.write_lammps_dump

::: molrs.io.lammps.LammpsDumpReader

::: molrs.io.lammps.LammpsDumpWriter

::: molrs.io.write_lammps_dump_local

::: molrs.io.lammps.BondReactTemplate

::: molrs.io.write_lammps_bond_react_map

::: molrs.io.write_lammps_bond_react_system

::: molrs.io.read_lammps_log

::: molrs.io.lammps.is_lammps_log

### AMBER

::: molrs.io.read_prmtop

::: molrs.io.read_inpcrd

::: molrs.io.read_ac

::: molrs.io.read_prep

::: molrs.io.write_prep

### Record files (`molrs.io.mrec`)

The [Record files guide](../guides/records.md) shows these in use.

::: molrs.io.write_mrec

::: molrs.io.mrec.MrecFile

::: molrs.io.mrec.SequenceSchema

::: molrs.io.mrec.MrecWriter

::: molrs.io.mrec.MrecReader

::: molrs.io.mrec.ForceFieldSection

::: molrs.io.mrec.pack_mrec_zip

### Force-field files

Each format is a reader and / or writer class: `read(source) -> ForceField`,
`write(sink, forcefield)`, and the format's other files as methods of the
same class.

::: molrs.io.lammps.LammpsForcefieldReader

::: molrs.io.lammps.LammpsForcefieldWriter

::: molrs.io.gromacs.GromacsTopForcefieldReader

::: molrs.io.gromacs.GromacsTopForcefieldWriter

::: molrs.io.amber.AmberPrmtopForcefieldReader

::: molrs.io.amber.AmberFrcmodWriter

::: molrs.io.openmm_xml.OpenmmXmlReader

::: molrs.io.openmm_xml.OpenmmXmlWriter

::: molrs.io.xml.XmlForcefieldReader

::: molrs.io.xml.XmlForcefieldWriter

### SMILES and CGsmiles

::: molrs.io.smiles.SmilesIr

::: molrs.io.smiles.FragmentIr

::: molrs.io.smiles.SmilesError

::: molrs.io.cgsmiles.CgSmilesIr

### Wire-encoded frames and CSV blocks

::: molrs.io.read_wire_frame

::: molrs.io.write_wire_frame

::: molrs.io.read_clpol_alpha

::: molrs.io.read_csv_block

::: molrs.io.write_csv_block

## `molrs.ff`

### `molrs.ff.forcefield`

The native force-field model exposes a `Style`/`ForceFieldType` handle hierarchy
(`BondStyle`/`BondType`, `PairStyle`/`PairType`, `CmapStyle`/`CmapType`,
…); a handle's `params` is a plain dict of numbers, strings and float64
arrays (a CMAP `grid`).

::: molrs.ff.forcefield.ForceField

::: molrs.ff.forcefield.Style

::: molrs.ff.forcefield.AtomStyle

::: molrs.ff.forcefield.BondStyle

::: molrs.ff.forcefield.AngleStyle

::: molrs.ff.forcefield.DihedralStyle

::: molrs.ff.forcefield.ImproperStyle

::: molrs.ff.forcefield.PairStyle

::: molrs.ff.forcefield.CmapStyle

::: molrs.ff.forcefield.ForceFieldType

::: molrs.ff.forcefield.AtomType

::: molrs.ff.forcefield.BondType

::: molrs.ff.forcefield.AngleType

::: molrs.ff.forcefield.DihedralType

::: molrs.ff.forcefield.ImproperType

::: molrs.ff.forcefield.PairType

::: molrs.ff.forcefield.CmapType

### `molrs.ff.compile`

::: molrs.ff.compile.PotentialCompiler

::: molrs.ff.compile.ExplicitTerms

### `molrs.ff.potential`

Every potential evaluates energy and forces together, the one door of the core
`Potential::evaluate` into an `Accumulator`: `Potentials.evaluate(coords)`, a
Python provider's `evaluate(pos)`, and for the pair kernel
`PairLjCut.evaluate_pairs(pos, neighbors)` (`PairDriven::evaluate_pairs`) and
`PairLjCut.pair_energy_force(r2, disp)`.

::: molrs.ff.potential.Potentials

::: molrs.ff.potential.PairLjCut

### `molrs.ff.typifier`

::: molrs.ff.typifier.Mmff94Typifier

::: molrs.ff.typifier.Mmff94sTypifier

::: molrs.ff.typifier.OplsAaTypifier

::: molrs.ff.typifier.AtdTypifier

::: molrs.ff.typifier.GaffTypifier

::: molrs.ff.typifier.Typifier

::: molrs.ff.typifier.TypeAssignment

### `molrs.ff.charge`

::: molrs.ff.charge.GasteigerModel

### `molrs.ff.params`

::: molrs.ff.params.clpol_polarizability

## `molrs.optimize`

`Lbfgs` mirrors `molrs::optimize::Lbfgs`: `minimize(frame)` honours the
frame's `atoms.free` mask, `minimize_coords(coords)` relaxes one `(N, 3)`
array, `minimize_batch(coords)` a `(B, N, 3)` stack; each relaxes its input in
place and returns the report(s).

::: molrs.optimize.Lbfgs

::: molrs.optimize.OptimizationReport

## `molrs.conformer`

::: molrs.conformer.Conformer

::: molrs.conformer.ConformerStageReport

::: molrs.conformer.ConformerReport

## `molrs.builder`

::: molrs.builder.Coarsener

## `molrs.compute`

The Rust compute module is flat, and so is `molrs.compute`: every class is
`molrs.compute.<Name>`, the core type of the same name. A compute's
constructor takes configuration only — every optional setting is `None`
unless given, and every default lives in the core — and `compute(...)` takes
the data by keyword. A result is a read-only object that the next compute
takes as is (`Msd` → `TimeCorrelation` → `LinearFit`, `Steinhardt` →
`SteinhardtResult` → `Pca`, `Cluster` → `ClusterResult` → `CenterOfMass`).

`Msd`, `Vacf` and `Rdf` are streams only, as in the core: `x.accumulator()`
hands out a `CorrelatorAccumulator` / `RdfAccumulator`; `push` one frame at a
time and `finish()` reads the result (without consuming the accumulator).

### Contract and descriptors

Every class is configured at construction and answers one call, `compute`,
whose keyword arguments are the fields of the core input — or, for a stream,
`accumulator`. `ComputeSpec` describes each one: its parameters (kind,
default, required) and inputs (for a stream, the fields `push` takes).

::: molrs.compute.Compute

::: molrs.compute.Stream

::: molrs.compute.ComputeSpec

::: molrs.compute.ParamSpec

::: molrs.compute.InputSpec

### Values

::: molrs.compute.Axis

::: molrs.compute.RegularGrid

::: molrs.compute.Histogram

### Time series

`Correlator` is the one estimator; `Msd`, `Vacf`, `OnsagerCorrelation`,
`DipoleAcf`, `CurrentAcf`, `ReorientationAcf`, `RotationalAutocorrelation`
and `Survival` are its fixed configurations (`Msd` and `Vacf` streamed through
a `CorrelatorAccumulator`; `Vacf` takes each frame's velocities as an
`(n_dof, 1)` column). Each returns a `TimeCorrelation`, which the fits and spectra take as is (and which pickles).
The origin set is tunable (`origins`, and `stride` for every `stride`-th
origin), as is the normalisation (`"unbiased"`, `"biased"`, `"none"`).
`ReorientationAcf` is the orientational autocorrelation (OACF) of unit
vectors; orders 1 and 2 run by FFT. A curve's derived quantities are the fits
applied to it: `CumulativeTrapezoid` (integral, correlation time), `LinearFit`
(Einstein slope), `Plateau` (Green–Kubo plateau), `DebyeFit` (exponential
decay), and `Spectrum` (power spectrum).

::: molrs.compute.Correlator

::: molrs.compute.TimeCorrelation

::: molrs.compute.Msd

::: molrs.compute.CorrelatorAccumulator

::: molrs.compute.Vacf

::: molrs.compute.OnsagerCorrelation

::: molrs.compute.DipoleAcf

::: molrs.compute.CurrentAcf

::: molrs.compute.ReorientationAcf

::: molrs.compute.RotationalAutocorrelation

::: molrs.compute.Survival

::: molrs.compute.Derivative

::: molrs.compute.Unwrap

::: molrs.compute.Spectrum

::: molrs.compute.OneSidedSpectrum

### Fits

::: molrs.compute.LinearFit

::: molrs.compute.LinearFitResult

::: molrs.compute.Plateau

::: molrs.compute.PlateauResult

::: molrs.compute.DebyeFit

::: molrs.compute.DebyeFitResult

::: molrs.compute.CumulativeTrapezoid

### Dielectric and spectroscopy

::: molrs.compute.KineticEnergy

::: molrs.compute.KineticEnergyResult

::: molrs.compute.DipoleMoment

::: molrs.compute.DipoleMomentResult

::: molrs.compute.StaticDielectric

::: molrs.compute.StaticDielectricResult

::: molrs.compute.EinsteinHelfandSpectrum

::: molrs.compute.GreenKuboSpectrum

::: molrs.compute.DipoleRateCrossSpectrum

::: molrs.compute.DipoleAutocorrelationSpectrum

::: molrs.compute.DielectricSpectrumResult

::: molrs.compute.KramersKronig

::: molrs.compute.KramersKronigCheck

::: molrs.compute.ConductivitySumRule

::: molrs.compute.SumRuleCheck

::: molrs.compute.RouteAgreement

::: molrs.compute.RouteAgreementCheck

::: molrs.compute.PolarizabilityParts

::: molrs.compute.PolarizabilityPartsResult

::: molrs.compute.RamanSpectrum

::: molrs.compute.RamanSpectrumResult

### Binned and gridded structure

`Rdf`, `DistributionFunction`, `CombinedDistribution` and
`SpatialDistribution` fold frames: their `compute` takes one list per input
field, an entry per frame.

::: molrs.compute.Rdf

::: molrs.compute.RdfAccumulator

::: molrs.compute.RdfResult

::: molrs.compute.CorrelationFunction

::: molrs.compute.CorrelationFunctionResult

::: molrs.compute.LocalDensity

::: molrs.compute.LocalDensityResult

::: molrs.compute.GaussianDensity

::: molrs.compute.GaussianDensityResult

::: molrs.compute.SphereVoxelization

::: molrs.compute.SphereVoxelizationResult

::: molrs.compute.SpatialDistribution

::: molrs.compute.SpatialDistributionResult

::: molrs.compute.DistributionFunction

::: molrs.compute.DistributionResult

::: molrs.compute.CombinedDistribution

::: molrs.compute.CombinedDistributionResult

::: molrs.compute.StaticStructureFactorDebye

::: molrs.compute.StaticStructureFactorDebyeResult

::: molrs.compute.StaticStructureFactorDirect

::: molrs.compute.StaticStructureFactorDirectResult

::: molrs.compute.DiffractionPattern

::: molrs.compute.DiffractionPatternResult

### Dynamics

::: molrs.compute.PairPresence

::: molrs.compute.Presence

::: molrs.compute.VanHove

::: molrs.compute.VanHoveResult

::: molrs.compute.HBonds

::: molrs.compute.HBondsResult

::: molrs.compute.HBond

### Order and environment

::: molrs.compute.Steinhardt

::: molrs.compute.SteinhardtResult

::: molrs.compute.QlmTable

::: molrs.compute.Hexatic

::: molrs.compute.HexaticResult

::: molrs.compute.SolidLiquid

::: molrs.compute.SolidLiquidResult

::: molrs.compute.ContinuousCoordination

::: molrs.compute.ContinuousCoordinationResult

::: molrs.compute.Nematic

::: molrs.compute.NematicResult

::: molrs.compute.Cubatic

::: molrs.compute.CubaticResult

::: molrs.compute.BondOrientationalOrder

::: molrs.compute.BondOrientationalOrderResult

::: molrs.compute.LocalDescriptors

::: molrs.compute.LocalDescriptorsResult

::: molrs.compute.LocalBondProjection

::: molrs.compute.LocalBondProjectionResult

::: molrs.compute.EnvironmentMatch

::: molrs.compute.EnvironmentMatchResult

::: molrs.compute.AngularSeparationGlobal

::: molrs.compute.AngularSeparationGlobalResult

::: molrs.compute.AngularSeparationNeighbor

::: molrs.compute.AngularSeparationNeighborResult

::: molrs.compute.PmftR12

::: molrs.compute.PmftXy

::: molrs.compute.PmftXyt

::: molrs.compute.PmftXyz

::: molrs.compute.PmftResult

### Groups and features

A partition argument is an unsigned-int label array or any partition result
(`ClusterResult`, `KmeansResult`, `EnvironmentMatchResult`,
`VoronoiDomainResult`); a feature matrix is a 2-D float array or a descriptor
result (`SteinhardtResult`, `LocalDescriptorsResult`, `PcaResult`).

::: molrs.compute.Cluster

::: molrs.compute.ClusterResult

::: molrs.compute.CenterOfMass

::: molrs.compute.CenterOfMassResult

::: molrs.compute.GyrationTensor

::: molrs.compute.GyrationTensorResult

::: molrs.compute.InertiaTensor

::: molrs.compute.InertiaTensorResult

::: molrs.compute.Pca

::: molrs.compute.PcaResult

::: molrs.compute.Kmeans

::: molrs.compute.KmeansResult

### Voronoi

::: molrs.compute.RadicalVoronoi

::: molrs.compute.VoronoiCells

::: molrs.compute.VoronoiFace

::: molrs.compute.VoronoiDomainAnalysis

::: molrs.compute.VoronoiDomainResult

::: molrs.compute.VoronoiVoidAnalysis

::: molrs.compute.VoronoiVoidResult

::: molrs.compute.VoronoiIntegration

::: molrs.compute.MolecularMoments

::: molrs.compute.FiniteFieldPolarizability
