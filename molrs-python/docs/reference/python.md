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
| `molrs.core` | `molrs::core` | `Block`, `Frame`, `FrameMeta`, `MetaValue`, `MetaDocument`, `Trajectory`, `ObservableRecord`, `BlockDtypeError`; `Box` (Rust `SimBox`), `NeighborList`, `Neighbors`, `NeighborQuery`, `VerletSkin`, the regions, `TriMesh`, `Trace`; `MolGraph`, `Atomistic`, `CoarseGrain`, the node / relation views, `ExtractedSubgraph`, `Element`, `BondOrder`, `BondNumber`, `Topology`; `Unit`, `Quantity`, `UnitRegistry`, `UnitPreset`, `UnitsError` |
| `molrs.core.keys` | `molrs::core::keys` | the canonical column, frame-meta and graph keys |
| `molrs.core.schema` | `molrs::core::schema` | `ColumnSpec`, `BlockSpec`, the block names, `relation_endpoints` |
| `molrs.core.constants` | `molrs::core::constants` | every physical and engine constant (`AVOGADRO`, `COULOMB_REAL`, `AMBER_COULOMB`, `AMBER_SCEE`, …) |
| `molrs.op` | `molrs::op` | `superpose`, `centroid`, `place_from_internal_coords`, `Superposition`, `DEFAULT_GAP_TOL` |
| `molrs.perceive` | `molrs::perceive` | `perceive_rings` → `RingSet`, the `perceive_*` / `assign_*` functions, `add_hydrogens`, `SmartsPattern`, `SmartsMatch`, `Reaction`, `SubgraphMatcher` |
| `molrs.io` | `molrs::io` | every file reader and writer, as a function `read_<fmt>[_<what>]` / `write_<fmt>[_<what>]` (`_str` / `_bytes` in memory): structure, trajectory and force-field files, `*.mrec` records (`read_mrec_frame` / `write_mrec_frame` and partners), wire-encoded frames, SMILES and CGsmiles text, the LAMMPS log, CSV blocks |
| `molrs.io.pdb`, `.xyz`, `.gro`, `.dcd`, `.trr`, `.xtc` | `molrs::io::{pdb, xyz, gro, dcd, trr, xtc}` | each format's lazy reader: `PdbReader`, `XyzReader`, `GroReader`, `DcdReader`, `TrrReader`, `XtcReader` |
| `molrs.io.lammps` | `molrs::io::lammps` | `LammpsDumpReader`, `BondReactTemplate`, the `Lammps*` log records, `infer_lammps_atom_style` |
| `molrs.io.smiles` | `molrs::io::smiles` | `SmilesIr`, `SmilesError`, `BondingDescriptor` |
| `molrs.io.cgsmiles` | `molrs::io::cgsmiles` | `CgSmilesIr` and the CGsmiles records |
| `molrs.io.mrec` | `molrs::io::mrec` | `MrecReader`, `MrecWriter`, `SequenceSchema`, `ForceFieldSection`, `section_names`, `pack_mrec_zip`, `validation` |
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
| `molrs.md` | `molrs::md` | `VelocityVerlet`, `Langevin`, `MdState`, `MaxwellBoltzmann`, `MdDriver` |
| `molrs.conformer` | `molrs::conformer` | `Conformer`, `ConformerReport`, `ConformerStageReport` |
| `molrs.builder` | `molrs::builder` | `GrapheneBuilder`, `CarbonTubeBuilder`, `Assembler` and its placers / orienter, `Coarsener` |
| `molrs.compute` | `molrs::compute` | one class per compute type, flat; their results; `ComputeSpec`; the `Compute` protocol |
| `molrs.stream` | `molrs::stream` | `Publisher`, `ControlCommand` |

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

Rigid-body moves are methods of `Atomistic` and `CoarseGrain`, not module
functions: `translate(delta)`, `rotate(axis, angle, about=None)` and
`scale(factor, about=None)`. Each moves every node that has coordinates in
place and returns the graph itself, so moves chain:
`mol.translate([1, 0, 0]).rotate([0, 0, 1], 0.5).scale([2, 2, 2])`.
In Rust the moves are `molrs::op`'s functions (`translate`, `scale`, `rotate`, `center`) over a `MolGraph`.

::: molrs.core.Box

A region is a solid with a signed distance to its boundary: every class
answers `contains`, `distance` (negative inside) and `bounds`, and composes
with `&`, `|` and `~`. Outside a shape is `~shape`; a shell is
`outer & ~inner`. `TriMesh` is the surface a `Polyhedron` is bounded by and
what `molrs.io.read_stl` reads (`molrs.io.read_stl_bytes` from bytes, as the
WASM binding's `readStlBytes` does).

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

Two verbs: `perceive_<fact>` reports a side table and leaves the graph
alone; `assign_<fact>` writes the fact onto a clone and returns it.

::: molrs.perceive.perceive_rings

::: molrs.perceive.assign_rings

::: molrs.perceive.assign_aromaticity

::: molrs.perceive.add_hydrogens

::: molrs.perceive.assign_stereo

::: molrs.perceive.assign_rotatable_bonds

::: molrs.perceive.assign_bond_orders

::: molrs.perceive.assign_kekule_bond_orders

::: molrs.perceive.assign_bcc_bond_types

::: molrs.perceive.assign_bcc_bond_types_from_connectivity

::: molrs.perceive.assign_equivalence_classes

::: molrs.perceive.RingSet

::: molrs.perceive.Reaction

## `molrs.io`

One module per file format. Every door is a function named after its format:
`read_X` / `write_X` for one frame, `read_X_trajectory` /
`write_X_trajectory` for a sequence, `_str` / `_bytes` for text and bytes in
memory; family formats carry the family name (`read_lammps_data`,
`read_prmtop`, `read_poscar`). No door picks the format for the
caller. Every reader emits the canonical column names (`molrs.core.keys`).
Each `read_X_trajectory` returns its format's lazy reader,
`molrs.io.X.<X>Reader`, over one path or a list of paths.

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

### Trajectories

::: molrs.io.read_pdb_trajectory

::: molrs.io.write_pdb_trajectory

::: molrs.io.pdb.PdbReader

::: molrs.io.read_xyz_trajectory

::: molrs.io.write_xyz_trajectory

::: molrs.io.xyz.XyzReader

::: molrs.io.read_gro_trajectory

::: molrs.io.write_gro_trajectory

::: molrs.io.gro.GroReader

::: molrs.io.read_dcd_trajectory

::: molrs.io.write_dcd_trajectory

::: molrs.io.dcd.DcdReader

::: molrs.io.read_trr_trajectory

::: molrs.io.write_trr_trajectory

::: molrs.io.trr.TrrReader

::: molrs.io.read_xtc_trajectory

::: molrs.io.write_xtc_trajectory

::: molrs.io.xtc.XtcReader

### LAMMPS (`molrs.io.lammps`)

::: molrs.io.read_lammps_data

::: molrs.io.write_lammps_data

::: molrs.io.lammps.infer_lammps_atom_style

::: molrs.io.read_lammps_molecule

::: molrs.io.write_lammps_molecule

::: molrs.io.read_lammps_molecule_json

::: molrs.io.write_lammps_molecule_json

::: molrs.io.read_lammps_dump_trajectory

::: molrs.io.write_lammps_dump_trajectory

::: molrs.io.write_lammps_dump_local

::: molrs.io.lammps.LammpsDumpReader

::: molrs.io.lammps.BondReactTemplate

::: molrs.io.write_lammps_bond_react_map

::: molrs.io.write_lammps_bond_react_system

::: molrs.io.read_lammps_log

::: molrs.io.read_lammps_log_str

### AMBER

::: molrs.io.read_prmtop

::: molrs.io.read_inpcrd

::: molrs.io.read_ac

::: molrs.io.read_prep

::: molrs.io.write_prep

### Record files (`molrs.io.mrec`)

The [Record files guide](../guides/records.md) shows these in use.

::: molrs.io.read_mrec_frame

::: molrs.io.write_mrec_frame

::: molrs.io.read_mrec_system

::: molrs.io.write_mrec_system

::: molrs.io.read_mrec_trajectory

::: molrs.io.write_mrec_trajectory

::: molrs.io.read_mrec_forcefield

::: molrs.io.write_mrec_forcefield

::: molrs.io.mrec.section_names

::: molrs.io.read_mrec_meta

::: molrs.io.mrec.SequenceSchema

::: molrs.io.mrec.MrecWriter

::: molrs.io.mrec.MrecReader

::: molrs.io.mrec.ForceFieldSection

::: molrs.io.mrec.pack_mrec_zip

### Force-field files

::: molrs.io.read_lammps_forcefield

::: molrs.io.write_lammps_forcefield

::: molrs.io.read_lammps_data_coeffs

::: molrs.io.read_lammps_data_coeffs_str

::: molrs.io.write_lammps_data_coeffs_str

::: molrs.io.read_top_forcefield

::: molrs.io.write_top_forcefield

::: molrs.io.read_top_system

::: molrs.io.write_top_system

::: molrs.io.read_prmtop_forcefield

::: molrs.io.read_prmtop_system

::: molrs.io.write_frcmod

::: molrs.io.read_openmm_forcefield

::: molrs.io.write_openmm_forcefield

::: molrs.io.write_openmm_system

::: molrs.io.write_openmm_system_str

::: molrs.io.read_xml_forcefield

::: molrs.io.write_xml_forcefield

### SMILES and CGsmiles

::: molrs.io.read_smiles_str

::: molrs.io.write_smiles_str

::: molrs.io.smiles.SmilesIr

::: molrs.io.smiles.SmilesError

::: molrs.io.read_cgsmiles_str

::: molrs.io.cgsmiles.CgSmilesIr

### Wire-encoded frames and CSV blocks

::: molrs.io.read_msgpack_frame_bytes

::: molrs.io.write_msgpack_frame_bytes

::: molrs.io.read_json_frame_str

::: molrs.io.write_json_frame_str

::: molrs.io.read_clpol_alpha

::: molrs.io.read_clpol_alpha_str

::: molrs.io.read_csv_block

::: molrs.io.read_csv_block_str

::: molrs.io.write_csv_block

::: molrs.io.write_csv_block_str

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

::: molrs.ff.potential.Potentials

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

### Contract and descriptors

Every class is configured at construction and answers one call, `compute`,
whose keyword arguments are the fields of the core input. `ComputeSpec`
describes each one: its parameters (kind, default, required) and inputs.

::: molrs.compute.Compute

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
and `Survival` are its fixed configurations. Each returns a
`TimeCorrelation`, which the fits and spectra take as is (and which pickles).
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
