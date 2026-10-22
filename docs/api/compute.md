# Compute

Trajectory and structure analyses, reached as `mp.compute.<Name>`.

Numerical kernels live in the native core, and there is no second science
implementation in Python. Every public name on `mp.compute` is a native
class; there is no wrapper and no free function.

## A compute is configured once and fed data by keyword

Every analysis is a class with one method. The constructor takes the
configuration that defines the measurement — bins, cutoffs, `max_lag`, a
window, a tolerance — and nothing else. `compute(...)` takes the data, always
by keyword: arrays, the sample interval `dt`, neighbour tables, a partition,
or the result of another compute. Every optional setting defaults to `None`,
which means "the core default"; `X.spec()` lists a class's parameters with
their defaults and `x.tunables()` reads an instance's.

A compute never reads a `Frame`. Pulling the columns an analysis needs out of
a frame is your step, through the Frame/Block API (`frame.coords`,
`frame["atoms"]["q"]`, `frame.box`), and stacking frames into a `(T, N, 3)`
array is one `numpy.stack`. Pair-based analyses take neighbour tables built
with the core `mp.core.NeighborList` (`nl.build(frame.coords, frame.box)`,
then `nl.neighbors()`); a built `NeighborList` can also be handed over as is,
and the kernel streams its pairs without materialising a table.

## A raw curve, then a fit, then your prefactor

Every time correlation runs on one engine, `Correlator`. The chemistry-named
classes are configurations of it with the defining settings fixed: `Msd`,
`Vacf`, `OnsagerCorrelation`, `DipoleAcf`, `CurrentAcf`, `ReorientationAcf`,
`RotationalAutocorrelation` and `Survival`. Each returns a `TimeCorrelation`:
`lag_times` and `values[lag, channel]`, nothing derived. A correlation with no
named class — a VCD or ROA cross term, a rate cross-correlation — is a
`Correlator` built directly.

A fit is a compute whose input is that curve: `LinearFit`, `Plateau`,
`DebyeFit` and `CumulativeTrapezoid` take `correlation=` and `channel=`. The SI
prefactor is the last step, in your script:

| Layer | Role | Examples |
|-------|------|----------|
| Series | build the time series a correlation needs | your own arrays, `DipoleMoment`, `Derivative`, `Unwrap`, `PolarizabilityParts` |
| Raw curve | the correlation, as a `TimeCorrelation` | `Msd`, `Vacf`, `OnsagerCorrelation`, `DipoleAcf`, `CurrentAcf`, `ReorientationAcf`, `Survival`, `Correlator` |
| Fit or transform | slope, plateau, running integral, spectrum | `LinearFit`, `Plateau`, `DebyeFit`, `CumulativeTrapezoid`, `Spectrum`, the ε(ω) routes |
| Scale | MD → SI prefactor in your script | $1/(2d)$ for self-diffusion from an MSD slope, $1/(6 V k_B T)$ for Einstein conductivity, $1/(3 V k_B T)$ for Green–Kubo conductivity |

There is no all-in-one recipe class that would bury the fit window and the
unit conversion in a library default. The [MSD](../compute/msd.md),
[PMSD](../compute/pmsd.md), [JACF](../compute/jacf.md) and
[Dielectric](../compute/dielectric.md) pages show each composition.

!!! warning "Time unit"
    The composed routes work in **femtoseconds**: pass `dt` in fs, and the
    S/m prefactor on the [PMSD](../compute/pmsd.md) page
    ($3.0988\times10^{9}$) assumes a slope in
    $e^2\,\text{Å}^2\,\text{fs}^{-1}$. A `dt` in picoseconds makes the lag
    axis 1000× too short and the conductivity 1000× too large.

!!! note "Analysis units (LAMMPS *real*)"
    Length **Å**, charge **e**, **time fs**, volume Å³, temperature K.
    Vibrational spectra report cm⁻¹. GROMACS trajectories are nm-native —
    scale lengths ×10 before analysis. Displacement correlations need
    **unwrapped** coordinates; given per-frame `boxes`, `Msd` refuses a
    wrapped trajectory, and `Unwrap` produces the continuous one.

## Family index

Like freud's [API modules](https://freud.readthedocs.io/en/stable/), each
analysis family has its own page under [Compute](../compute/index.md).

| Family | Exports on `molpy.compute` | Guide |
|--------|-----------------|-------|
| neighbour search | `mp.core.NeighborList`, `mp.core.Neighbors` (core, not on `molpy.compute`) | [NeighborList](../compute/neighborlist.md) |
| rdf | `Rdf`, `CorrelationFunction` | [RDF](../compute/rdf.md) |
| density | `LocalDensity`, `GaussianDensity`, `SphereVoxelization` | [Density](../compute/density.md) |
| diffraction | `StaticStructureFactorDebye`, `StaticStructureFactorDirect`, `DiffractionPattern` | [Diffraction](../compute/diffraction.md) |
| pmft | `PmftXy`, `PmftXyt`, `PmftXyz`, `PmftR12` | [PMFT](../compute/pmft.md) |
| distribution | `DistributionFunction`, `CombinedDistribution`, `Axis` | [Distribution](../compute/distribution.md) |
| spatial | `SpatialDistribution`, `RegularGrid` | [Spatial](../compute/spatial.md) |
| order | `Steinhardt`, `Hexatic`, `SolidLiquid`, `ContinuousCoordination`, `Nematic`, `Cubatic` | [Order](../compute/order.md) |
| environment | `BondOrientationalOrder`, `LocalDescriptors`, `EnvironmentMatch`, `AngularSeparationNeighbor` | [Environment](../compute/environment.md) |
| shape | `CenterOfMass`, `GyrationTensor`, `InertiaTensor` | [Shape](../compute/shape.md) |
| cluster | `Cluster` | [Cluster](../compute/cluster.md) |
| decomposition | `Pca`, `Kmeans` | [Decomposition](../compute/decomposition.md) |
| hbond | `HBonds` | [HBond](../compute/hbond.md) |
| voronoi | `RadicalVoronoi`, `VoronoiIntegration`, `VoronoiDomainAnalysis`, `VoronoiVoidAnalysis` | [Voronoi](../compute/voronoi.md) |
| msd | `Msd`, `Unwrap` → `LinearFit` | [MSD](../compute/msd.md) |
| vacf | `Vacf` → `CumulativeTrapezoid` → `Plateau` | [VACF](../compute/vacf.md) |
| pmsd | `DipoleMoment` → `Msd` → `LinearFit` | [PMSD](../compute/pmsd.md) |
| jacf | `CurrentAcf` → `CumulativeTrapezoid` | [JACF](../compute/jacf.md) |
| onsager | `OnsagerCorrelation` | [Onsager](../compute/onsager.md) |
| persist | `PairPresence` → `Survival` | [Persist](../compute/persist.md) |
| van hove | `VanHove` | [Van Hove](../compute/van_hove.md) |
| reorientation | `ReorientationAcf`, `RotationalAutocorrelation` | [Reorientation](../compute/reorientation.md) |
| dielectric | `DipoleMoment`, `StaticDielectric`, `DipoleAcf`, the four ε(ω) routes, `DebyeFit`, `KramersKronig`, `RouteAgreement`, `ConductivitySumRule` | [Dielectric](../compute/dielectric.md) |
| spectra | `Spectrum`, `Correlator`, `Derivative`, `PolarizabilityParts`, `RamanSpectrum` | [Spectra](../compute/spectra.md) |

---

## Shared types

### The `Compute` contract

`Compute` is not a base class to inherit from. It is a structural
`typing.Protocol` owned by the native backend and re-exported here: any class
that defines a `compute(...)` method taking its data as keyword arguments
satisfies it, with no subclassing and no registration. Writing one is covered
in [Adding a Compute Operation](../developer/extending-compute.md).

::: molpy.compute.Compute
