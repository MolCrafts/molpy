# Compute

Trajectory and structure analyses. All available via `from molpy.compute import ...`.

See the [Compute overview](../compute/index.md) for the Compute → Result pattern
and the [Dielectric Spectroscopy](../compute/dielectric.md) guide for the theory
behind the dielectric classes.

## Quick reference

| Symbol | Summary | Returns |
|--------|---------|---------|
| `Compute` | Base class: configure once, call on data | a `Result` subclass |
| `DielectricSusceptibility` | $\varepsilon^*(\omega)$ via Einstein–Helfand / Green–Kubo | `DielectricSusceptibilityResult` |
| `IonicConductivity` | Ionic conductivity $\sigma$ (Einstein–Helfand MSD) | `ConductivityResult` |
| `ACFAnalyzer` | Autocorrelation of selected columns | `ACFResult` |
| `SpectralAnalyzer` | Windowed time→frequency transform | `SpectralResult` |
| `MCDCompute` | Mean displacement correlation (diffusion) | `MCDResult` |
| `PMSDCompute` | Polarization mean-squared displacement | `PMSDResult` |
| `Onsager` | Onsager transport coefficients $L_{ij}$ (collective displacement cross-correlation) | `OnsagerResult` |
| `JACF` | Green–Kubo conductivity $\sigma$ from the current ACF | `JACFResult` |
| `Persist` | Pair-survival / residence-time correlation | `PersistResult` |
| `RDF` | Radial distribution function $g(r)$ | structural result |
| `MSD` | Single-particle mean-squared displacement | time series |
| `StaticStructureFactorDebye` | Static structure factor $S(k)$ (Debye equation) | structural result |
| `NeighborList` | Spatial neighbor-pair query within a cutoff | pair list |
| `LocalDensity`, `GaussianDensity` | Per-particle / grid number density | density field |
| `Steinhardt`, `Hexatic`, `Nematic`, `SolidLiquid` | Bond-orientational order parameters | per-particle order |
| `BondOrder` | Bond-orientational $(\theta,\phi)$ diagram | spherical histogram |
| `PMFTXY` | Potential of mean force and torque on an $(x,y)$ grid | free-energy field |
| `RadiusOfGyration`, `GyrationTensor`, `InertiaTensor`, `CenterOfMass` | Molecular shape descriptors | per-frame values |
| `Cluster`, `ClusterCenters`, `Pca`, `KMeans` | Clustering & decomposition | labels / components |
| `DistanceDistribution`, `AngleDistribution`, `DihedralDistribution` | Geometric distribution functions (distance / ADF / DDF) | distribution |
| `CombinedDistribution` | Joint multi-axis distribution (CDF) | N-D histogram |
| `SpatialDistribution` | Spatial distribution function (SDF) | body-fixed density grid |
| `VanHove` | Van Hove correlation $G(r,t)$ (self + distinct) | time-resolved $g(r,t)$ |
| `LegendreReorientation` | Reorientational TCFs $C_1(t)$, $C_2(t)$ | correlation curves |
| `HBonds`, `HBondCriterion` | Geometric hydrogen-bond detection | per-frame bond lists |
| `RadicalVoronoi`, `VoronoiIntegration`, `voronoi_domains`, `voronoi_voids` | Radical Voronoi tessellation, domains/voids, charge integration | cells / domains / moments |
| `PowerSpectrum`, `IRSpectrum`, `RamanSpectrum`, `VcdSpectrum`, `RoaSpectrum`, `ResonanceRamanSpectrum` | Vibrational spectra from ACFs (VDOS / IR / Raman / VCD / ROA) | spectrum |
| `Workflow` | Directed graph of chained computes | per-node results |

---

## Full API

### Base

::: molpy.compute.base

### Result types

::: molpy.compute.result

### Dielectric

::: molpy.compute.dielectric

### Mean displacement correlation

::: molpy.compute.mcd

### Polarization MSD

::: molpy.compute.pmsd

### Onsager coefficients

::: molpy.compute.onsager

### Current-ACF conductivity (Green–Kubo)

::: molpy.compute.jacf

### Pair persistence

::: molpy.compute.persist

### Radial distribution

::: molpy.compute.rdf

### Mean-squared displacement

::: molpy.compute.msd

### Static structure factor

::: molpy.compute.diffraction

### Neighbor list

::: molpy.compute.neighborlist

### Local & grid density

::: molpy.compute.density

### Bond-orientational order parameters

::: molpy.compute.order

### Bond-orientational environment

::: molpy.compute.environment

### Potential of mean force & torque

::: molpy.compute.pmft

### Shape descriptors

::: molpy.compute.shape

### Decomposition

::: molpy.compute.decomposition

### Clustering

::: molpy.compute.cluster

### Distribution functions (ADF / DDF / distance / combined)

::: molpy.compute.distribution

### Spatial distribution function

::: molpy.compute.spatial

### Van Hove correlation

::: molpy.compute.van_hove

### Reorientational correlations

::: molpy.compute.reorientation

### Hydrogen bonds

::: molpy.compute.hbond

### Radical Voronoi

::: molpy.compute.voronoi

### Vibrational spectra

::: molpy.compute.spectra

### Time series

::: molpy.compute.time_series

### Workflow

::: molpy.compute.workflow
