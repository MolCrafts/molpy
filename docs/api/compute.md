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
| `RadiusOfGyration`, `GyrationTensor`, `InertiaTensor`, `CenterOfMass` | Molecular shape descriptors | per-frame values |
| `Cluster`, `ClusterCenters`, `Pca`, `KMeans` | Clustering & decomposition | labels / components |
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

### Shape descriptors

::: molpy.compute.shape

### Decomposition

::: molpy.compute.decomposition

### Clustering

::: molpy.compute.cluster

### Time series

::: molpy.compute.time_series

### Workflow

::: molpy.compute.workflow
