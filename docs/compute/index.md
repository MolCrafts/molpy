# Compute

The **Compute** layer turns a `Trajectory` or `Frame` into physical observables:
structural distributions, dynamical correlations, and spectra. Every analysis
follows one uniform pattern, so once you have used one you have used them all.

## The Compute → Result pattern

Each analysis is a small configurable object built once and then *called* on
data. Calling it returns a typed `Result` dataclass — never a bare tuple — so
outputs are self-describing and serializable.

```python
from molpy.compute import RDF

rdf = RDF(r_max=10.0, n_bins=100)   # 1. configure
result = rdf(trajectory)            # 2. call on data -> Result
result.to_dict()                    # 3. inspect / serialize
```

Heavy numerical kernels (autocorrelation, FFT, spectral prefactors) are
implemented in Rust inside `molrs`; the MolPy classes handle data extraction,
periodic-image unwrapping, and vectorized assembly, then delegate the physics.

!!! note "Units"
    Compute kernels use LAMMPS *real* units: length Å, charge $e$, time ps,
    volume Å³, temperature K, angular frequency rad·ps⁻¹. GROMACS trajectories
    are read in native nm — scale lengths by 10 before analysis.

## Available analyses

| Method | Class / entry point | Returns | Measures |
|--------|---------------------|---------|----------|
| **Dielectric spectroscopy** | `DielectricSusceptibility` | `DielectricSusceptibilityResult` | $\varepsilon^*(\omega)$, $\varepsilon(0)$, $\sigma$ |
| **Ionic conductivity** | `IonicConductivity` | `ConductivityResult` | $\sigma$ (S/m) via Einstein-Helfand |
| Autocorrelation | `ACFAnalyzer` | `ACFResult` | time-correlation $C(t)$ |
| Time → frequency | `SpectralAnalyzer` | `SpectralResult` | windowed spectrum |
| Mean displacement correlation | `MCDCompute` | `MCDResult` | diffusion / MSD per group |
| Polarization MSD | `PMSDCompute` | `PMSDResult` | collective charge transport |
| Radial distribution | `RDF` | structural $g(r)$ | pair structure |
| Mean-squared displacement | `MSD` | time series | single-particle diffusion |
| Shape descriptors | `RadiusOfGyration`, `GyrationTensor`, `InertiaTensor`, `CenterOfMass` | per-frame tensors/scalars | molecular shape |
| Clustering / decomposition | `Cluster`, `ClusterCenters`, `Pca`, `KMeans` | labels / components | grouping & dimensionality reduction |

Analyses can be chained into a directed graph with `Workflow` for multi-step
pipelines (e.g. dipole → ACF → spectrum).

## Featured guide

- **[Dielectric Spectroscopy](dielectric.md)** — a complete, textbook-style
  derivation of $\varepsilon^*(\omega)$ and the ionic conductivity $\sigma$ from
  an MD trajectory: the fluctuation–dissipation basis, the Einstein–Helfand and
  Green–Kubo routes, every numerical choice (windowing, FFT, unbiased ACF), the
  electrolyte dipole decomposition, and the spectral fitting recipes
  (Debye, Cole–Cole, Havriliak–Negami).

## Related

- [API reference: Compute](../api/compute.md) — autodoc for the classes above.
- [Concepts: Trajectory](../tutorials/05_trajectory.md) — the input data model.
- [Concepts: Box and Periodicity](../tutorials/03_box_and_periodicity.md) — minimum-image conventions used by dynamical analyses.
