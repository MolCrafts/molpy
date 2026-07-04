# Compute

The **Compute** layer turns a `Trajectory` or `Frame` into physical observables:
structural distributions, dynamical correlations, and spectra. Every analysis
follows one uniform pattern, so once you have used one you have used them all.

## The Compute → Result pattern

Each analysis is a small configurable object built once and then *called* on
data. Calling it returns a typed `Result` object — never a bare tuple — so
outputs are self-describing. Some analyses (`RDF`, the densities, order
parameters) first need a `NeighborList` — a single spatial query you build the
same way and pass alongside the frames:

```python
from molpy.compute import NeighborList, RDF

nlist  = NeighborList(cutoff=10.0)(frame)   # 1. spatial query (pair neighbours)
rdf    = RDF(n_bins=100, r_max=10.0)        # 2. configure the analysis
result = rdf([frame], [nlist])              # 3. call on data -> typed RDFResult
result.rdf, result.bin_centers              # 4. read self-describing fields
```

Analyses that do not need pair distances (e.g. `MSD`, the distribution
functions) are called directly on the frames — same configure-then-call shape.

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
| Onsager coefficients | `Onsager` | `OnsagerResult` | $L_{ij}$ collective displacement cross-correlation |
| Current-ACF conductivity | `JACF` | `JACFResult` | $\sigma$ (S/m) via Green-Kubo $\langle J(0)\cdot J(t)\rangle$ |
| Pair persistence | `Persist` | `PersistResult` | residence-time / survival $C(\tau)$ |
| Radial distribution | `RDF` | structural $g(r)$ | pair structure |
| Static structure factor | `StaticStructureFactorDebye` | $S(k)$ | reciprocal-space structure |
| Mean-squared displacement | `MSD` | time series | single-particle diffusion |
| Velocity autocorrelation (VACF) | `compute_acf`, `PowerSpectrum` | ACF / VDOS | velocity memory, Green–Kubo diffusion, vibrational modes |
| Neighbor list | `NeighborList` | pair list | cutoff neighbor queries |
| Local / grid density | `LocalDensity`, `GaussianDensity` | density field | number-density fields |
| Order parameters | `Steinhardt`, `Hexatic`, `Nematic`, `SolidLiquid` | per-particle order | crystallinity / phase / alignment |
| Bond-orientational diagram | `BondOrder` | $(\theta,\phi)$ histogram | local bonding geometry |
| Potential of mean force & torque | `PMFTXY` | free-energy field | orientation-resolved PMF |
| Shape descriptors | `RadiusOfGyration`, `GyrationTensor`, `InertiaTensor`, `CenterOfMass` | per-frame tensors/scalars | molecular shape |
| Clustering / decomposition | `Cluster`, `ClusterCenters`, `Pca`, `KMeans` | labels / components | grouping & dimensionality reduction |
| Geometric distributions | `DistanceDistribution`, `AngleDistribution`, `DihedralDistribution` | $p(r)$, $p(\theta)$, $p(\phi)$ | bond-angle / torsion structure |
| Combined distribution | `CombinedDistribution` | N-D histogram | correlated observables (combined distribution function, CDF) |
| Spatial distribution | `SpatialDistribution` | body-fixed density | 3-D orientation-resolved structure (spatial distribution function, SDF) |
| Van Hove correlation | `VanHove` | $G(r,t)$ | time-resolved structure / dynamics |
| Reorientational TCFs | `LegendreReorientation` | $C_1(t)$, $C_2(t)$ | vector reorientation times |
| Hydrogen bonds | `HBonds`, `HBondCriterion` | per-frame bond lists | H-bond networks & counts |
| Radical Voronoi | `RadicalVoronoi`, `VoronoiIntegration`, `voronoi_domains`, `voronoi_voids` | cells / domains / moments | tessellation, domains, voids, charges |
| Vibrational spectra | `PowerSpectrum`, `IRSpectrum`, `RamanSpectrum`, `VcdSpectrum`, `RoaSpectrum` | spectra | vibrational density of states (VDOS), IR, Raman, vibrational circular dichroism (VCD), Raman optical activity (ROA) from ACFs |

Analyses can be chained into a directed graph with
[`Workflow`](workflows.md) for multi-step pipelines (e.g. dipole → ACF →
spectrum). See **[Compute Workflows](workflows.md)**.

## Featured guides

These are complete, textbook-style derivations that build each method from first
principles — read them to understand *why* the analyses work, not just how to
call them.

### Structure

- **[Structural Analysis](structure.md)** — the pair distribution function
  $g(r)$, coordination numbers, the static structure factor $S(k)$ (Debye
  equation), local and grid number densities, the shared neighbor-list primitive,
  and the potential of mean force. Covers `RDF`, `StaticStructureFactorDebye`,
  `LocalDensity`, `GaussianDensity`, `NeighborList`, and `PMFTXY`.
- **[Distribution Functions](distributions.md)** — angular (ADF), dihedral (DDF),
  distance, combined (CDF) and spatial (SDF) distribution functions. Covers
  `AngleDistribution`, `DihedralDistribution`, `DistanceDistribution`,
  `CombinedDistribution`, and `SpatialDistribution`.
- **[Bond-Orientational Order](order.md)** — Steinhardt $q_\ell$/$w_\ell$ from the
  spherical harmonics of the bonds, fcc/hcp/bcc discrimination, the hexatic
  $\psi_6$, solid–liquid classification, and the nematic $Q$-tensor. Covers
  `Steinhardt`, `Hexatic`, `SolidLiquid`, `Nematic`, and `BondOrder`.
- **[Shape, Clustering & Decomposition](descriptors.md)** — the gyration and
  inertia tensors, shape anisotropy, aggregate detection, and PCA / k-means over
  descriptor sets. Covers the shape descriptors, `Cluster`, `ClusterProperties`,
  `Pca`, and `KMeans`.
- **[Hydrogen-Bond Networks](hbonds.md)** — geometric H-bond detection and the
  link to lifetimes. Covers `HBonds` and `HBondCriterion`.
- **[Radical Voronoi](voronoi.md)** — radical tessellation, domain and void
  analysis, and electron-density charge integration. Covers `RadicalVoronoi`,
  `VoronoiIntegration`, `voronoi_domains`, and `voronoi_voids`.

### Dynamics

- **[Diffusion & Ionic Transport](transport.md)** — from the random walk and the
  Einstein relation to the mean-squared displacement, self vs distinct diffusion
  (the mean displacement correlation, MDC), the Onsager phenomenological
  coefficients, and the two equivalent
  conductivity routes (PMSD / current ACF). Covers `MCDCompute`, `Onsager`,
  `PMSDCompute`, `JACF`, and `IonicConductivity`.
- **[Velocity Autocorrelation & VDOS](vacf.md)** — the velocity memory function:
  gas/liquid/solid signatures and the cage effect, the Green–Kubo route to the
  diffusion coefficient, and the vibrational density of states by Fourier
  transform. Covers `compute_acf` and `PowerSpectrum`.
- **[Van Hove & Reorientational Dynamics](van-hove.md)** — the time-resolved
  $G(r,t)$ and the Legendre reorientational TCFs $C_1$/$C_2$. Covers `VanHove`
  and `LegendreReorientation`.
- **[Pair Persistence](persistence.md)** — residence-time correlation functions:
  the survival indicator, continuous vs intermittent vs stable-states
  persistence (SSP) definitions, coordination numbers, and the link to pairing
  diffusion. Covers
  `Persist`.

### Spectroscopy

- **[Dielectric Spectroscopy](dielectric.md)** — a complete derivation of
  $\varepsilon^*(\omega)$ and the ionic conductivity $\sigma$: the
  fluctuation–dissipation basis, the Einstein–Helfand and Green–Kubo routes,
  every numerical choice (windowing, FFT, unbiased ACF), the electrolyte dipole
  decomposition, and the spectral fitting recipes (Debye, Cole–Cole,
  Havriliak–Negami).
- **[Vibrational Spectra from MD](spectra.md)** — IR, Raman, VDOS, VCD and ROA via
  the time-correlation route. Covers `PowerSpectrum`, `IRSpectrum`,
  `RamanSpectrum`, `VcdSpectrum`, `RoaSpectrum`, and `ResonanceRamanSpectrum`.

## Related

- [API reference: Compute](../api/compute.md) — autodoc for the classes above.
- [Tutorials: Trajectory](../tutorials/05_trajectory.md) — the input data model.
- [Tutorials: Box and Periodicity](../tutorials/03_box_and_periodicity.md) — minimum-image conventions used by dynamical analyses.
