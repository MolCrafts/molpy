# Compute

Trajectory and structure analyses. Import with `from molpy.compute import...`.

Numerical kernels live in the high-performance native backend, and there is no
second science implementation in molpy. Most public types are the backend
classes re-exported under a stable `molpy.compute` import path (for example
`EinsteinConductivity`, `LinearFit`, `Dielectric`, the spectrum classes). A few
are thin molpy wrappers that add input handling and nothing numerical: `RDF`
accepts one frame or a list, refuses a frame whose `box` is free (no periodic
cell) with `ValueError`, and returns the backend's `RDFResult` unchanged; `MSD`
forwards to the backend estimator and returns its `MSDTimeSeries`.

Compose **raw Computes** with **Fits** (and an optional SI scale) yourself. The
all-in-one recipe classes (`IonicConductivity`, `DielectricSusceptibility`, and
their `ConductivityResult` / `DielectricSusceptibilityResult`) that used to wrap
that pipeline in one call were removed in 0.15 — they buried the fit window and
the unit conversion in a library default. Compose `EinsteinConductivity` →
`LinearFit` → your own prefactor, as the [PMSD](../compute/pmsd.md) and
[Dielectric](../compute/dielectric.md) pages show.

!!! warning "Time unit when migrating from `IonicConductivity`"
    `IonicConductivity` took its frame spacing `dt` in **picoseconds**. The
    composed route works in **femtoseconds**: pass `dt` in fs to
    `EinsteinConductivity().compute(...)`, and the S/m prefactor on the
    [PMSD](../compute/pmsd.md) page ($3.0988\times10^{9}$) assumes a slope in
    $e^2\,\text{Å}^2\,\text{fs}^{-1}$. Reusing an old picosecond `dt` makes the
    lag axis 1000× too short and the conductivity 1000× too large. See
    [Migrating to 0.15](../getting-started/migration-0-15.md).

Like freud’s [API modules](https://freud.readthedocs.io/en/stable/), each
`molpy.compute` module has its own page under [Compute](../compute/index.md)
with an overview table and full signatures. This page is the **index** plus the
shared contract / result types.

!!! note "Analysis units (LAMMPS *real*)"
    Length **Å**, charge **e**, **time fs**, volume Å³, temperature K.
    Vibrational spectra take `dt_fs` in femtoseconds and report cm⁻¹.
    GROMACS trajectories are nm-native — scale lengths ×10 before analysis.
    MSD / Einstein routes need **unwrapped** coordinates.

## Architecture: raw Compute → Fit → scale

| Layer | Role | Examples |
|-------|------|----------|
| Raw Compute | Correlation / MSD / ACF curve only | `EinsteinConductivity`, `GreenKuboConductivity`, `DebyeRelaxation`, `MSD` |
| Fit | Integrate or slope-fit the curve | `CumulativeTrapezoid`, `LinearFit`, `DebyeFit`, `EinsteinHelfandSpectrum`, `GreenKuboSpectrum` |
| Scale | MD → SI prefactor in your script | $1/(6 V k_B T)$ (Einstein conductivity), $1/(3 V k_B T)$ (Green–Kubo conductivity), $1/(2d)$ (self-diffusion from an MSD slope in $d$ dimensions, so $1/6$ in 3D) |

Self-diffusion uses `MSD` (Einstein) and `Acf` / `signal.acf_fft` (Green–Kubo);
see the [MSD](../compute/msd.md) and [VACF](../compute/vacf.md) guides.

## Module index

| Module | Primary exports | Guide |
|--------|-----------------|-------|
| `neighborlist` | `NeighborList` | [NeighborList](../compute/neighborlist.md) |
| `rdf` | `RDF` | [RDF](../compute/rdf.md) |
| `density` | `LocalDensity`, `GaussianDensity` | [Density](../compute/density.md) |
| `diffraction` | `StaticStructureFactorDebye` | [Diffraction](../compute/diffraction.md) |
| `pmft` | `PMFTXY` | [PMFT](../compute/pmft.md) |
| `distribution` | distance / angle / dihedral / combined DF | [Distribution](../compute/distribution.md) |
| `spatial` | `SpatialDistribution` | [Spatial](../compute/spatial.md) |
| `order` | Steinhardt family | [Order](../compute/order.md) |
| `environment` | `BondOrder` | [Environment](../compute/environment.md) |
| `shape` | COM, gyration, inertia, $R_g$ | [Shape](../compute/shape.md) |
| `cluster` | `Cluster`, `ClusterCenters`, `ClusterProperties` | [Cluster](../compute/cluster.md) |
| `decomposition` | `DescriptorRow`, `Pca`, `KMeans` | [Decomposition](../compute/decomposition.md) |
| `hbond` | `HBonds`, `HBondCriterion` | [HBond](../compute/hbond.md) |
| `voronoi` | radical Voronoi tessellation | [Voronoi](../compute/voronoi.md) |
| `msd` | `MSD` | [MSD](../compute/msd.md) |
| `pmsd` | `EinsteinConductivity` | [PMSD](../compute/pmsd.md) |
| `jacf` | `GreenKuboConductivity` | [JACF](../compute/jacf.md) |
| `onsager` | `Onsager` | [Onsager](../compute/onsager.md) |
| `persist` | `Persist` | [Persist](../compute/persist.md) |
| `van_hove` | `VanHove` | [Van Hove](../compute/van_hove.md) |
| `reorientation` | `LegendreReorientation` | [Reorientation](../compute/reorientation.md) |
| `signal` | `acf_fft`, windows, frequency grid | [Signal](../compute/signal.md) |

Two families have no module of their own; import them from `molpy.compute`
directly:

| Family | Exports on `molpy.compute` | Guide |
|--------|----------------------------|-------|
| Dielectric response | `Dielectric`, `DebyeRelaxation`, `DebyeFit`, `EinsteinHelfandSpectrum`, `GreenKuboSpectrum`, `DielectricResult` | [Dielectric](../compute/dielectric.md) |
| Vibrational spectra | `PowerSpectrum` (VDOS), `IRSpectrum`, `RamanSpectrum`, `ResonanceRamanSpectrum`, `VcdSpectrum`, `RoaSpectrum` | [Spectra](../compute/spectra.md) |

---

## Shared types

### The `Compute` contract

`Compute` is not a base class to inherit from. It is a structural
`typing.Protocol` owned by the molrs backend and re-exported here
(`molpy.compute.Compute is molrs.compute.Compute`): any class that defines a
`compute(...)` method satisfies it, with no subclassing and no registration.
Writing one is covered in
[Adding a Compute Operation](../developer/extending-compute.md).

::: molpy.compute.Compute

### Result types

::: molpy.compute.result
