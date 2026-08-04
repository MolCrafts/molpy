# Compute

Trajectory and structure analyses. Import with `from molpy.compute import ...`.

Numerical kernels live in **molrs** (Rust). MolPy re-exports the same types
identity-style for a stable Python import path — there is no second science
implementation in molpy. Compose **raw Computes** with **Fits** (and an optional
SI scale) the same way the Rust API does.

See the [Compute overview](../compute/index.md) for the configure → call pattern,
and the domain guides under `docs/compute/` for derivations.

!!! note "Analysis units (LAMMPS *real*)"
    Length **Å**, charge **e**, **time fs**, volume Å³, temperature K.
    Vibrational spectra take `dt_fs` in femtoseconds and report cm⁻¹.
    GROMACS trajectories are nm-native — scale lengths ×10 before analysis.
    MSD / Einstein routes need **unwrapped** coordinates.

## Architecture: raw Compute → Fit → scale

| Layer | Role | Examples |
|-------|------|----------|
| Raw Compute | Correlation / MSD / ACF curve only | `EinsteinConductivity`, `GreenKuboConductivity`, `DebyeRelaxation`, `VACF`, `MSD` |
| Fit | Integrate or slope-fit the curve | `CumulativeTrapezoid`, `LinearFit`, `DebyeFit`, `EinsteinHelfandSpectrum`, `GreenKuboSpectrum` |
| Scale | MD → SI prefactor in your script | $1/(6 V k_B T)$, $1/(3 V k_B T)$, $1/d$ |

There is **no** all-in-one `IonicConductivity` / `DielectricSusceptibility` recipe
class. Historical tame names remain as **aliases** of the molrs types:

| Alias (deprecated name) | Canonical molrs type |
|-------------------------|----------------------|
| `JACF` | `GreenKuboConductivity` |
| `PMSDCompute` | `EinsteinConductivity` |

## Quick reference

| Symbol | Summary | Returns |
|--------|---------|---------|
| `Compute` | Base class for frame-oriented analyses | typed result |
| `Dielectric` | Dipole / current / Neumann $\varepsilon(0)$ (static methods) | arrays / scalar |
| `DebyeRelaxation` | Raw fluctuation-dipole ACF + $\langle M^2\rangle$ metadata | ACF + invariants |
| `EinsteinHelfandSpectrum` | Fit: dipole ACF → $\varepsilon^*(\omega)$ (EH) | spectrum |
| `GreenKuboSpectrum` | Fit: current ACF → $\varepsilon^*(\omega)$ (GK) | spectrum |
| `DebyeFit` | Time-domain Debye $\tau$ on normalized $\Phi(t)$ | $\tau$, $A$ |
| `EinsteinConductivity` (`PMSDCompute`) | Raw collective charge-dipole MSD | `lag_times`, MSD curve |
| `GreenKuboConductivity` (`JACF`) | Raw current ACF $\langle J(0)\cdot J(t)\rangle$ | `lag_times`, `jacf` |
| `LinearFit` | OLS slope over a fractional window | slope, intercept |
| `CumulativeTrapezoid` | Running $\int_0^\tau y\,dt$ | running integral |
| `Onsager` | `Onsager.correlation(P_i, P_j, dt, max_lag)` | $L_{ij}(\tau)$ |
| `Persist` | `Persist.pair_survival_tcf(...)` | residence $C(\tau)$ |
| `RDF` | Radial distribution $g(r)$ | structural result |
| `MSD` | Single-particle MSD (`direct` / `window`) | `MSDTimeSeries` |
| `StaticStructureFactorDebye` | $S(k)$ (Debye) | structural result |
| `NeighborList` | Cutoff neighbor pairs | pair list |
| `LocalDensity`, `GaussianDensity` | Number-density fields | density field |
| `Steinhardt`, `Hexatic`, `Nematic`, `SolidLiquid` | Bond-orientational order | per-particle order |
| `BondOrder` | $(\theta,\phi)$ bond diagram | spherical histogram |
| `PMFTXY` | Potential of mean force & torque | free-energy field |
| Shape / cluster / PCA | `RadiusOfGyration`, `Cluster`, `Pca`, `KMeans`, … | tensors / labels |
| Geometric distributions | `DistanceDistribution`, `AngleDistribution`, … | histograms |
| `SpatialDistribution` | SDF (body-fixed density) | 3-D grid |
| `VanHove` | $G(r,t)$ | time-resolved $g$ |
| `LegendreReorientation` | $C_1(t)$, $C_2(t)$ | TCFs |
| `HBonds`, `HBondCriterion` | Geometric H-bonds | bond lists |
| Radical Voronoi | `RadicalVoronoi`, `voronoi_domains`, … | cells / domains |
| Vibrational spectra | `PowerSpectrum`, `IRSpectrum`, `RamanSpectrum`, … | spectrum (cm⁻¹) |
| `signal` | `acf_fft`, `apply_window`, `frequency_grid` | arrays |
| `Workflow` | Directed graph of chained computes | per-node results |

Diffusion Green–Kubo / raw VACF types (`VACF`, `GreenKuboDiffusion`,
`EinsteinDiffusion`) live on `molrs.compute.transport` and are the SSOT for those
kernels; re-export them only if you need a molpy-local name.

---

## Full API

### Base

::: molpy.compute.base

### Result types

::: molpy.compute.result

### Dielectric & spectroscopy fits

::: molpy.compute.dielectric

### Einstein conductivity (polarization MSD)

::: molpy.compute.pmsd

### Onsager coefficients

::: molpy.compute.onsager

### Green–Kubo conductivity (current ACF)

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

### Signal processing

::: molpy.compute.signal

### Workflow

::: molpy.compute.workflow
