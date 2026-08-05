# Compute

Trajectory and structure analyses. Import with `from molpy.compute import ...`.

Numerical kernels live in **molrs** (Rust). MolPy re-exports the same types
identity-style for a stable Python import path — there is no second science
implementation in molpy. Compose **raw Computes** with **Fits** (and an optional
SI scale) the same way the Rust API does.

Like freud’s [API modules](https://freud.readthedocs.io/en/stable/), each
`molpy.compute` module has its own page under [Compute](../compute/index.md)
with an overview table and full signatures. This page is the **index** plus the
shared base / result types.

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
| Scale | MD → SI prefactor in your script | $1/(6 V k_B T)$, $1/(3 V k_B T)$, $1/d$ |

There is **no** all-in-one `IonicConductivity` / `DielectricSusceptibility` recipe
class. Historical tame names remain as **aliases** of the molrs types:

| Alias (deprecated name) | Canonical molrs type |
|-------------------------|----------------------|
| `JACF` | `GreenKuboConductivity` |
| `PMSDCompute` | `EinsteinConductivity` |

`VACF` / `GreenKuboDiffusion` / `EinsteinDiffusion` are **not** re-exported;
import them from `molrs.compute.transport`.

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
| `dielectric` | dielectric raw/fit helpers | [Dielectric](../compute/dielectric.md) |
| `spectra` | VDOS / IR / Raman / VCD / ROA | [Spectra](../compute/spectra.md) |
| `signal` | `acf_fft`, windows, frequency grid | [Signal](../compute/signal.md) |
| `workflow` | `Workflow` | [Workflow](../compute/workflow.md) |

---

## Shared base types

### Base

::: molpy.compute.base

### Result types

::: molpy.compute.result
