# Compute

The **Compute** layer turns a `Trajectory` or `Frame` (or pre-assembled arrays)
into physical observables: structural distributions, dynamical correlations, and
spectra.

Heavy numerics run in the high-performance backend. The public types below are
the stable import path. Transport and dielectric quantities follow an explicit
**compose** pattern: raw Compute → Fit → optional SI scale.

Navigation follows a **freud-style analysis map** (short names under categories).
Each page is written **textbook-first**: physical theory and equations, then
copy-paste usage against the live API, then figures where they clarify the
curve. Full signatures live in [API reference: Compute](../api/compute.md).

## Patterns

### Frame-oriented analyses

Structural and many dynamical analyses are small configurable objects: build once,
call on frames / neighbor lists, get a typed result.

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(200, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(20.0)
```

```python
from molpy.compute import NeighborList, RDF

nlist = NeighborList(cutoff=10.0)(frame)
rdf = RDF(n_bins=100, r_max=10.0)
result = rdf([frame], [nlist])
result.rdf, result.bin_centers
```

### Array-oriented transport / dielectric (compose)

Electrolyte transport and frequency-dependent dielectrics take **pre-assembled**
arrays (collective dipole $\mathbf{M}(t)$, current $\mathbf{J}(t)$, species COM
sum $\mathbf{P}_\alpha(t)$). The caller unwraps coordinates and builds those
observables; the analysis stack correlates / transforms.

```python
import numpy as np
from molpy.compute import (
    EinsteinConductivity,
    GreenKuboConductivity,
    LinearFit,
    CumulativeTrapezoid,
)

rng = np.random.default_rng(0)
M_trans = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.01, size=(80, 3)), axis=0))
J = np.ascontiguousarray(rng.normal(0, 1.0, size=(80, 3)))
dt = 10.0  # fs

raw = EinsteinConductivity().compute(M_trans, dt=dt, max_correlation_time=20)
fit = LinearFit(0.1, 0.5).fit(raw["lag_times"], raw["msd"])
# sigma_S_per_m = fit["slope"] / (6 * V * k_B * T) * SI_prefactor

jacf = GreenKuboConductivity().compute(J, dt=dt, max_correlation_time=20)
integ = CumulativeTrapezoid().fit(jacf["jacf"], dt=dt)["integral"]
# sigma_S_per_m = integ[-1] / (3 * V * k_B * T) * SI_prefactor
```

!!! note "Units"
    Analysis time is **fs** (LAMMPS *real*). Length Å, charge $e$, volume Å³,
    temperature K. Vibrational spectra use `dt_fs` and report cm⁻¹. GROMACS is
    nm-native — scale lengths ×10. Displacement-based kernels need **unwrapped**
    coordinates.

## Analysis map

### Structure

| Page | Entry points | Measures |
|------|----------------|----------|
| [NeighborList](neighborlist.md) | `NeighborList` | cutoff pairs (shared primitive) |
| [RDF](rdf.md) | `RDF` | $g(r)$ |
| [Density](density.md) | `LocalDensity`, `GaussianDensity` | local / grid density |
| [Diffraction](diffraction.md) | `StaticStructureFactorDebye` | $S(k)$ |
| [PMFT](pmft.md) | `PMFTXY` | orientation-resolved free energy |

### Distributions

| Page | Entry points | Measures |
|------|----------------|----------|
| [Distribution](distribution.md) | `DistanceDistribution`, `AngleDistribution`, `DihedralDistribution`, `CombinedDistribution` | $p(r)$, $p(\theta)$, CDF |
| [Spatial](spatial.md) | `SpatialDistribution` | body-fixed 3-D SDF |

### Order

| Page | Entry points | Measures |
|------|----------------|----------|
| [Order](order.md) | `Steinhardt`, `Hexatic`, `Nematic`, `SolidLiquid` | $q_\ell$, $\psi_6$, $S$, solid/liquid |
| [Environment](environment.md) | `BondOrder` | $(\theta,\phi)$ bond diagram |

### Shape & Clustering

| Page | Entry points | Measures |
|------|----------------|----------|
| [Shape](shape.md) | `RadiusOfGyration`, `GyrationTensor`, `InertiaTensor`, … | molecular shape |
| [Cluster](cluster.md) | `Cluster`, `ClusterProperties` | connected aggregates |
| [Decomposition](decomposition.md) | `Pca`, `KMeans` | PCA / states |

### Bonds & Voronoi

| Page | Entry points | Measures |
|------|----------------|----------|
| [HBond](hbond.md) | `HBonds`, `HBondCriterion` | geometric H-bonds |
| [Voronoi](voronoi.md) | `RadicalVoronoi`, `voronoi_domains`, `VoronoiIntegration` | tessellation / charges |

### Transport

| Page | Entry points | Measures |
|------|----------------|----------|
| [MSD](msd.md) | `MSD` | Einstein self-diffusion curve |
| [PMSD](pmsd.md) | `EinsteinConductivity` → `LinearFit` | ionic $\sigma$ (Einstein) |
| [JACF](jacf.md) | `GreenKuboConductivity` → `CumulativeTrapezoid` | ionic $\sigma$ (GK) |
| [Onsager](onsager.md) | `Onsager.correlation` | $\Omega_{\alpha\beta}$ |
| [Persist](persist.md) | `Persist.pair_survival_tcf` | residence-time $C(\tau)$ |
| [VACF](vacf.md) | `Acf` / `signal.acf_fft` | velocity memory, GK $D$, VDOS |

### Dynamics

| Page | Entry points | Measures |
|------|----------------|----------|
| [Van Hove](van_hove.md) | `VanHove` | $G(r,t)$ |
| [Reorientation](reorientation.md) | `LegendreReorientation` | $C_1$, $C_2$ |

### Spectroscopy

| Page | Entry points | Measures |
|------|----------------|----------|
| [Dielectric](dielectric.md) | `Dielectric`, EH/GK spectra, Fits | $\varepsilon^*(\omega)$, $\sigma$ |
| [Spectra](spectra.md) | `PowerSpectrum`, `IRSpectrum`, `RamanSpectrum`, … | VDOS / IR / Raman / VCD / ROA |
| [Signal](signal.md) | `signal.acf_fft`, `apply_window`, `Acf` | ACF / windows / grids |

### Workflow

| Page | Entry points | Measures |
|------|----------------|----------|
| [Workflow](workflow.md) | `Workflow` | DAG composition of computes |

## Related

- [API reference: Compute](../api/compute.md)
- [Tutorials: Trajectory](../tutorials/05_trajectory.md)
- [Tutorials: Box and Periodicity](../tutorials/03_box_and_periodicity.md)
