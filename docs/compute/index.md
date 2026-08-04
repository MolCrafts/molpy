# Compute

The **Compute** layer turns a `Trajectory` or `Frame` (or pre-assembled arrays)
into physical observables: structural distributions, dynamical correlations, and
spectra.

Heavy numerics run in **molrs** (Rust). MolPy re-exports those types identity-style
— science is not reimplemented in Python. Transport and dielectric quantities
follow an explicit **compose** pattern: raw Compute → Fit → optional SI scale.

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
observables; molrs only correlates / transforms.

```python
import numpy as np
from molpy.compute import (
    EinsteinConductivity,
    GreenKuboConductivity,
    LinearFit,
    CumulativeTrapezoid,
)

rng = np.random.default_rng(0)
# Pre-assembled series (caller builds M / J from unwrapped q·r and q·v)
M_trans = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.01, size=(80, 3)), axis=0))
J = np.ascontiguousarray(rng.normal(0, 1.0, size=(80, 3)))
dt = 10.0  # fs

# Einstein σ: raw charge-dipole MSD → slope → SI scale
raw = EinsteinConductivity().compute(M_trans, dt=dt, max_correlation_time=20)
fit = LinearFit(0.1, 0.5).fit(raw["lag_times"], raw["msd"])
# sigma_S_per_m = fit["slope"] / (6 * V * k_B * T) * SI_prefactor

# Green–Kubo σ: raw current ACF → ∫ → SI scale
jacf = GreenKuboConductivity().compute(J, dt=dt, max_correlation_time=20)
integ = CumulativeTrapezoid().fit(jacf["jacf"], dt=dt)["integral"]
# sigma_S_per_m = integ[-1] / (3 * V * k_B * T) * SI_prefactor
```


!!! note "Units"
    Analysis time is **fs** (LAMMPS *real*). Length Å, charge $e$, volume Å³,
    temperature K. Vibrational spectra use `dt_fs` and report cm⁻¹. GROMACS is
    nm-native — scale lengths ×10. Displacement-based kernels need **unwrapped**
    coordinates.

## Available analyses

| Method | Class / entry point | Measures |
|--------|---------------------|----------|
| Static dielectric $\varepsilon(0)$ | `Dielectric.static_dielectric_constant` | Neumann fluctuation formula |
| Dielectric spectrum $\varepsilon^*(\omega)$ | `DebyeRelaxation` → `EinsteinHelfandSpectrum` / `GreenKuboConductivity` → `GreenKuboSpectrum` | frequency-dependent permittivity |
| Ionic conductivity $\sigma$ (Einstein) | `EinsteinConductivity` → `LinearFit` → SI scale | DC $\sigma$ from charge-dipole MSD |
| Ionic conductivity $\sigma$ (Green–Kubo) | `GreenKuboConductivity` → `CumulativeTrapezoid` → SI scale | DC $\sigma$ from $\langle J\cdot J\rangle$ |
| Onsager $L_{ij}$ | `Onsager.correlation` | collective displacement cross-correlation |
| Pair persistence | `Persist.pair_survival_tcf` | residence-time $C(\tau)$ |
| Radial distribution | `RDF` | pair structure $g(r)$ |
| Static structure factor | `StaticStructureFactorDebye` | $S(k)$ |
| Mean-squared displacement | `MSD` | single-particle diffusion curve |
| Velocity ACF / GK diffusion | `molrs.compute.transport.VACF` / `GreenKuboDiffusion` | velocity memory, Green–Kubo $D$ |
| Neighbor list | `NeighborList` | cutoff pairs |
| Local / grid density | `LocalDensity`, `GaussianDensity` | number density |
| Order parameters | `Steinhardt`, `Hexatic`, `Nematic`, `SolidLiquid` | crystallinity / alignment |
| Bond-orientational diagram | `BondOrder` | $(\theta,\phi)$ histogram |
| PMFT | `PMFTXY` | orientation-resolved free energy |
| Shape descriptors | `RadiusOfGyration`, `GyrationTensor`, … | molecular shape |
| Clustering / decomposition | `Cluster`, `Pca`, `KMeans` | grouping & PCA |
| Geometric distributions | `DistanceDistribution`, `AngleDistribution`, … | $p(r)$, $p(\theta)$, … |
| Combined / spatial DF | `CombinedDistribution`, `SpatialDistribution` | CDF / SDF |
| Van Hove | `VanHove` | $G(r,t)$ |
| Reorientational TCFs | `LegendreReorientation` | $C_1$, $C_2$ |
| Hydrogen bonds | `HBonds`, `HBondCriterion` | H-bond lists |
| Radical Voronoi | `RadicalVoronoi`, `voronoi_domains`, … | tessellation / voids |
| Vibrational spectra | `PowerSpectrum`, `IRSpectrum`, `RamanSpectrum`, … | VDOS / IR / Raman / VCD / ROA |
| Signal helpers | `signal.acf_fft`, `apply_window`, `frequency_grid` | array ACF / windows |

Chain multi-step pipelines with [`Workflow`](workflows.md).

## Featured guides

### Structure

- **[Structural Analysis](structure.md)** — $g(r)$, $S(k)$, densities, neighbor list, PMFT.
- **[Distribution Functions](distributions.md)** — ADF, DDF, CDF, SDF.
- **[Bond-Orientational Order](order.md)** — Steinhardt, hexatic, solid–liquid, nematic.
- **[Shape, Clustering & Decomposition](descriptors.md)** — tensors, cluster, PCA.
- **[Hydrogen-Bond Networks](hbonds.md)** — geometric H-bonds.
- **[Radical Voronoi](voronoi.md)** — tessellation, domains, voids.

### Dynamics

- **[Diffusion & Ionic Transport](transport.md)** — MSD, Onsager, Einstein & Green–Kubo $\sigma$ via compose.
- **[Velocity Autocorrelation & VDOS](vacf.md)** — VACF, Green–Kubo $D$, VDOS.
- **[Van Hove & Reorientational Dynamics](van-hove.md)** — $G(r,t)$, $C_1$/$C_2$.
- **[Pair Persistence](persistence.md)** — continuous / intermittent / SSP survival.

### Spectroscopy

- **[Dielectric Spectroscopy](dielectric.md)** — $\varepsilon^*(\omega)$, FDT, EH/GK compose, electrolyte decomposition.
- **[Vibrational Spectra from MD](spectra.md)** — IR, Raman, VDOS, VCD, ROA from ACFs.

## Related

- [API reference: Compute](../api/compute.md)
- [Tutorials: Trajectory](../tutorials/05_trajectory.md)
- [Tutorials: Box and Periodicity](../tutorials/03_box_and_periodicity.md)
