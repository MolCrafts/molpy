# Compute

A finished simulation is a pile of coordinates. On its own it proves nothing.
The `molpy.compute` layer is what turns that pile into quantities you can plot,
publish, and compare against experiment: distribution functions, diffusion
coefficients, order parameters, spectra.

These pages are written to be read, not just searched. Each one starts from the
physical question, builds the quantity from something you could in principle
measure by hand, shows it on real data, and then gives the code. Full
signatures live in the [API reference](../api/compute.md).

If you are new here, read [NeighborList](neighborlist.md) and [RDF](rdf.md)
first, in that order. Most of the structural analyses are variations on those
two, and the conventions introduced there apply everywhere.

## How a compute is used

Every analysis follows the same three beats: **configure an object, call it on
data, read the result.**

```python
import numpy as np
import molpy as mp
from molpy.compute import NeighborList, RDF

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(200, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(20.0)

nlist = NeighborList(cutoff=10.0)(frame)          # configure, then call
result = RDF(n_bins=100, r_max=10.0)([frame], [nlist])
print(result.rdf.shape, result.bin_centers.shape)  # -> (100,) (100,)
```

The parameters that define the measurement (`cutoff`, `n_bins`, `r_max`) go into
the constructor; the data goes into the call. That separation is deliberate — it
means one configured analyzer can be applied to many trajectories and the
measurement stays identical.

### Two things to know about what comes back

Results are not uniform across the layer, and expecting the wrong one is the
most common early mistake.

Some computes return a **result object with named fields**, like `RDF` above
(`.rdf`, `.bin_centers`, `.n_frames`). Others return a **plain Python list with
one entry per frame**, where each entry is a tuple of arrays:

```python
from molpy.compute import LocalDensity

per_frame = LocalDensity(r_max=10.0)([frame], [nlist])
print(len(per_frame), len(per_frame[0]))           # -> 1 2
```

One frame in, one entry out, and that entry is a 2-tuple of
`(num_neighbors, density)`. It is an ordinary list, so you can index it, iterate
it more than once, and pass it around. When you have exactly one frame it reads
better to unpack immediately:

```python
(counts, density), = LocalDensity(r_max=10.0)([frame], [nlist])
print(density.shape)                               # -> (200,)
```

That trailing comma is doing real work. `(a, b), = xs` means "`xs` has exactly
one element, and that element is a pair; name its parts `a` and `b`". Drop the
comma and you get a `ValueError` about unpacking. If it looks too clever, the
long form `counts, density = per_frame[0]` is identical.

The per-page "Computing it" section always states the exact shape returned, and
the [API reference](../api/compute.md) lists them all.

### Frame-oriented and array-oriented analyses

Structural analyses take frames, because "where are the atoms" is the question.
Transport and dielectric analyses instead take **pre-assembled time series** —
the collective dipole $\mathbf{M}(t)$, the current $\mathbf{J}(t)$, per-species
centre-of-mass sums $\mathbf{P}_\alpha(t)$ — because only you know which
molecules and which charges belong in the sum.

Those follow an explicit **compose** pattern: a raw correlation function comes
out of the compute, and turning it into a transport coefficient is a separate,
visible step.

```python
from molpy.compute import EinsteinConductivity, LinearFit

rng = np.random.default_rng(0)
M = np.ascontiguousarray(np.cumsum(rng.normal(0, 0.01, size=(80, 3)), axis=0))

raw = EinsteinConductivity().compute(M, dt=10.0, max_correlation_time=20)
print(sorted(raw))                                 # -> ['lag_times', 'msd']
```

Three details in those two lines regularly trip people up.

**They are called with `.compute(...)`, not `(...)`.** Frame-oriented analyses
are callable objects; the array-oriented ones expose a named `compute` method.
The per-page examples show which is which.

**A conductivity returns something called `msd`.** That is not a mistake. The
Einstein route to conductivity is the mean squared displacement of the
collective charge dipole $\mathbf{M}(t) = \sum_a q_a \mathbf{r}_a$ — the same
mathematics as a [particle MSD](msd.md), applied to a different vector. See
[PMSD](pmsd.md).

**The fit window is a fraction, not a time.** `LinearFit(0.1, 0.5)` fits the
straight line over the portion of the curve from 10 % to 50 % of its length,
which is how you exclude the short-time transient and the noisy tail:

```python
fit = LinearFit(0.1, 0.5).fit(raw["lag_times"], raw["msd"])
print(sorted(fit))     # -> ['fit_end', 'fit_start', 'intercept', 'r2', 'slope']
```

`r2` is there so you can check the window really was linear before quoting the
slope.

Nothing hides that window or the unit conversion behind a convenience wrapper.
If a published number depends on where you chose to fit, that choice belongs in
your script where a reviewer can see it — not buried in a library default.

## Conventions that apply to every page

!!! note "Units and coordinates"
    - **Time is femtoseconds** (LAMMPS *real* units). Vibrational spectra take
      `dt_fs` and report cm⁻¹.
    - **Length is Å**, charge is $e$, volume Å³, temperature K. GROMACS is
      nm-native — multiply lengths by 10 on the way in.
    - **Displacement kernels need unwrapped coordinates.** Dynamics dumps are
      normally written that way already (`xu yu zu`, no-jump, …). Pass those
      frames straight in. Wrapped `x y z` without images turns a diffusion
      curve into box-sized jumps. See [MSD](msd.md).
    - **Structural kernels need a periodic box.** `frame.box` must be set. A
      *free* box — one with no periodicity, the state of a `Frame` you built
      without setting `box` — makes these raise rather than silently guess.
    - **Everything is an ensemble average.** Statistical mechanics predicts
      averages over all the microstates a system visits, not the value in any
      one of them. In practice you approximate that by averaging over frames
      of a trajectory (and, for most of these quantities, over every atom and
      every time origin too). A single frame is one sample, not a measurement.

## Starting from your own trajectory

Structural demos on these pages often build a single `Frame` with NumPy so the
snippet is self-contained. **Trajectory / dynamics work starts from a file.**
For a LAMMPS dump of unwrapped coordinates:

```python
# docs: skip — needs a trajectory file of your own
from molpy.io import read_lammps_trajectory
from molpy.compute import NeighborList, RDF

reader = read_lammps_trajectory("run.lammpstrj")
frames = reader.read_all()   # list[Frame]; frame.box from BOX BOUNDS

nlists = [NeighborList(cutoff=8.0)(f) for f in frames]
gr = RDF(n_bins=160, r_max=8.0)(frames, nlists)
```

Prefer dumps that already store continuous paths (`xu yu zu`, or `x y z` plus
`ix iy iz` unwrapped on read). There is no molpy helper that “turns a NumPy
array into a trajectory” for you — `MSD`, `VanHove`, and friends take a sequence
of `Frame`s.

Use `reader.read_frame(i)`, `read_range`, or `read_frames` instead of
`read_all()` when the trajectory does not fit in memory. Other formats live in
[`molpy.io`](../user-guide/11_io.md) (`read_gro`, `read_dcd_trajectory`, …).

### Partial (species-resolved) distributions

There is no species argument on `RDF` or the other structural computes. To get
an O–O rather than an all-pairs $g(r)$, build a frame containing only the atoms
you want and analyse that:

```python
rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(300, 3))
kinds = rng.integers(1, 3, 300)
mixture = mp.Frame()
mixture["atoms"] = {
    "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "type_id": kinds
}
mixture.box = mp.Box.cubic(20.0)

selected = mixture["atoms"]["type_id"] == 1
subset = mp.Frame()
subset["atoms"] = {k: mixture["atoms"][k][selected] for k in ("x", "y", "z")}
subset.box = mixture.box
print(subset["atoms"].nrows)                       # -> 143
```

The density used for normalization then comes from the subset, which is what you
want for a like–like partial. A cross-species $g_{AB}(r)$ needs a neighbour
query between two *different* point sets and is not covered by the self-query
wrapper here.

## Which analysis answers which question

The pages are grouped the way the questions group, not the way the classes do.

### Structure — where the atoms are

| Question | Page | Entry points |
|---|---|---|
| Which atoms are near which? | [NeighborList](neighborlist.md) | `NeighborList` |
| How are neighbours spaced? | [RDF](rdf.md) | `RDF` |
| Where is the matter, in space? | [Density](density.md) | `LocalDensity`, `GaussianDensity` |
| What would a diffraction experiment see? | [Diffraction](diffraction.md) | `StaticStructureFactorDebye` |
| Where do neighbours sit around an anisotropic particle? | [PMFT](pmft.md) | `PMFTXY` |
| How are bond lengths, angles, torsions distributed? | [Distribution](distribution.md) | `DistanceDistribution`, `AngleDistribution`, `DihedralDistribution` |
| Where do neighbours sit in a molecule's own frame? | [Spatial](spatial.md) | `SpatialDistribution` |

### Order, shape, and aggregates — what the structure adds up to

| Question | Page | Entry points |
|---|---|---|
| Crystalline, hexatic, or nematic order? | [Order](order.md) | `Steinhardt`, `Hexatic`, `Nematic`, `SolidLiquid` |
| What does the local environment look like? | [Environment](environment.md) | `BondOrder` |
| How big and how anisotropic is a molecule? | [Shape](shape.md) | `RadiusOfGyration`, `GyrationTensor`, `InertiaTensor` |
| What is connected to what? | [Cluster](cluster.md) | `Cluster`, `ClusterProperties` |
| What are the dominant collective coordinates? | [Decomposition](decomposition.md) | `Pca`, `KMeans` |
| Which atoms are hydrogen-bonded? | [HBond](hbond.md) | `HBonds`, `HBondCriterion` |
| How much space does each atom own? | [Voronoi](voronoi.md) | `RadicalVoronoi`, `VoronoiIntegration` |

### Motion — how things move and how fast

| Question | Page | Entry points |
|---|---|---|
| How far do atoms wander? (self-diffusion) | [MSD](msd.md) | `MSD` |
| What does the velocity remember? | [VACF](vacf.md) | `Acf`, `signal.acf_fft` |
| Ionic conductivity, from displacements? | [PMSD](pmsd.md) | `EinsteinConductivity` → `LinearFit` |
| …and from currents? | [JACF](jacf.md) | `GreenKuboConductivity` → `CumulativeTrapezoid` |
| Do ions move together or independently? | [Onsager](onsager.md) | `Onsager.correlation` |
| How long does a contact survive? | [Persist](persist.md) | `Persist.pair_survival_tcf` |
| How does a density fluctuation decay? | [Van Hove](van_hove.md) | `VanHove` |
| How fast do molecules tumble? | [Reorientation](reorientation.md) | `LegendreReorientation` |

### Spectra and composition

| Question | Page | Entry points |
|---|---|---|
| What is $\varepsilon^*(\omega)$? | [Dielectric](dielectric.md) | `Dielectric`, EH/GK spectra, fits |
| Vibrational, IR, or Raman spectrum? | [Spectra](spectra.md) | `PowerSpectrum`, `IRSpectrum`, `RamanSpectrum`, … |
| How do I window, transform, and correlate a signal? | [Signal](signal.md) | `signal.acf_fft`, `apply_window` |
| How do I chain several analyses together? | [Workflow](workflow.md) | `Workflow` |

## Where the figures come from

Every curve on these pages is computed, not drawn. The reference system is 500
argon atoms at 85 K and 1.374 g cm⁻³ — the Rahman state point — integrated for
30 ps at constant energy, conserving total energy to a relative drift of
$1.4\times10^{-5}$ (dimensionless, $|E(t)-E(0)|/|E(0)|$). The
generator lives in `scripts/docs_data/` and writes to `docs/data/`, so any
figure can be reproduced or challenged:

```python
# docs: skip — runs a 30 ps MD trajectory (minutes, not seconds)
from docs_data.run import argon_trajectory
from docs_data.structure import radial_distribution

radial_distribution(argon_trajectory())
```

Where no honest dataset exists yet, the page says so and carries a `TODO`
instead of a decorative sketch.

## Related

- [API reference: Compute](../api/compute.md)
- [Tutorials: Trajectory](../tutorials/05_trajectory.md)
- [Tutorials: Box and Periodicity](../tutorials/03_box_and_periodicity.md)
