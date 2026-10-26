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

## A compute is configured once, then fed data by keyword

Every analysis follows the same three beats: **configure an object, call
`.compute(...)` on data, read the result.**

```python
import numpy as np
import molpy as mp
from molpy.compute import LocalDensity

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(200, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box(mp.core.Cell.lengths([20.0] * 3))

nl = mp.core.NeighborList(10.0)
nl.build(frame.coords, frame.box)
nlist = nl.neighbors()
local = LocalDensity(r_max=10.0).compute(pairs=nlist)
print(local.density.shape)                         # -> (200,)
```

The parameters that define the measurement (here `r_max`) go into the
constructor, and nothing else does. The data goes into `.compute(...)`, always
by keyword: here the pair table of one frame. That separation is deliberate —
one configured analyzer can be applied to many trajectories and the
measurement stays identical. Every optional setting has its default in the
native core; `LocalDensity.spec()` lists them, and an instance reports its own
through `.tunables()`.

### Three computes are streams

`Msd`, `Vacf` and `Rdf` average over a whole trajectory, so they have no
`.compute(...)`. Their second beat is a stream: `.accumulator()` hands out an
accumulator, you `push` one frame at a time, and `finish()` reads the result
over every frame pushed so far — without consuming the accumulator, so a run
can be inspected and continued. Memory stays bounded however long the
trajectory is:

```python
from molpy.compute import Rdf

accumulator = Rdf(n_bins=100, r_max=10.0, r_min=0.0).accumulator()
accumulator.push(nlist)                            # one pair table per frame
result = accumulator.finish()
print(result.rdf.shape, result.bin_centers.shape)  # -> (100,) (100,)
```

### What comes back is a typed result

Every compute returns one result object with named fields, never a bare list
or tuple; only a compute whose answer is itself one array — `Derivative`,
`Unwrap`, `FiniteFieldPolarizability` — returns that NumPy array. `Rdf` gives `.rdf`, `.bin_centers`, `.n_frames`; `LocalDensity`
gives one `.density` and one `.n_neighbors` entry per particle:

```python
print(result.n_frames, local.n_neighbors.shape)    # -> 1 (200,)
```

The per-page "Computing it" section always names the fields a result carries.
A result is read-only, and the result one compute produces is exactly what
the next one consumes: a `ClusterResult` is the partition `CenterOfMass`
groups by, a `SteinhardtResult` the feature matrix `Pca` reads, a
`TimeCorrelation` the curve `LinearFit` fits.

### You read the columns; the compute reads arrays

A compute never touches a `Frame`. Which columns feed an analysis — unwrapped
or wrapped coordinates, which charges, which molecules — is a decision only
you can make, so you make it with the Frame/Block API (`frame.coords`,
`frame["atoms"]["q"]`, `frame.box`) and hand the arrays over. A trajectory is
one `np.stack` of per-frame coordinates into a `(T, N, 3)` array — or, for
the three streams, one `push` per frame.

Transport and dielectric analyses take those time series and follow an
explicit **compose** pattern: a series is built, a raw correlation comes out of
a correlation compute, and turning it into a transport coefficient is a
separate, visible step. The Einstein route to ionic conductivity shows all
three. The collective dipole $\mathbf{M}(t) = \sum_a q_a \mathbf{r}_a$ comes
from `DipoleMoment`, its mean squared displacement from `Msd`, and the slope
from `LinearFit`:

```python
from molpy.compute import DipoleMoment, LinearFit, Msd

positions = np.cumsum(rng.normal(0.0, 0.1, size=(80, 4, 3)), axis=0)  # (T, N, 3)
charges = np.array([1.0, -1.0, 1.0, -1.0])

dipole = DipoleMoment().compute(positions=positions, charges=charges)
print(dipole.moments.shape)                        # -> (80, 1, 3)

accumulator = Msd(max_lag=20).accumulator()
for moment in dipole.moments:                      # one (1, 3) "particle" per frame
    accumulator.push(moment, dt=10.0)
raw = accumulator.finish()
print(raw.values.shape)                            # -> (21, 1)
```

Two details in those lines regularly trip people up.

**A conductivity is an MSD.** The Einstein route to conductivity is the mean
squared displacement of the collective charge dipole — the same mathematics as
a [particle MSD](msd.md), applied to one "particle" whose position is
$\mathbf{M}(t)$. That is why `DipoleMoment` returns a `(T, 1, 3)` series: each
frame of it is a `(1, 3)` "position" that `Msd` takes as it is. See [PMSD](pmsd.md). Every time correlation runs on one
engine, `Correlator`, and returns the same `TimeCorrelation`: `.values[lag,
channel]` is the curve, `.lag_times` the time axis, and nothing is derived.
Integrals, slopes, fits and spectra are computes of their own that read it.

**The fit window is a fraction, not a time.** `LinearFit(0.1, 0.5)` fits the
straight line over the portion of the curve from 10 % to 50 % of its length,
which is how you exclude the short-time transient and the noisy tail:

```python
fit = LinearFit(0.1, 0.5).compute(correlation=raw, channel=0)
print(fit.fit_start, fit.fit_end)                  # -> 2 10
print(fit.r2 > 0.9)                                # -> True
```

`fit_start` and `fit_end` are the lags the window resolved to, and `r2` is
there so you can check the window really was linear before quoting the slope.

Nothing hides that window or the unit conversion behind a convenience wrapper.
If a published number depends on where you chose to fit, that choice belongs in
your script where a reviewer can see it — not buried in a library default.

## Conventions that apply to every page

!!! note "Units and coordinates"
    - **Time is femtoseconds** (LAMMPS *real* units): every `dt` is in fs.
      Vibrational spectra report cm⁻¹.
    - **Length is Å**, charge is $e$, volume Å³, temperature K. GROMACS
      files are nm-native; the readers convert them to Å, so a `Frame` is
      always in Å.
    - **Displacement kernels need unwrapped coordinates.** Dynamics dumps are
      normally written that way already (`xu yu zu`, no-jump, …). Wrapped
      `x y z` without images turns a diffusion curve into box-sized jumps;
      push each frame with its box (`simbox=`) and `Msd` refuses such a
      trajectory. See
      [MSD](msd.md).
    - **Structural kernels need a periodic box.** The neighbour search and
      every kernel that needs a volume read the box the pair table was
      searched in; a table with no box makes them raise rather than
      silently guess.
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
from pathlib import Path
import molpy as mp
from molpy.compute import Msd, Rdf

reader = mp.io.lammps.LammpsDumpReader(Path("run.lammpstrj"))

nl = mp.core.NeighborList(8.0)
gr = Rdf(n_bins=160, r_max=8.0, r_min=0.0).accumulator()
msd = Msd(max_lag=500).accumulator()
for f in reader:                                   # frame.box from BOX BOUNDS
    unwrapped = f["atoms"]["xu", "yu", "zu"]       # (N, 3)
    nl.build(unwrapped, f.box)                     # the search wraps them itself
    gr.push(nl)
    msd.push(unwrapped, dt=10.0, simbox=f.box)
g_of_r, msd_curve = gr.finish(), msd.finish()
```

Prefer dumps that already store continuous paths (`xu yu zu`, or `x y z` plus
`ix iy iz` unwrapped on read). Given each frame's box, `Msd` checks that no
atom jumps by more than half a box between frames and refuses a wrapped
trajectory instead of returning a curve of box-sized jumps; `Unwrap` turns
wrapped coordinates (with image flags, or by continuity) into the continuous
path.

The reader parses frames on demand, so the loop above never holds more than
one frame; `reader[i]` reads a single frame by random access. Other formats
live in [`molpy.io`](../user-guide/11_io.md) (`read_gro`, `read_dcd`,
`read_xtc`, …).

### Partial (species-resolved) distributions

There is no species argument on `Rdf` or the other structural computes. To get
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
mixture.box = mp.Box(mp.core.Cell.lengths([20.0] * 3))

selected = mixture["atoms"]["type_id"] == 1
subset = mp.Frame()
subset["atoms"] = {k: mixture["atoms"][k][selected] for k in ("x", "y", "z")}
subset.box = mixture.box
print(subset["atoms"].n_rows)                       # -> 143
```

The density used for normalization then comes from the subset, which is what you
want for a like–like partial.

A cross-species $g_{AB}(r)$ needs pairs between two *different* point sets.
`mp.core.NeighborQuery` indexes one set and queries it with the other, and the
directed table it returns is what `Rdf` normalises against both counts:

```python
species = mixture["atoms"]["type_id"]
a_xyz = mixture.coords[species == 1]
b_xyz = mixture.coords[species == 2]

cross = mp.core.NeighborQuery(mixture.box, b_xyz, 8.0).query(a_xyz)
partial = Rdf(n_bins=40, r_max=8.0, r_min=0.0).accumulator()
partial.push(cross)
g_ab = partial.finish()
print(g_ab.n_query_points, g_ab.n_points, g_ab.self_query)  # -> 143 157 False
```

## Which analysis answers which question

The pages are grouped the way the questions group, not the way the classes do.

### Structure — where the atoms are

| Question | Page | Entry points |
|---|---|---|
| Which atoms are near which? | [NeighborList](neighborlist.md) | `NeighborList` |
| How are neighbours spaced? | [RDF](rdf.md) | `Rdf` |
| Where is the matter, in space? | [Density](density.md) | `LocalDensity`, `GaussianDensity`, `SphereVoxelization` |
| What would a diffraction experiment see? | [Diffraction](diffraction.md) | `StaticStructureFactorDebye`, `StaticStructureFactorDirect`, `DiffractionPattern` |
| Where do neighbours sit around an anisotropic particle? | [PMFT](pmft.md) | `PmftXy`, `PmftXyt`, `PmftXyz`, `PmftR12` |
| How are bond lengths, angles, torsions distributed? | [Distribution](distribution.md) | `DistributionFunction`, `CombinedDistribution` |
| Where do neighbours sit in a molecule's own frame? | [Spatial](spatial.md) | `SpatialDistribution` |

### Order, shape, and aggregates — what the structure adds up to

| Question | Page | Entry points |
|---|---|---|
| Crystalline, hexatic, or nematic order? | [Order](order.md) | `Steinhardt`, `Hexatic`, `Nematic`, `SolidLiquid` |
| What does the local environment look like? | [Environment](environment.md) | `BondOrientationalOrder` |
| How big and how anisotropic is a molecule? | [Shape](shape.md) | `CenterOfMass`, `GyrationTensor`, `InertiaTensor` |
| What is connected to what? | [Cluster](cluster.md) | `Cluster` → `CenterOfMass`, `GyrationTensor` |
| What are the dominant collective coordinates? | [Decomposition](decomposition.md) | `Pca`, `Kmeans` |
| Which atoms are hydrogen-bonded? | [HBond](hbond.md) | `HBonds` |
| How much space does each atom own? | [Voronoi](voronoi.md) | `RadicalVoronoi`, `VoronoiIntegration`, `VoronoiDomainAnalysis` |

### Motion — how things move and how fast

| Question | Page | Entry points |
|---|---|---|
| How far do atoms wander? (self-diffusion) | [MSD](msd.md) | `Msd` → `LinearFit` |
| What does the velocity remember? | [VACF](vacf.md) | `Vacf` → `CumulativeTrapezoid` → `Plateau` |
| Ionic conductivity, from displacements? | [PMSD](pmsd.md) | `DipoleMoment` → `Msd` → `LinearFit` |
| …and from currents? | [JACF](jacf.md) | `CurrentAcf` → `CumulativeTrapezoid` |
| Do ions move together or independently? | [Onsager](onsager.md) | `OnsagerCorrelation` |
| How long does a contact survive? | [Persist](persist.md) | `PairPresence` → `Survival` |
| How does a density fluctuation decay? | [Van Hove](van_hove.md) | `VanHove` |
| How fast do molecules tumble? | [Reorientation](reorientation.md) | `ReorientationAcf`, `RotationalAutocorrelation` |

### Spectra and composition

| Question | Page | Entry points |
|---|---|---|
| What is $\varepsilon^*(\omega)$? | [Dielectric](dielectric.md) | `StaticDielectric`, `DipoleAcf` → `EinsteinHelfandSpectrum`, `CurrentAcf` → `GreenKuboSpectrum`, `DebyeFit` |
| Vibrational, IR, or Raman spectrum? | [Spectra](spectra.md) | `Vacf` → `Spectrum`, `Derivative` → `CurrentAcf` → `Spectrum`, `RamanSpectrum` |
| A correlation no class names? | [Spectra](spectra.md) | `Correlator`, `Spectrum` |

## Where the figures come from

Every curve on these pages is computed, not drawn. The reference system is 500
argon atoms at 1.374 g cm⁻³, the density of Rahman's study, equilibrated by
velocity rescaling at 85 K and then integrated for 30 ps at constant energy,
conserving total energy to a relative drift of $2.0\times10^{-5}$
(dimensionless, $|E(t)-E(0)|/|E(0)|$). Once the thermostat is off, kinetic and
potential energy keep trading, so a constant-energy run settles near the target
temperature rather than on it: the reference run (seed 0), behind every
structural figure, averages 88.3 K. The transport pages ([MSD](msd.md),
[VACF](vacf.md)) average over eight such runs that differ only in the seed of
their initial velocities, because one run of a chaotic liquid is one noisy draw,
and quote each coefficient with its spread over the runs; their mean
temperature is 84.9 K. A caption names the run its numbers come from.
Every run is a pure function of its seed, with nothing cached between runs.
The generator lives in `molpy/scripts/docs_series/` and writes to `docs/series/`, so any
figure can be reproduced or challenged:

```python
# docs: skip — runs a 30 ps MD trajectory (minutes, not seconds)
from docs_series.run import argon_trajectory
from docs_series.structure import radial_distribution

radial_distribution(argon_trajectory())
```

Where no honest dataset exists yet, the page says so and carries a `TODO`
instead of a decorative sketch.

## Related

- [API reference: Compute](../api/compute.md)
- [Tutorials: Trajectory](../tutorials/05_trajectory.md)
- [Tutorials: Box and Periodicity](../tutorials/03_box_and_periodicity.md)
