# Density

[$g(r)$](rdf.md) answers "how far apart are atoms, on average" — it averages
over every direction and every atom. Sometimes that is exactly the wrong
question. If your box contains a droplet, a slab, a pore, or an interface, you
want to know **where** the matter is, not just how it is spaced.

Two computes answer that, and they answer different questions.

## Two ways to ask "how dense is it here"

**`LocalDensity` gives every atom a number.** Put a sphere of radius `r_max`
around atom $i$, count the atoms inside, divide by the sphere's volume:

$$
\rho_i = \frac{N_i(r_\max)}{\tfrac{4}{3}\pi r_\max^{3}}.
$$

You get one density per atom, so you can colour a rendering by it, histogram it,
or select the atoms in the dense phase.

**`GaussianDensity` gives space a number.** Forget which atom is which; smear
each one into a Gaussian of width $\sigma$ and add them all onto a fixed grid:

$$
\rho(\mathbf{r}) = \sum_i
\frac{1}{(2\pi\sigma^{2})^{3/2}}
\exp\!\left(-\frac{|\mathbf{r}-\mathbf{r}_i|^{2}}{2\sigma^{2}}\right).
$$

You get a continuous field on a grid, which is what you want for volume
rendering, for slicing along an interface normal, or for feeding an adsorption
map.

The first is per-particle and needs a [neighbor list](neighborlist.md). The
second is per-voxel and does not.

## The probe radius is the whole story

Neither number means anything until you say over what length scale it was
measured. This is the part students trip on, so it is worth seeing directly.

Below is the distribution of per-atom local density in *homogeneous* liquid
argon — a system with no interfaces, no droplets, nothing to find — measured
with two probe radii.

<figure id="fig-local-density" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
config:
  legend:
    orient: bottom
    direction: horizontal
    title: null
data: {$file: series/density/local_histogram.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: density
    type: quantitative
    title: "ρ_local (Å⁻³)"
    scale: {domain: [0.005, 0.04]}
  y:
    field: p
    type: quantitative
    title: "pdf"
  color:
    field: probe
    type: nominal
    title: null
```

</div>

**Figure 1.** Distribution of local densities in bulk liquid argon, from
the reference run (seed 0, 88.3 K), for probe radii of 4 Å and 8 Å. Both are centred on the same physical
density; the narrow one is not "better data", it is more smoothing.
</figure>

Both distributions sit on the bulk density, 0.0207 Å⁻³. What differs is the
width:

| Probe `r_max` | ⟨N⟩ in the sphere | mean $\rho$ (Å⁻³) | spread | Poisson would give |
|---|---|---|---|---|
| 4 Å | 5.5 | 0.0206 | 27 % | 43 % |
| 8 Å | 45.1 | 0.0210 | 5.3 % | 15 % |

A small sphere holds few atoms, so gaining or losing one moves the answer a lot;
that is why the 4 Å curve is broad. But notice the last column. If the atoms
were placed independently, the count in a sphere would be Poisson-distributed
with relative width $1/\sqrt{\langle N\rangle}$ — 43 % and 15 %. The measured
widths are much *narrower* than that, and the gap widens as the probe grows.

This is not an error, and it is the most physical thing on the page. A liquid is
nearly incompressible: pushing extra atoms into a region costs energy, so
density fluctuations are strongly suppressed relative to independent placement.
Formally the suppression factor tends to $S(0)$, the $k\to 0$ limit of the
[structure factor](diffraction.md), which is proportional to the isothermal
compressibility. So the width of this histogram is a *thermodynamic*
measurement wearing a geometric disguise — and quoting it as "counting noise"
would throw the physics away.

The practical lesson still holds: the probe radius is a resolution knob with a
cost at both ends. Too small and the distribution is dominated by how few atoms
fit inside; too large and a real interface is averaged across its own width and
vanishes. Pick it from a physical length — the first minimum of $g(r)$ if you
mean "the first shell", or the expected interface thickness if you mean "which
phase is this atom in".

One consequence surprises people: the mean local density around an atom is not
exactly the bulk density. You are measuring from an atom, and atoms have
neighbours, so you are sampling the $g(r)$-weighted density

$$
\langle\rho_{\text{local}}(R)\rangle = \frac{n(R)}{\tfrac{4}{3}\pi R^{3}},
$$

with $n(R)$ the coordination number from [RDF](rdf.md), which that page shows how
to compute in four lines. For argon $n(8\,\text{Å}) = 44.7$, and dividing by the
sphere volume $\tfrac{4}{3}\pi 8^3 = 2145$ Å³ gives 0.0208 Å⁻³ against the
0.0210 measured here — two different computes agreeing. The local density
approaches the bulk value only as $R$ grows past the last correlation shell.

## Computing both

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(400, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box(mp.core.Cell.lengths([20.0] * 3))
```

`LocalDensity` is configured with the probe radius and computed on one
snapshot's pairs — a [neighbour table](neighborlist.md) or the built search
itself. It returns a result with two arrays of length $N$: `n_neighbors`, the
(possibly fractional) count inside the probe, and `density`, that count over
the probe volume:

```python
from molpy.compute import LocalDensity

nl = mp.core.NeighborList(5.0)
nl.build(frame.coords, frame.box)
nlist = nl.neighbors()
local = LocalDensity(r_max=5.0).compute(pairs=nlist)

print(local.n_neighbors.shape, local.density.shape)   # -> (400,) (400,)
print(round(float(local.density.mean()), 4))          # -> 0.0507
```

A trajectory is a loop: one table per frame, one result per table, and the
per-atom densities concatenated or averaged as the question requires.

Random points have no structure, so the mean local density here should be the
bulk value $400/20^3 = 0.05$ Å⁻³, and it is. That is a good habit: run a compute
on an uncorrelated system first, where you know the answer.

Note that `r_max` and the neighbor list's `cutoff` are two separate numbers that
you must keep consistent. `LocalDensity` counts only pairs the list contains, so
a cutoff smaller than `r_max` silently undercounts and reports a density that is
too low. Set them equal unless you have a reason not to.

The optional `diameter` argument softens the hard edge of the counting sphere.
With the default `0.0` an atom is either in or out, and a particle drifting
across the boundary makes the density jump. With `diameter = d`, atoms near the
edge are counted with a weight that ramps linearly from 1 to 0 across a shell of
thickness $d$ centred on `r_max`:

$$
w_j = \mathrm{clamp}\!\left(
\frac{r_\max + d/2 - r_{ij}}{d},\, 0,\, 1\right),
$$

which reduces to plain counting when $d = 0$. Use it when you care about the
density of *finite-sized* particles rather than of points, or when a jumpy
per-atom density is causing trouble downstream.

`GaussianDensity` needs no pairs: it is configured with the grid resolution
and the kernel width, and computed on one snapshot's positions and the box the
grid divides. The positions are columns you read from the frame:

```python
from molpy.compute import GaussianDensity

field = GaussianDensity(dims=(32, 32, 32), sigma=1.5).compute(
    positions=frame.coords, simbox=frame.box
)
print(field.density.shape)                   # -> (32, 32, 32)
print(round(field.grid.voxel_volume, 4))     # -> 0.2441
```

The result carries the field `density` and the `grid` it lives on (origin,
voxel basis, periodicity), so nothing downstream has to re-derive the voxel
geometry. Grid spacing is the box edge divided by the resolution — here
20/32 = 0.63 Å.
Keep $\sigma$ at or above the spacing, or you are sampling a Gaussian narrower
than your own grid and the field becomes a set of spikes.

One property of this kernel is worth knowing before you integrate anything from
it. Each Gaussian is truncated at $3\sigma$ (or at `r_max`, if you set one) and
the remaining tail is *not* folded back in, so the grid integrates to 97.1 % of
the particle count rather than 100 %:

```python
voxel = field.grid.voxel_volume
print(round(float(field.density.sum() * voxel), 2))   # -> 388.28, not 400
```

The deficit is the fraction of a 3-D Gaussian lying beyond $3\sigma$, so as long
as the grid actually resolves the Gaussian it is a fixed 2.9 %, whatever
$\sigma$ and resolution you choose. Under-resolve it and you lose more: at
`dims=(16, 16, 16)` with $\sigma = 0.3$ Å the spacing is 1.25 Å, four times
$\sigma$, and only the voxel centres that happen to fall inside the truncation
sphere receive any mass.

How many of those centres a particle reaches is a convention, and the
constructor names it with `reach`. The default, `reach="ceil"`, visits every
voxel whose centre lies within the truncation radius. `reach="floor"` is
freud's stencil: it stops $\lfloor r_\max/\Delta\rfloor$ voxels from the
particle's own voxel and so skips the outermost shell of centres. When the grid
resolves the Gaussian the two agree; when it does not, they differ visibly:

```python
coarse = {
    reach: GaussianDensity(dims=(16, 16, 16), sigma=0.3, reach=reach).compute(
        positions=frame.coords, simbox=frame.box
    )
    for reach in ("ceil", "floor")
}
for reach, out in coarse.items():
    print(reach, round(float(out.density.sum() * out.grid.voxel_volume), 1))
# -> ceil 373.9
# -> floor 343.8
```

Pick `floor` only to reproduce numbers from freud. Either way the lesson is the
same: the loss is harmless for visualization and for ratios, but if you need
absolute particle counts from the field, keep $\sigma$ at or above the spacing
and divide by 0.971.

When you want to know *whether* space is occupied rather than how much,
replace the Gaussian with a hard sphere. `SphereVoxelization` marks every voxel
whose centre lies within `r_max` of some particle; `voxels` is the 0/1
occupancy and `raw_counts` how many spheres cover each voxel. For random points
the occupied fraction is the Poisson probability that a sphere of radius
`r_max` around a voxel holds at least one particle, $1 - e^{-\rho\,4\pi r^3/3}$:

```python
from molpy.compute import SphereVoxelization

cover = SphereVoxelization(dims=(32, 32, 32), r_max=1.5).compute(
    positions=frame.coords, simbox=frame.box
)
rho = 400 / 20.0**3
print(round(float(cover.voxels.mean()), 3))                       # -> 0.504
print(round(float(1 - np.exp(-rho * 4 / 3 * np.pi * 1.5**3)), 3))  # -> 0.507
```

!!! note "No figure for `GaussianDensity` yet — TODO"
    A density-field slice only says anything for an inhomogeneous system: a
    slab, a droplet, a pore. The reference trajectory behind every figure in
    these pages is bulk argon, which renders as featureless noise, so there is
    nothing honest to plot. This page will get a figure when an interfacial
    trajectory is added under `scripts/docs_series/`.

## When it goes wrong

**Every atom reports nearly the same density, and you expected two phases.**
`r_max` is larger than the feature you are looking for. A 10 Å probe cannot see
a 5 Å interface.

**The distribution is wide and lumpy, and it changes every frame.**
`r_max` is too small — you are looking at counting noise. Widen it, or average
over frames.

**`LocalDensity` values look systematically low near a boundary.**
This is expected only if the box is non-periodic. With a periodic box the sphere
wraps and there is no edge effect; if you see one, check `frame.box`.

**The Gaussian grid is mostly zeros with sharp dots.**
$\sigma$ is smaller than the grid spacing. Raise $\sigma$ or the resolution.

**Densities from two systems disagree and you cannot see why.**
Compare the probe radii before anything else. A local density without its
`r_max` (or $\sigma$) is not a reportable number.

## Check yourself

- Scatter $N$ points at random in a box and check that the mean `LocalDensity`
  equals $N/V$. Then halve `r_max` and confirm the *mean* is unchanged while the
  *spread* roughly doubles.
- Sum a `GaussianDensity` grid over all voxels, multiply by the voxel volume,
  and check you get $0.971 N$ whenever $\sigma$ is at least the grid spacing.
  Then work out where the missing 2.9 % went. (Hint: the fraction of a 3-D
  Gaussian beyond $3\sigma$.)
- Predict which of `r_max = 3` Å or `r_max = 10` Å gives the larger standard
  deviation, then measure it.

## References

- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) — the
  freud density kernels this API mirrors.
- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
  Oxford (2017) — density profiles and interfacial averaging.

## See also

- [RDF](rdf.md) — the same information averaged over all directions
- [NeighborList](neighborlist.md) — what `LocalDensity` consumes
- [Voronoi](voronoi.md) — a parameter-free alternative local volume
- [API reference](../api/compute.md)
