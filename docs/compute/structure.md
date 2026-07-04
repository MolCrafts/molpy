# Structural Analysis: Pair Structure, Structure Factor & Density

This page is a self-contained, textbook-style introduction to the **structural**
compute operators in MolPy: the radial distribution function $g(r)$, the static
structure factor $S(k)$, local and grid number densities, the neighbor-list
primitive they all build on, and the potential of mean force. These are the tools
that answer "how is matter arranged?" — the static counterpart to the dynamical
analyses in the [Diffusion & Ionic Transport](transport.md) guide. No background
beyond the pair-distribution idea is assumed.

As elsewhere in `compute`, every heavy kernel (pair binning, the Debye sum, grid
smearing) runs in Rust inside `molrs`; the MolPy layer extracts coordinates and
box dimensions and returns a typed result.

!!! note "Conventions used throughout"
    - Length is in Å, wavenumber $k$ in Å⁻¹, number density $\rho$ in Å⁻³.
    - $g(r)$ and $S(k)$ are dimensionless.
    - A **frame** must carry a periodic `box`; the minimum-image convention is
      used for every pair distance.
    - The structural operators are *frame* analyses — feed one `Frame`, or a list
      of frames to average over.

---

## 1. The pair distribution function counts neighbours relative to an ideal gas

The radial distribution function $g(r)$ is the probability of finding a particle
at distance $r$ from a reference particle, **normalized by what an ideal gas of
the same density would give**. Formally, for $N$ particles in volume $V$ with
number density $\rho = N/V$,

$$
g(r) = \frac{V}{N^2}\Big\langle\sum_{i}\sum_{j\ne i}\delta\big(r - r_{ij}\big)\Big\rangle
       \Big/ 4\pi r^2 .
$$

The $4\pi r^2$ is the surface area of the shell at radius $r$: dividing the raw
pair count in a shell by its volume $4\pi r^2\,\Delta r$ and by the bulk density
$\rho$ is exactly what turns a histogram into $g(r)$. The limits are the things to
remember:

- $g(r)\to 0$ as $r\to 0$ — excluded volume; particles cannot overlap.
- $g(r)\to 1$ as $r\to\infty$ — at long range the structure washes out and the
  local density equals the bulk density.
- **Peaks are coordination shells.** The first peak is the nearest-neighbour
  distance; its first following minimum defines the first solvation shell.

---

## 2. Coordination numbers are the running integral of g(r)

Integrating $g(r)$ over a shell recovers the **number of neighbours** out to a
distance $R$:

$$
n(R) = 4\pi\rho \int_0^{R} r^2\, g(r)\, \mathrm{d}r .
$$

Evaluating $n(R)$ at the first minimum of $g(r)$ gives the **coordination
number** — the average count in the first shell. This is the single most useful
number to extract from an RDF, and it is the natural place to read off the cutoff
for a [pair-persistence](persistence.md) or clustering analysis.

---

## 3. Computing g(r) with `RDF`

`RDF` bins pair distances supplied by a `NeighborList`, so the workflow is always
*build neighbours → histogram*. The neighbor cutoff must be at least `r_max`.

```python
from molpy.compute import NeighborList, RDF

nlist = NeighborList(cutoff=10.0)(frame)        # pairs within 10 Å
rdf = RDF(n_bins=200, r_max=10.0)               # configure
result = rdf([frame], [nlist])                  # call on data -> RDFResult

result.rdf          # g(r), shape (n_bins,)
result.bin_centers  # r at each bin centre, Å
```

Pass parallel lists of frames and neighbor lists to average over a trajectory:

```python
nlists = [NeighborList(cutoff=10.0)(f) for f in frames]
result = RDF(n_bins=200, r_max=10.0)(frames, nlists)   # ensemble-averaged g(r)
```

`result.bin_edges`, `result.volume`, and `result.n_frames` carry the histogram
geometry and the normalization bookkeeping.

---

## 4. The structure factor is g(r) seen in reciprocal space

Scattering experiments do not measure $g(r)$ directly; they measure the **static
structure factor** $S(k)$, the Fourier-space image of the same pair correlations.
MolPy computes it straight from coordinates with the **Debye scattering
equation**:

$$
S(k) = \frac{1}{N}\Big\langle\sum_{i}\sum_{j}\frac{\sin(k\,r_{ij})}{k\,r_{ij}}\Big\rangle .
$$

For an isotropic system $S(k)$ is related to $g(r)$ by the Fourier transform

$$
S(k) = 1 + 4\pi\rho\int_0^\infty r^2\,[g(r)-1]\,\frac{\sin(kr)}{kr}\,\mathrm{d}r ,
$$

so the two contain the same information. $S(k)$ is the right view when you want to
compare with X-ray/neutron diffraction, locate a structural length scale (the
first sharp diffraction peak), or diagnose long-wavelength density fluctuations
($S(k\to 0)$ is set by the isothermal compressibility).

---

## 5. Computing S(k) with `StaticStructureFactorDebye`

You choose the wavenumber grid; the Debye sum is evaluated directly (no binning,
so it is exact for the chosen $k$ but $\mathcal{O}(N^2)$ per frame).

```python
import numpy as np
from molpy.compute import StaticStructureFactorDebye

k = np.linspace(0.2, 12.0, 300)             # Å^-1; avoid k = 0
sk = StaticStructureFactorDebye(k)([frame]) # call on one or more frames
```

Because the cost scales with the square of the particle count, use the
Debye route for small-to-medium systems or sub-sampled frames; for very large
boxes prefer the FFT-on-a-grid structure factor (not covered here).

---

## 6. Local and grid densities resolve where matter sits

Two operators turn positions into a density field:

- **`LocalDensity`** assigns each particle the number density inside a cutoff
  sphere around it — a per-particle scalar, useful for detecting interfaces,
  voids, or local packing variations. Like `RDF` it consumes a neighbor list.
- **`GaussianDensity`** smears every particle with a Gaussian of width `sigma`
  onto a fixed 3-D grid, producing a continuous $\rho(\mathbf r)$ field suitable
  for visualization or for locating adsorption sites.

```python
from molpy.compute import LocalDensity, GaussianDensity

nlist = NeighborList(cutoff=4.0)(frame)
local = LocalDensity(r_max=4.0)([frame], [nlist])   # per-particle density

grid = GaussianDensity(nx=64, ny=64, nz=64, sigma=1.0)([frame])  # ρ(r) on a grid
```

The cutoff (`r_max`) and the smearing width (`sigma`) set the resolution: too
small and the field is shot-noise; too large and real features are washed out.

---

## 7. The neighbor list is the shared primitive

`RDF`, `LocalDensity`, the order parameters, clustering, and PMFT all consume a
**`NeighborList`** — a single spatial query that returns every pair within a
cutoff under the minimum-image convention. It is worth understanding directly,
both because it is reusable across analyses and because its cutoff governs their
cost and meaning.

```python
nlist = NeighborList(cutoff=5.0)(frame)
nlist.n_pairs            # number of pairs found
nlist.pairs              # (n_pairs, 2) index array
nlist.distances          # pair distances, Å
```

Build it once per frame and pass it to every cutoff-based analysis on that frame.

---

## 8. Potentials of mean force turn structure into free energy

A pair distribution is a Boltzmann factor in disguise. The **potential of mean
force** (PMF) is

$$
w(r) = -k_\mathrm{B}T \ln g(r),
$$

the effective free energy along the pair separation after averaging over all
other degrees of freedom — its minima are the coordination shells of §1, its
barriers the desolvation penalties between them.

`PMFTXY` generalizes this to a **2-D potential of mean force and torque**:
instead of a single $r$, it accumulates neighbour positions in the local
$(x, y)$ frame of each reference particle, exposing directional structure that an
isotropic $g(r)$ averages away — face-to-face vs edge-to-edge contacts, for
example. When the frame carries an `orientations` topology block (one
`(head, tail)` atom pair per particle), every bond is rotated into that
particle's local frame — the per-particle angle is `atan2` of its `head - tail`
axis. Without the block the analyzer works in the lab frame.

```python
from molpy.compute import PMFTXY

nlist = NeighborList(cutoff=6.0)(frame)
pmft = PMFTXY(x_max=6.0, y_max=6.0, n_x=120, n_y=120)
result = pmft([frame], [nlist])   # aligned if frame has an `orientations` block, else lab frame
```

---

## 9. Parameters and hyperparameters

### 9.1 Parameters and their meaning

| Parameter | Compute | Meaning |
|---|---|---|
| `n_bins` | `RDF` | number of histogram bins over $[r_\text{min}, r_\text{max}]$ (100–300 typical) |
| `r_max` | `RDF` | upper edge of the last bin, **Å**; keep $\le L/2$ (minimum image) |
| `r_min` | `RDF` | lower edge of bin 0, **Å** (default `0.0`) |
| `k_values` | `StaticStructureFactorDebye` | wavenumber grid, **Å⁻¹**; start above 0 — the smallest physically meaningful $k$ is $\approx 2\pi/L$ |
| `cutoff` | `NeighborList` | pair-search radius, **Å**; must be **≥ the `r_max`** of every analysis that consumes the list |
| `r_max`, `diameter` | `LocalDensity` | counting-sphere radius, **Å**; `diameter` (default `0.0`) applies a particle-size correction — `0.0` counts centres only |
| `nx`, `ny`, `nz`, `sigma` | `GaussianDensity` | grid resolution per axis; Gaussian smearing width, **Å** |

Bin geometry matches the Rust kernel exactly: bin width
$\Delta r = (r_\text{max} - r_\text{min})/n_\text{bins}$ and bin centres
$r_i = r_\text{min} + (i + \tfrac12)\,\Delta r$ — what `result.bin_centers`
returns.

### 9.2 Hyperparameter effects

- **`n_bins` (bin width vs noise).** Counts per bin scale as $1/n_\text{bins}$,
  so the relative shot noise of each bin grows as
  $\sigma_\text{bin} \propto 1/\sqrt{\text{counts}} \propto \sqrt{n_\text{bins}}$.
  Too **few** bins under-sample the first peak — its height reads low and the
  coordination number of §2 with it (aim for ≥ 5 bins across the first peak).
  Too **many** bins buy resolution you pay for in frames.
- **`r_max` beyond half the box.** Distances past $L/2$ are contaminated by
  periodic images under the minimum-image convention; the tail of $g(r)$ stops
  meaning anything. This is a *correctness* limit, not a resolution knob.
- **`r_max` / `cutoff` and cost.** The number of pairs a `NeighborList` returns
  grows as $\rho\,r^3$; the linked-cell build and every downstream histogram
  scale with it. Do not query 12 Å of neighbours to histogram 6 Å of $g(r)$.
- **Debye $k$-grid density.** The Debye sum is exact at each requested $k$ but
  costs $\mathcal{O}(N^2)$ **per $k$ per frame** — a 300-point grid is 300
  double sums. Refine the grid only where features live (the first sharp
  diffraction peak), and remember $k < 2\pi/L$ probes fluctuations larger than
  the box.
- **Number of frames.** $g(r)$ and $S(k)$ are ensemble averages; the noise of
  every bin falls as $1/\sqrt{N_\text{frames}}$ for *uncorrelated* frames.
  Frames closer in time than the structural relaxation time add cost, not
  statistics.
- **`LocalDensity.r_max` / `GaussianDensity.sigma` (field resolution).** Both
  set the smoothing length of the density field: too small → shot noise per
  particle, too large → interfaces and voids wash out (see §6). Choose them
  against a physical length (first $g(r)$ minimum, interface width), not the
  grid.

The failure modes of these knobs are collected in
[§10](#10-pitfalls-checklist).

---

## 10. Pitfalls checklist

1. **`r_max` (or the largest $k$ feature) beyond half the box** → periodic images
   contaminate the result. Keep `r_max ≤ L/2` for the minimum-image convention.
2. **Neighbor cutoff smaller than `r_max`** → the RDF is truncated; the neighbor
   `cutoff` must be at least the histogram `r_max`.
3. **Too few bins** → a sharp first peak is smeared and the coordination number
   reads low; too many and each bin is noisy. 100–300 bins is typical.
4. **$k = 0$ in the Debye grid** → division by zero; start the grid at a small
   positive $k$.
5. **Single frame for an ensemble quantity** → $g(r)$ and $S(k)$ are statistical;
   average over many uncorrelated frames before trusting peak heights.
6. **Free box** → all of these require a periodic `box`; a free frame raises.

---

## 11. References

- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
  Oxford (2017) — $g(r)$, coordination numbers, and structure factors.
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed., Academic
  (2013) — the $g(r)\!\leftrightarrow\!S(k)$ relation and compressibility sum rule.
- P. Debye, *Ann. Phys.* **351**, 809 (1915) — the Debye scattering equation.
- G. van Anders et al., *ACS Nano* **8**, 931 (2014) — potential of mean force
  and torque.
- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) — the
  freud library, on which these kernels are modelled.

## See also

- [Diffusion & Ionic Transport](transport.md) — the dynamical counterpart (MSD,
  Onsager, conductivity).
- [Bond-Orientational Order & Local Environment](order.md) — when *which*
  neighbours matter, not just how many.
- [Pair Persistence](persistence.md) — pick its cutoff from the first $g(r)$ minimum.
- [Compute overview](index.md) — the Compute → Result pattern.
- [API reference: Compute](../api/compute.md).
