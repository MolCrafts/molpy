# Orientational autocorrelation

[Reorientation](reorientation.md) explains what the Legendre correlation
functions $C_\ell(t)$ measure and reads them off a frame's bond list. This page
is the same measurement with the vector in your hands: you build the unit
vectors, `Oacf` correlates them.

That is the right shape for most real questions. The vector that matters is
rarely a single bond — it is the O–H direction of every water molecule under
the box's minimum image, a molecular dipole, the normal of an aromatic ring,
the long axis of a chain. Building it is a few lines of NumPy you want to see;
the correlation is the part you do not want to write twice.

## The quantity

For a unit vector $\hat{\mathbf{u}}$ fixed in the molecule,

$$
C_\ell(t) = \big\langle P_\ell\big(\hat{\mathbf{u}}(\tau)\cdot\hat{\mathbf{u}}(\tau+t)\big)\big\rangle_{\tau,\ \text{molecules}},
\qquad P_1(x) = x,\quad P_2(x) = \tfrac12(3x^2 - 1),
$$

averaged over every molecule and every time origin $\tau$. $C_\ell(0) = 1$ and
the curve decays to zero in a liquid. Its integral,

$$
\tau_\ell = \int_0^\infty C_\ell(t)\,\mathrm{d}t,
$$

is the reorientational correlation time — model-free, unlike the single
exponential a fit assumes. $\ell = 1$ is what dielectric relaxation and
infrared absorption see; $\ell = 2$ is what NMR relaxation, depolarized light
scattering and fluorescence anisotropy see, and it is even under
$\hat{\mathbf{u}} \to -\hat{\mathbf{u}}$, so an axis without a direction still
has a $C_2$.

## How it is computed

$C_1$ is the vector autocorrelation itself, one FFT per vector and component.
$C_2$ looks like it should not go through an FFT, because the square sits inside
the average — but $(\hat{\mathbf{u}}\cdot\hat{\mathbf{v}})^2$ is the dot
product of the six independent components of $\hat{\mathbf{u}}\otimes\hat{\mathbf{u}}$
(the three squares with weight 1, the three cross terms with weight 2), so
$C_2 = \tfrac32\langle Q(\tau)\cdot Q(\tau+t)\rangle - \tfrac12$ is six FFTs
per vector. Both orders are exact, $\mathcal{O}(T\log T)$, and the same
estimator as every other correlation in molpy; orders above 2 fall back to the
direct origin sum.

## Building the vectors: O–H in water

The vectors are an `(n_frames, n_vectors, 3)` array of **unit** vectors. For
water, one per hydrogen, pointing from the hydrogen to its nearest oxygen under
the minimum image:

```python
# docs: skip — needs your own trajectory
import numpy as np
from molpy.compute import Oacf

def oh_vectors(pos, o_idx, h_idx, box_length):
    """Unit vectors H -> nearest O, minimum image, shape (n_frames, n_H, 3)."""
    out = np.empty((len(pos), len(h_idx), 3))
    for t, xyz in enumerate(pos):
        d = xyz[o_idx][None, :, :] - xyz[h_idx][:, None, :]      # (n_H, n_O, 3)
        d -= box_length * np.round(d / box_length)                # minimum image
        r = np.linalg.norm(d, axis=2)
        nearest = d[np.arange(len(h_idx)), r.argmin(axis=1)]
        out[t] = nearest / np.linalg.norm(nearest, axis=1, keepdims=True)
    return out

u = oh_vectors(pos, o_idx, h_idx, box_length)         # pos: (n_frames, n_atoms, 3)
c2 = Oacf(order=2).compute(u, max_lag=2000, dt=0.05)  # dt in ps -> lag_times in ps
tau2 = c2.integrate()                                 # ps
```

Normalize before you pass the array: a vector whose length is not 1 raises
`ValueError` rather than being silently rescaled, because a wrong length almost
always means the wrong pair of atoms.

## On a known case

Rods undergoing pure rotational diffusion have $C_\ell(t) = e^{-\ell(\ell+1)D_r t}$,
so $\tau_1/\tau_2 = 3$. Here are 200 of them, built directly as unit vectors —
no frames, no bond block:

```python
import numpy as np
from molpy.compute import Oacf

rng = np.random.default_rng(0)
n_rods, n_frames = 200, 400
axis = rng.normal(size=(n_rods, 3))
axis /= np.linalg.norm(axis, axis=1, keepdims=True)

u = np.empty((n_frames, n_rods, 3))
for t in range(n_frames):
    kick = rng.normal(0.0, 0.06, size=(n_rods, 3))
    kick -= (kick * axis).sum(axis=1, keepdims=True) * axis   # rotation only
    axis = axis + kick
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    u[t] = axis

c1 = Oacf(order=1).compute(u, max_lag=200)
c2 = Oacf(order=2).compute(u, max_lag=200)
print(c1.kind, c1.order, c2.order, round(float(c1.values[0]), 6))   # -> oacf 1 2 1.0
print(round(c1.fit_exponential()["tau"] / c2.fit_exponential()["tau"], 1))   # -> 3.0
```

`fit_exponential` is `DebyeFit` on the normalized curve: a single exponential
through the leading positive run. `integrate()` makes no such assumption and is
what to quote when the decay is stretched, as it is for water.

The result is a `TimeCorrelation`: `values` is the curve, `lags` the lag axis in
frames and `lag_times` in your `dt` unit, `counts` the number of time origins
behind each lag — the statistics thin out at long lag exactly as for an
[MSD](msd.md). `per_entity=True` keeps one curve per vector.

```python
c2 = Oacf(order=2, stride=4).compute(u, max_lag=50, per_entity=True)
print(c2.per_entity.shape, int(c2.counts[0]), int(c2.counts[-1]))   # -> (51, 200) 100 88
```

`stride` takes every fourth frame as a time origin: four times fewer origins,
four times faster, and the same curve to within noise — the way to make a long
trajectory affordable.

## When it goes wrong

**`ValueError: ... unit vectors required`.**
A vector has length $\ne 1$: normalize, and check the pairing that produced it.

**$C_2(0) \ne 1$ or the curve does not decay.**
The vectors are not reorienting — frozen molecules, or the same vector copied
into every frame.

**$\tau_1/\tau_2$ far from 3.**
Physics or statistics, as on the [Reorientation](reorientation.md) page: jump
reorientation and hydrogen bonding push the ratio away from 3, a short
trajectory makes the long-lag tail noise. Compare `integrate()` with
`fit_exponential()`; if they disagree, the decay is not a single exponential
and the fitted $\tau$ is not the number to quote.

## References

- B. J. Berne, R. Pecora, *Dynamic Light Scattering*, Wiley (1976), ch. 7 —
  which spectroscopy sees which $\ell$.
- G. Lipari, A. Szabo, *J. Am. Chem. Soc.* **104**, 4546 (1982) — $C_2$ as the
  NMR relaxation correlation function.
- D. Laage, J. T. Hynes, *Science* **311**, 832 (2006) — O–H reorientation in
  water by large-angle jumps.

## See also

- [Reorientation](reorientation.md) — the same correlation from a frame's bonds
- [VACF](vacf.md) — the `TimeCorrelation` object and its derived quantities
- [API reference](../api/compute.md)
