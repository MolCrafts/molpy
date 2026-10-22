# Onsager coefficients

[PMSD](pmsd.md) says the conductivity of a real electrolyte sits below the
Nernst–Einstein estimate, because ions do not move independently. This page is
how you find out *which* correlations are responsible.

## Splitting the collective motion by species

Instead of one collective vector for the whole system, track one per species.
For species $\alpha$ — all the cations, say — define

$$
\mathbf{P}_\alpha(t) = \sum_{i\in\alpha}\mathbf{r}_i(t),
$$

the sum of the unwrapped positions of every ion of that type. Then correlate the
displacement of one species against another:

$$
L_{\alpha\beta}(\tau) = \big\langle
\Delta\mathbf{P}_\alpha(\tau)\cdot\Delta\mathbf{P}_\beta(\tau)
\big\rangle_{t},
\qquad
\Delta\mathbf{P}_\alpha(\tau) = \mathbf{P}_\alpha(t+\tau)-\mathbf{P}_\alpha(t).
$$

The long-time slope gives the **Onsager transport coefficients**:

$$
\boxed{\;\Omega_{\alpha\beta} = \lim_{\tau\to\infty}
\frac{L_{\alpha\beta}(\tau)}{6\,V k_B T\,\tau}\;}
$$

Read the two kinds of entry differently.

**Diagonal** $\Omega_{\alpha\alpha}$ is the collective displacement of one
species. It is *not* the self-diffusion coefficient: it also contains the
correlations between ions of the same type. If cations tended to move as a
group, $\Omega_{++}$ would exceed $N_+D_+$; if they obstructed each other, it
would fall below.

**Off-diagonal** $\Omega_{+-}$ is the interesting one. It measures whether
cations and anions move *together*. A positive value means they drift in the
same direction — ion pairing — and since they carry opposite charge, that shared
motion carries no current. Off-diagonal terms are the microscopic reason
conductivity falls short of Nernst–Einstein.

It all assembles into the conductivity:

$$
\sigma = \frac{e^{2}}{V k_B T}\sum_{\alpha\beta}
z_\alpha z_\beta\,\Omega_{\alpha\beta},
$$

and dropping the off-diagonal terms recovers $\sigma_\mathrm{NE}$ exactly. So the
ionicity $\sigma/\sigma_\mathrm{NE}$ is not a fudge factor — it is a statement
about $\Omega_{+-}$.

## Computing it

`OnsagerCorrelation` is configured with the longest lag, in frames, and
computes on two collective coordinate series `a` and `b`, each
`(n_frames, 3)`, built from **unwrapped** positions. It returns
$L_{\alpha\beta}(\tau)$ as a `TimeCorrelation` with one channel.

The $\mathbf{P}_\alpha(t)$ themselves are a `DipoleMoment` in disguise: with
every "charge" set to 1, $\sum_i q_i\mathbf{r}_i$ is the plain position sum,
and a partition — one unsigned species label per atom — splits it by species.
`series(k)` is the `(n_frames, 3)` series of the `k`-th label in `groups`:

```python
import numpy as np
from molpy.compute import DipoleMoment

rng = np.random.default_rng(0)
species = np.array([0, 0, 1, 1], dtype=np.uint32)          # two cations, two anions
unwrapped = np.cumsum(rng.normal(0.0, 0.05, size=(400, 4, 3)), axis=0)

sums = DipoleMoment().compute(
    positions=unwrapped, charges=np.ones(4), partition=species
)
p_plus, p_minus = sums.series(0), sums.series(1)
print(sums.groups, p_plus.shape)                           # -> [0 1] (400, 3)
```

To see that the correlation measures what it claims, build a cation walk and
an anion walk that deliberately shares 60 % of the cation's steps:

```python
from molpy.compute import LinearFit, OnsagerCorrelation

rng = np.random.default_rng(0)
steps = rng.normal(0.0, 0.01, size=(400, 3))
p_cation = np.cumsum(steps, axis=0)
p_anion = np.cumsum(0.6 * steps + 0.8 * rng.normal(0.0, 0.01, size=(400, 3)), axis=0)

onsager = OnsagerCorrelation(max_lag=100)
l_same = onsager.compute(a=p_cation, b=p_cation, dt=10.0)
l_cross = onsager.compute(a=p_cation, b=p_anion, dt=10.0)
print(l_same.n_lags, l_same.n_channels)                    # -> 101 1
```

One configured instrument, two calls: the configuration is fixed, only the
data changes. The ratio of the two slopes should recover that 0.6 coupling, and
the slopes are a `LinearFit` of each curve:

```python
window = LinearFit(0.1, 0.5)
same = window.compute(correlation=l_same, channel=0)
cross = window.compute(correlation=l_cross, channel=0)
print(round(cross.slope / same.slope, 2))                  # -> 0.72
```

0.72 against a true value of 0.60 — the sign and magnitude are right, and the
20 % error is the most useful thing on this page. There is exactly **one**
$\mathbf{P}_\alpha$ per frame, so a 400-frame run gives one collective random
walk, not 400 particle walks. Off-diagonal coefficients are differences between
comparably sized noisy numbers, and they converge slowly. Budget accordingly.

The control confirms there is no artificial coupling in the estimator itself:
replace `p_anion` with an independent walk and the ratio drops to 0.024.

A diagonal element is not a separate quantity. With `a` and `b` the same,
$L_{\alpha\alpha}$ is the mean squared displacement of $\mathbf{P}_\alpha$ —
the [Msd](msd.md) of a one-item trajectory, number for number:

```python
from molpy.compute import Msd

as_msd = Msd(max_lag=100).compute(positions=p_cation[:, None, :], dt=10.0)
print(np.allclose(as_msd.values, l_same.values))           # -> True
```

As always the compute returns the raw $L_{\alpha\beta}(\tau)$ curve; the slope
is yours to fit, over a window you choose and report. `stride` thins the time
origins when the trajectory is long. `boxes` takes the per-frame boxes and
refuses a wrapped input, but only for real positions — a sum of positions is
not a point in the box, so leave it out for $\mathbf{P}_\alpha$ and unwrap
the atoms (`Unwrap`) before summing.

!!! note "No figure on this page yet — TODO"
    The figure worth having is the set of $L_{\alpha\beta}(\tau)$ curves for a
    real electrolyte, with a visibly non-zero off-diagonal term. That needs a
    charged multi-species trajectory, which the argon reference system is not.
    Synthetic curves would only display the coupling that was typed into them —
    as the 0.6 above does. This page gets a figure when a charged reference
    trajectory exists under `scripts/docs_series/`.

## When it goes wrong

**Off-diagonal terms come out larger than diagonal ones.**
Usually unwrapping: a jump in one species' collective coordinate correlates with
everything else.

**$\Omega$ values are enormous.**
$\mathbf{P}_\alpha$ is a *sum* over $N_\alpha$ ions, not an average, so its
magnitude grows with system size. That is intended and the $V$ in the
denominator compensates — but never compare raw $L$ values between systems of
different size.

**The correlation curve is mostly noise.**
The worst statistics in this section, for the reason given above. Long
trajectories and honest error bars.

**$\sigma$ from the Onsager matrix disagrees with [PMSD](pmsd.md).**
They are algebraically the same quantity — $\mathbf{M} = e\sum_\alpha z_\alpha
\mathbf{P}_\alpha$ for point ions — so a disagreement means different fitting
windows, or a species missing from the sum.

**Everything correlates with everything.**
Centre-of-mass drift appears in *every* $\mathbf{P}_\alpha$ and therefore in
every off-diagonal term as a spurious positive correlation. Remove it first.

## Check yourself

- Build two independent random walks and confirm the cross term is near zero
  while the diagonals are not. (Above: 0.024 against 1.)
- Reconstruct $\mathbf{M} = e\sum_\alpha z_\alpha \mathbf{P}_\alpha$ and check
  that the [Einstein conductivity](pmsd.md) from it matches what you assemble
  from the $\Omega$ matrix.
- Compute $\sigma/\sigma_\mathrm{NE}$ for your system. If it exceeds 1, look
  hard at the sign of your off-diagonal terms.

## References

- L. Onsager, *Phys. Rev.* **37**, 405 (1931) — the reciprocal relations.
- K. D. Fong et al., *Macromolecules* **53**, 9503 (2020) — Onsager coefficients
  for polymer electrolytes, and why self-diffusion is not enough.
- N. Molinari, J. P. Mailoa, B. Kozinsky, *Chem. Mater.* **31**, 8748 (2019).

## See also

- [PMSD](pmsd.md) — the total conductivity these coefficients decompose
- [JACF](jacf.md) · [MSD](msd.md) · [Persist](persist.md)
- [API reference](../api/compute.md)
