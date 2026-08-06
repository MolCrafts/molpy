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

`Onsager.correlation` is a static method taking two collective coordinate
arrays, each `(n_frames, 3)`, built from **unwrapped** positions. Pass the same
array twice for a diagonal element.

To see that it measures what it claims, build a cation walk and an anion walk
that deliberately shares 60 % of the cation's steps:

```python
import numpy as np
from molpy.compute import Onsager, LinearFit

rng = np.random.default_rng(0)
steps = rng.normal(0.0, 0.01, size=(400, 3))
p_cation = np.ascontiguousarray(np.cumsum(steps, axis=0))
p_anion = np.ascontiguousarray(
    np.cumsum(0.6 * steps + 0.8 * rng.normal(0.0, 0.01, size=(400, 3)), axis=0)
)

l_same = Onsager.correlation(p_cation, p_cation, dt=10.0, max_correlation_time=100)
l_cross = Onsager.correlation(p_cation, p_anion, dt=10.0, max_correlation_time=100)
print(sorted(l_same))              # -> ['correlation', 'lag_times']
```

The ratio of the two slopes should recover that 0.6 coupling:

```python
same = LinearFit(0.1, 0.5).fit(l_same["lag_times"], l_same["correlation"])
cross = LinearFit(0.1, 0.5).fit(l_cross["lag_times"], l_cross["correlation"])
print(round(cross["slope"] / same["slope"], 2))     # -> 0.72
```

0.72 against a true value of 0.60 — the sign and magnitude are right, and the
20 % error is the most useful thing on this page. There is exactly **one**
$\mathbf{P}_\alpha$ per frame, so a 400-frame run gives one collective random
walk, not 400 particle walks. Off-diagonal coefficients are differences between
comparably sized noisy numbers, and they converge slowly. Budget accordingly.

The control confirms there is no artificial coupling in the estimator itself:
replace `p_anion` with an independent walk and the ratio drops to 0.024.

As always the compute returns the raw $L_{\alpha\beta}(\tau)$ curve; the slope
is yours to fit, over a window you choose and report.

!!! note "No figure on this page yet — TODO"
    The figure worth having is the set of $L_{\alpha\beta}(\tau)$ curves for a
    real electrolyte, with a visibly non-zero off-diagonal term. That needs a
    charged multi-species trajectory, which the argon reference system is not.
    Synthetic curves would only display the coupling that was typed into them —
    as the 0.6 above does. This page gets a figure when a charged reference
    trajectory exists under `scripts/docs_data/`.

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
