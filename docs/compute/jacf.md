# Green–Kubo conductivity

[PMSD](pmsd.md) got the conductivity by watching charge *displacement* spread
out over time. This page gets the same number by watching the charge *current*
forget itself — the same pairing as [MSD](msd.md) and [VACF](vacf.md), one level
up in collectivity.

## The charge current and its memory

The current is what the ions are doing right now, weighted by charge:

$$
\mathbf{J}(t) = \sum_a q_a \mathbf{v}_a(t).
$$

Like the collective displacement $\mathbf{M}$, it is built so that neutral
co-moving pairs cancel: a cation and anion travelling together contribute
$q\mathbf{v} + (-q)\mathbf{v} = 0$. And in fact $\mathbf{J} = \dot{\mathbf{M}}$,
which is why the two routes must agree.

In equilibrium $\langle\mathbf{J}\rangle = 0$ — no net current without a field.
What is not zero is the correlation of the current with its own past, and
linear-response theory says that correlation is the conductivity:

$$
\boxed{\;\sigma = \frac{1}{3\,V k_B T}
\int_0^{\infty}\big\langle \mathbf{J}(0)\cdot\mathbf{J}(t)\big\rangle\,
\mathrm{d}t\;}
$$

The factor is $1/3$, not $1/6$. The $1/6$ in the Einstein route comes from
differentiating a mean *squared* displacement in three dimensions; here you
integrate a correlation function directly, and only the dimensionality survives.
Mixing the two up gives an answer wrong by a factor of two, and it is a common
error.

## Which route to use

Both are exact and both are estimates of the same number, so the choice is
practical.

| | Einstein ([PMSD](pmsd.md)) | Green–Kubo (this page) |
|---|---|---|
| Input | unwrapped **positions** | **velocities** |
| Sampling | tolerates coarse dumps | needs fine dumps (a few fs) |
| What you look for | linear region of a growing curve | plateau of a converging integral |
| Sensitive to | unwrapping errors | thermostat friction, drift |

Velocity correlations decay far faster than displacements become diffusive, so
Green–Kubo needs densely sampled frames but a shorter total window. If you have
both, compute both: agreement is the best evidence that either is right.

## Computing it

The Green–Kubo conductivity is a chain of three computes, each taking the
previous one's output: the current autocorrelation, its running integral, and
the plateau of that integral.

The current comes first, and only you can build it, because only you know
which atoms carry which charge. Two routes give the same $\mathbf{J}(t)$,
shape `(n_frames, 3)`. With velocities in the dump, it is a charge-weighted
sum, one line of NumPy:

```python
# docs: skip — needs your own trajectory with velocities
J = np.einsum("a,tad->td", charges, velocities)    # (T, N, 3) -> (T, 3)
```

With positions only, it is the time derivative of the collective dipole,
$\mathbf{J} = \dot{\mathbf{M}}$: a `DipoleMoment` followed by a `Derivative`.
The stencil is configuration — `"central"` drops one frame at each end,
`"gradient"` keeps every frame with one-sided ends, `"backward"` drops the
first:

```python
import numpy as np
from molpy.compute import Derivative, DipoleMoment

rng = np.random.default_rng(2)
charges = np.array([1.0, -1.0, 1.0, -1.0])
unwrapped = np.cumsum(rng.normal(0.0, 0.05, size=(400, 4, 3)), axis=0)

M = DipoleMoment().compute(positions=unwrapped, charges=charges).series(0)
J_from_M = Derivative("central").compute(series=M, dt=10.0)
print(M.shape, J_from_M.shape)            # -> (400, 3) (398, 3)
```

`CurrentAcf` correlates the current with its own past, summed over the three
Cartesian components — exactly $\langle\mathbf{J}(0)\cdot\mathbf{J}(t)\rangle$,
with nothing averaged away. It returns a `TimeCorrelation` and stops. To keep
the numbers predictable, the current here is white noise of unit width per
component, so the zero-lag value is 3:

```python
from molpy.compute import CurrentAcf

J = np.random.default_rng(2).normal(0.0, 1.0, size=(400, 3))
raw = CurrentAcf(max_lag=100).compute(current=J, dt=10.0)
print(raw.n_lags, round(float(raw.values[0, 0]), 2))   # -> 101 3.05
```

Integrating is a separate step. `CumulativeTrapezoid` deliberately returns the
**running** integral as another `TimeCorrelation`, rather than one number, so
you can see whether it has converged, and `Plateau` averages it over a window
given as fractions of the lag axis:

```python
from molpy.compute import CumulativeTrapezoid, Plateau

running = CumulativeTrapezoid().compute(correlation=raw, channel=0)
plateau = Plateau(0.5, 1.0).compute(correlation=running, channel=0)
print(running.n_lags)                     # -> 101
integral = plateau.value                  # e² Å² / fs
```

Quote the plateau, not the last point, and look at `plateau.std` before you
believe it. The [VACF](vacf.md) page shows the same procedure with a real
curve, where reading the integral at 400 fs, before the negative lobe, gives
an answer 38 % too high; exactly the same trap applies here.

Each lag is divided by the number of time origins that reach it
(`normalization="unbiased"`, the default). `"biased"` divides by the number of
frames instead, which damps the noisy tail at the cost of a bias; `"none"`
returns the raw sums, the convention some older codes integrate.

### Getting to S/m

With charges in $e$ and velocities in Å fs⁻¹, the integral is in
$e^2\,\text{Å}^2\,\text{fs}^{-1}$ and

$$
\sigma\ [\mathrm{S\,m^{-1}}] = 6.1975\times10^{9}\;
\frac{\int_0^{\tau}\!\langle \mathbf J(0)\!\cdot\!\mathbf J(t)\rangle\mathrm{d}t}
{V\ [\text{Å}^{3}]\times T\ [\mathrm{K}]}
$$

```python
def conductivity_si(integral, volume_a3, temperature_k):
    """Green-Kubo conductivity in S/m from the plateau of the running integral."""
    return 6.1975e9 * integral / (volume_a3 * temperature_k)
```

Do not apply that to the `plateau` above and believe the result. `J` there
is white noise of arbitrary magnitude, not a physical current, so the number it
produces is arbitrary too. The helper is correct; the input is a placeholder.

Note this constant is exactly twice the Einstein one on the [PMSD](pmsd.md)
page, because of the $1/3$ against $1/6$. The $1/3$ is already inside it.

!!! note "No figure on this page yet — TODO"
    The current autocorrelation of a real electrolyte is worth plotting — it
    oscillates and changes sign, and the running integral's plateau is the whole
    lesson. Both need a charged trajectory, and the argon reference system
    behind these pages has $\mathbf{J} \equiv 0$. Rather than plot filtered
    noise, this page defers to [VACF](vacf.md), which shows the identical
    "integrate to the plateau" figure with real data. A dedicated figure follows
    a charged reference trajectory in `molpy/scripts/docs_series/`.

## When it goes wrong

**The running integral never flattens; it drifts linearly.**
A non-decaying component in $\mathbf{J}$ — usually net momentum or a
centre-of-mass drift. Remove it before building the current.

**$\sigma$ is much smaller than the Einstein estimate.**
Look at the JACF near $t=0$. If the first decay is resolved by only two or three
points, you are integrating a triangle instead of a curve. Dump velocities more
often.

**$\sigma$ is exactly half or double the Einstein answer.**
The $1/3$ against $1/6$ factor.

**The JACF is smooth and featureless and $\sigma$ is low.**
A strong thermostat is damping the dynamics. Sample in NVE.

**The result changes every time you rerun with a different seed.**
Collective quantities converge slowly. One collective vector per frame is a much
weaker statistical sample than $N$ particle velocities.

## Check yourself

- Verify $\langle\mathbf{J}\rangle \approx 0$ over your run. A non-zero mean
  current in equilibrium means drift.
- Compare against the [Einstein route](pmsd.md) on the same trajectory. They
  should agree within their error bars; if they do not, at least one window is
  wrong.
- Halve your velocity dump frequency and recompute. If $\sigma$ moves, you were
  under-sampling the initial decay.

## References

- M. S. Green, *J. Chem. Phys.* **22**, 398 (1954); R. Kubo, *J. Phys. Soc.
  Jpn.* **12**, 570 (1957) — the linear-response relations.
- D. Frenkel, B. Smit, *Understanding Molecular Simulation*, 2nd ed. (2002),
  §4.4 — practical Green–Kubo estimation and its error behaviour.

## See also

- [PMSD](pmsd.md) — the Einstein route to the same $\sigma$
- [VACF](vacf.md) — the same integrate-to-the-plateau procedure, with a figure
- [Onsager](onsager.md) — species-resolved correlations
- [Dielectric](dielectric.md) — $\sigma(\omega)$ rather than $\sigma$
- [API reference](../api/compute.md)
