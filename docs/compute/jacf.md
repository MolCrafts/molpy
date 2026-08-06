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

Assemble $\mathbf{J}(t)$ yourself, shape `(n_frames, 3)`:

```python
import numpy as np
from molpy.compute import GreenKuboConductivity, CumulativeTrapezoid

rng = np.random.default_rng(2)
J = np.ascontiguousarray(rng.normal(0.0, 1.0, size=(400, 3)))

raw = GreenKuboConductivity().compute(J, dt=10.0, max_correlation_time=100)
print(sorted(raw), raw["jacf"].shape)     # -> ['jacf', 'lag_times'] (101,)
```

In real work:

```python
# docs: skip — needs your own trajectory with velocities
J = np.array([(charges[:, None] * velocities[i]).sum(axis=0)
              for i in range(len(velocities))])
```

The compute gives you $\langle\mathbf{J}(0)\cdot\mathbf{J}(t)\rangle$ and stops.
Integrating is a separate step, and `CumulativeTrapezoid` deliberately returns
the **running** integral rather than one number, so you can see whether it has
converged:

```python
running = np.asarray(CumulativeTrapezoid().fit(raw["jacf"], dt=10.0)["integral"])
print(running.shape)                      # -> (101,)
```

Quote the plateau of that curve, not its last point. The [VACF](vacf.md) page
shows the same procedure with a real curve, where reading the integral 400 fs
too early gives an answer 25 % too high; exactly the same trap applies here.

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

Do not apply that to the `running` array above and believe the result. `J` there
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
    a charged reference trajectory in `scripts/docs_data/`.

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
