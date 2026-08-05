# MSD

Textbook guide to the **mean-squared displacement** and the Einstein route to
self-diffusion.

!!! note "Conventions"
    - Time in **fs** (LAMMPS *real*); length Å.
    - Displacement kernels need **unwrapped** coordinates.
    - $d=3$; Einstein factor $1/(2d)=1/6$.

---

## 1. Random walk and Einstein relation

$$
\mathrm{MSD}(\tau)
= \big\langle |\mathbf{r}_i(t+\tau)-\mathbf{r}_i(t)|^2 \big\rangle_{i,t}.
$$

$$
\boxed{D = \lim_{\tau\to\infty}
\frac{1}{6\tau}\,\mathrm{MSD}(\tau)}
$$

Regimes: **ballistic** $\propto\tau^2$ → **diffusive** $\propto\tau$ (fit here)
→ **noisy** long lag. Use `MSD(method="window")` to average every time origin.

Periodic images must be unwrapped **before** MSD. Crossing a box face by $L$ is
a continuous path, not a jump.

<figure id="fig-msd" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:9"
mark:
  type: line
  strokeWidth: 2.2
  interpolate: monotone
data:
  values:
    - {t: 0.1, msd: 0.01}
    - {t: 1.0, msd: 1.0}
    - {t: 10.0, msd: 10.0}
    - {t: 100.0, msd: 95.0}
encoding:
  x: {field: t, type: quantitative, scale: {type: log}, title: "lag τ"}
  y: {field: msd, type: quantitative, scale: {type: log}, title: MSD}
  color: {value: "#0284c7"}
annotations:
  - kind: scaleBar
    x: 20
    y: 0.03
    length: 30
    label: "Δτ"
    color: "#18432b"
    tick: 0.008
  - kind: arrow
    x: 0.35
    y: 0.05
    x2: 1.2
    y2: 0.8
    label: "ballistic"
    color: "#64748b"
  - kind: arrow
    x: 3
    y: 2
    x2: 12
    y2: 12
    label: "diffusive"
    color: "#18432b"
```

</div>

**Figure 1.** Schematic MSD: ballistic, linear diffusive window, noisy long lag.
</figure>

---

## 2. Usage

```python
import numpy as np
import molpy as mp
from molpy.compute import MSD, LinearFit

rng = np.random.default_rng(0)
frames = []
for step in range(20):
    xyz = rng.uniform(0.0, 10.0, size=(40, 3)) + 0.05 * step
    f = mp.Frame()
    f["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    f.box = mp.Box.cubic(10.0)
    frames.append(f)

series = MSD(method="window")(frames)
# D = slope / 6 after LinearFit on the linear window of series.mean vs lag
```

Green–Kubo $D$ from velocities: [VACF](vacf.md). Collective coupling:
[Onsager](onsager.md).

---

## 3. Pitfalls

1. Wrapped coordinates → MSD saturates at the box size.
2. Fitting the ballistic or noisy regime.
3. Thermostat-heavy sampling for comparison with NVE [VACF](vacf.md).

## See also

- [VACF](vacf.md) · [Onsager](onsager.md) · [PMSD](pmsd.md) · [Van Hove](van_hove.md)
- [API reference](../api/compute.md)
