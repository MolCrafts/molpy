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
# Times New Roman + math-serif stack (LaTeX-like axis type).
config:
  font: "Times New Roman, Times, STIX Two Text, STIXGeneral, Latin Modern Roman, serif"
  axis:
    titleFont: "Times New Roman, Times, STIX Two Text, STIXGeneral, Latin Modern Roman, serif"
    labelFont: "Times New Roman, Times, STIX Two Text, STIXGeneral, Latin Modern Roman, serif"
    titleFontStyle: italic
    titleFontSize: 15
    labelFontSize: 12
    titlePadding: 10
    tickCount: 6
    gridOpacity: 0.35
  text:
    font: "Times New Roman, Times, STIX Two Text, STIXGeneral, Latin Modern Roman, serif"
    fontSize: 12
# Shared log domains so the |———| bars and the curve share axes.
encoding:
  x:
    type: quantitative
    scale: {type: log, domain: [0.1, 100]}
    title: "lag τ"
    axis: {titleFontStyle: italic}
  y:
    type: quantitative
    scale: {type: log, domain: [0.008, 120]}
    title: "MSD"
    axis: {titleFontStyle: italic}
layer:
  # —— curve: ballistic (∝τ²) → linear diffusive (∝τ) → noisy long lag ——
  - data:
      values:
        - {t: 0.10, msd: 0.010}
        - {t: 0.20, msd: 0.040}
        - {t: 0.40, msd: 0.16}
        - {t: 0.80, msd: 0.64}
        - {t: 1.20, msd: 1.15}
        - {t: 2.00, msd: 2.0}
        - {t: 4.00, msd: 4.0}
        - {t: 8.00, msd: 8.0}
        - {t: 15.0, msd: 14.5}
        - {t: 30.0, msd: 26}
        - {t: 55.0, msd: 42}
        - {t: 100.0, msd: 70}
    mark: {type: line, strokeWidth: 2.4, interpolate: monotone, color: "#0c5da5"}
    encoding:
      x: {field: t}
      y: {field: msd}

  # —— |———| region bars (bottom of plot) ——
  - data:
      values:
        - {t: 0.12, t2: 0.85, msd: 0.012}   # ballistic
        - {t: 1.5,  t2: 14,   msd: 0.012}   # linear diffusive
        - {t: 22,   t2: 90,   msd: 0.012}   # noisy long lag
    mark: {type: rule, strokeWidth: 1.5, color: "#18432b", strokeCap: butt}
    encoding:
      x: {field: t}
      x2: {field: t2}
      y: {field: msd}

  # end-caps |   |  (geometric half-height on log-y)
  - data:
      values:
        - {t: 0.12, msd: 0.0095, msd2: 0.015}
        - {t: 0.85, msd: 0.0095, msd2: 0.015}
        - {t: 1.5,  msd: 0.0095, msd2: 0.015}
        - {t: 14,   msd: 0.0095, msd2: 0.015}
        - {t: 22,   msd: 0.0095, msd2: 0.015}
        - {t: 90,   msd: 0.0095, msd2: 0.015}
    mark: {type: rule, strokeWidth: 1.5, color: "#18432b"}
    encoding:
      x: {field: t}
      y: {field: msd}
      y2: {field: msd2}

  # region names under each |———|
  - data:
      values:
        - {t: 0.32, msd: 0.012, label: "ballistic"}
        - {t: 4.6,  msd: 0.012, label: "diffusive"}
        - {t: 45,   msd: 0.012, label: "noisy"}
    mark:
      type: text
      dy: 14
      fontSize: 12
      font: "Times New Roman, Times, STIX Two Text, STIXGeneral, serif"
      color: "#18432b"
      align: center
      baseline: top
      fontStyle: italic
    encoding:
      x: {field: t}
      y: {field: msd}
      text: {field: label, type: nominal}
```

</div>

**Figure 1.** Schematic MSD with regimes marked by scale bars: ballistic $\propto\tau^{2}$, linear diffusive window (fit here), noisy long lag.
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
