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
# Font family only — sizes come from molplot fontScaleForWidth (2× paper,
# tracks host width). Axis titles roman (not italic).
config:
  font: "Times New Roman, Times, STIX Two Text, STIXGeneral, Latin Modern Roman, serif"
  axis:
    titleFont: "Times New Roman, Times, STIX Two Text, STIXGeneral, Latin Modern Roman, serif"
    labelFont: "Times New Roman, Times, STIX Two Text, STIXGeneral, Latin Modern Roman, serif"
    titleFontStyle: normal
    labelFontStyle: normal
    tickCount: 6
    gridOpacity: 0.35
  text:
    font: "Times New Roman, Times, STIX Two Text, STIXGeneral, Latin Modern Roman, serif"
    fontStyle: normal
encoding:
  x:
    type: quantitative
    scale: {type: log, domain: [0.1, 100]}
    title: "lag τ"
    axis: {titleFontStyle: normal}
  y:
    type: quantitative
    scale: {type: log, domain: [0.008, 120]}
    title: "MSD"
    axis: {titleFontStyle: normal}
layer:
  # curve: ballistic (∝τ²) → linear diffusive (∝τ) → noisy long lag
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
    mark: {type: line, strokeWidth: 2.6, interpolate: monotone, color: "#0c5da5"}
    encoding:
      x: {field: t}
      y: {field: msd}

  # |———| along the curve: each bar at the local MSD height of that regime
  # (slightly above the line so the stroke stays readable).
  - data:
      values:
        # ballistic ~ τ²: bar rides the rising flank
        - {t: 0.15, t2: 0.70, msd: 0.22}
        # linear diffusive window
        - {t: 2.0,  t2: 12,   msd: 9.5}
        # noisy long lag
        - {t: 28,   t2: 85,   msd: 48}
    mark: {type: rule, strokeWidth: 1.8, color: "#18432b", strokeCap: butt}
    encoding:
      x: {field: t}
      x2: {field: t2}
      y: {field: msd}

  # end-caps |   |  (geometric half-height on log-y, local to each bar)
  - data:
      values:
        - {t: 0.15, msd: 0.16, msd2: 0.30}
        - {t: 0.70, msd: 0.16, msd2: 0.30}
        - {t: 2.0,  msd: 7.0,  msd2: 13}
        - {t: 12,   msd: 7.0,  msd2: 13}
        - {t: 28,   msd: 36,   msd2: 64}
        - {t: 85,   msd: 36,   msd2: 64}
    mark: {type: rule, strokeWidth: 1.8, color: "#18432b"}
    encoding:
      x: {field: t}
      y: {field: msd}
      y2: {field: msd2}

  # region names (Times New Roman, roman — not italic), next to each bar
  - data:
      values:
        - {t: 0.32, msd: 0.35, label: "ballistic"}
        - {t: 4.9,  msd: 14,   label: "diffusive"}
        - {t: 49,   msd: 70,   label: "noisy"}
    mark:
      type: text
      dy: -10
      font: "Times New Roman, Times, STIX Two Text, STIXGeneral, serif"
      fontStyle: normal
      color: "#18432b"
      align: center
      baseline: bottom
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
