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
# Matplotlib-style: |———| is a chord ON the curve (x,y)→(x2,y2), end-caps
# perpendicular in log–log space. Times New Roman, roman (not italic).
# Large type sizes so docs stay readable even before host fontScale updates.
config:
  font: "Times New Roman, Times, STIX Two Text, STIXGeneral, Latin Modern Roman, serif"
  axis:
    titleFont: "Times New Roman, Times, STIX Two Text, STIXGeneral, Latin Modern Roman, serif"
    labelFont: "Times New Roman, Times, STIX Two Text, STIXGeneral, Latin Modern Roman, serif"
    titleFontStyle: normal
    labelFontStyle: normal
    titleFontSize: 28
    labelFontSize: 24
    titlePadding: 14
    labelPadding: 8
    tickSize: 10
    tickCount: 6
    gridOpacity: 0.3
  text:
    font: "Times New Roman, Times, STIX Two Text, STIXGeneral, Latin Modern Roman, serif"
    fontStyle: normal
    fontSize: 22
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
  # schematic: ballistic ∝τ² → linear diffusive ∝τ → noisy long lag
  - data:
      values:
        - {t: 0.10, msd: 0.010}
        - {t: 0.15, msd: 0.0225}
        - {t: 0.20, msd: 0.040}
        - {t: 0.30, msd: 0.090}
        - {t: 0.45, msd: 0.2025}
        - {t: 0.65, msd: 0.4225}
        - {t: 0.90, msd: 0.81}
        - {t: 1.20, msd: 1.20}
        - {t: 1.80, msd: 1.80}
        - {t: 2.50, msd: 2.50}
        - {t: 4.00, msd: 4.00}
        - {t: 6.00, msd: 6.00}
        - {t: 9.00, msd: 9.00}
        - {t: 12.0, msd: 12.0}
        - {t: 18.0, msd: 18.0}
        - {t: 28.0, msd: 24.4}
        - {t: 40.0, msd: 31.0}
        - {t: 55.0, msd: 39.25}
        - {t: 75.0, msd: 50.25}
        - {t: 100,  msd: 64.0}
    mark: {type: line, strokeWidth: 3, interpolate: monotone, color: "#0c5da5"}
    encoding:
      x: {field: t}
      y: {field: msd}

  # |———| parallel to the local curve chord, offset in log–log space so the
  # bar does not sit on the line (matplotlib: translate along path normal).
  # End-caps are ⊥ to the bar in the same log–log metric.
  - data:
      values:
        # ballistic  (offset from (0.18,0.0324)→(0.65,0.4225))
        - {t: 0.1236, msd: 0.03909, t2: 0.4464, msd2: 0.5098}
        # diffusive  (offset from (2.2,2.2)→(12,12))
        - {t: 1.635,  msd: 2.961,   t2: 8.917,  msd2: 16.15}
        # noisy      (offset from (28,24.4)→(80,53))
        - {t: 21.82,  msd: 34.21,   t2: 62.33,  msd2: 74.3}
    mark: {type: rule, strokeWidth: 2.2, color: "#18432b", strokeCap: butt}
    encoding:
      x: {field: t}
      y: {field: msd}
      x2: {field: t2}
      y2: {field: msd2}

  # end-caps ⊥ bar (log–log)
  - data:
      values:
        - {t: 0.1071, msd: 0.04199, t2: 0.1427, msd2: 0.0364}
        - {t: 0.3869, msd: 0.5476,  t2: 0.5151, msd2: 0.4746}
        - {t: 1.46,   msd: 3.315,   t2: 1.831,  msd2: 2.644}
        - {t: 7.963,  msd: 18.08,   t2: 9.985,  msd2: 14.42}
        - {t: 19.84,  msd: 38.9,    t2: 23.99,  msd2: 30.07}
        - {t: 56.68,  msd: 84.5,    t2: 68.55,  msd2: 65.33}
    mark: {type: rule, strokeWidth: 2.2, color: "#18432b"}
    encoding:
      x: {field: t}
      y: {field: msd}
      x2: {field: t2}
      y2: {field: msd2}

  # labels further out along the same offset normal (Times roman)
  - data:
      values:
        - {t: 0.1765, msd: 0.1629, label: "ballistic"}
        - {t: 3.045,  msd: 8.671,  label: "diffusive"}
        - {t: 30.49,  msd: 65.21,  label: "noisy"}
    mark:
      type: text
      font: "Times New Roman, Times, STIX Two Text, STIXGeneral, serif"
      fontStyle: normal
      fontSize: 22
      color: "#18432b"
      align: center
      baseline: middle
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
