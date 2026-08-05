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

  # |———| chords ON the curve (matplotlib annotate-along-path style)
  - data:
      values:
        - {t: 0.18, msd: 0.0324, t2: 0.65, msd2: 0.4225}   # ballistic
        - {t: 2.2,  msd: 2.2,    t2: 12,   msd2: 12}        # diffusive
        - {t: 28,   msd: 24.4,   t2: 80,   msd2: 53}        # noisy
    mark: {type: rule, strokeWidth: 2.2, color: "#18432b", strokeCap: butt}
    encoding:
      x: {field: t}
      y: {field: msd}
      x2: {field: t2}
      y2: {field: msd2}

  # end-caps ⊥ segment in log–log space
  - data:
      values:
        - {t: 0.1532, msd: 0.03512, t2: 0.2114, msd2: 0.02989}
        - {t: 0.5533, msd: 0.4579,  t2: 0.7635, msd2: 0.3898}
        - {t: 1.937,  msd: 2.499,   t2: 2.499,  msd2: 1.937}
        - {t: 10.57,  msd: 13.63,   t2: 13.63,  msd2: 10.57}
        - {t: 25.16,  msd: 28.2,    t2: 31.16,  msd2: 21.11}
        - {t: 71.88,  msd: 61.26,   t2: 89.03,  msd2: 45.86}
    mark: {type: rule, strokeWidth: 2.2, color: "#18432b"}
    encoding:
      x: {field: t}
      y: {field: msd}
      x2: {field: t2}
      y2: {field: msd2}

  # labels (Times roman), offset above the chord midpoint
  - data:
      values:
        - {t: 0.342, msd: 0.181, label: "ballistic"}
        - {t: 5.14,  msd: 7.96,  label: "diffusive"}
        - {t: 47.3,  msd: 55.7,  label: "noisy"}
    mark:
      type: text
      font: "Times New Roman, Times, STIX Two Text, STIXGeneral, serif"
      fontStyle: normal
      fontSize: 22
      color: "#18432b"
      align: center
      baseline: bottom
      dy: -8
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
