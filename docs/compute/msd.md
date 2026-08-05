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
# Vega-Lite only. Data via $file (relative to docs/) — not inlined in the fence.
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
resolve:
  scale: {x: shared, y: shared}
layer:
  - data: {$file: data/msd/curve.json}
    mark: {type: line, strokeWidth: 3, interpolate: monotone, color: "#0c5da5"}
    encoding:
      x: {field: x}
      y: {field: y}
  - data: {$file: data/msd/spines.json}
    mark: {type: rule, strokeWidth: 2.2, color: "#18432b", strokeCap: butt}
    encoding:
      x: {field: x}
      y: {field: y}
      x2: {field: x2}
      y2: {field: y2}
  - data: {$file: data/msd/caps.json}
    mark: {type: rule, strokeWidth: 2.2, color: "#18432b"}
    encoding:
      x: {field: x}
      y: {field: y}
      x2: {field: x2}
      y2: {field: y2}
  - data: {$file: data/msd/labels.json}
    mark:
      type: text
      font: "Times New Roman, Times, STIX Two Text, STIXGeneral, serif"
      fontStyle: normal
      fontSize: 22
      color: "#18432b"
      align: center
      baseline: middle
    encoding:
      x: {field: x}
      y: {field: y}
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
