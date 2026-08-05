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
# Pure Vega-Lite (layered). Formatter only YAML→JSON embeds this spec.
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
  # curve: ballistic ∝τ² → linear diffusive ∝τ → noisy long lag
  - data:
      values:
        - {x: 0.10, y: 0.010}
        - {x: 0.15, y: 0.0225}
        - {x: 0.20, y: 0.040}
        - {x: 0.30, y: 0.090}
        - {x: 0.45, y: 0.2025}
        - {x: 0.65, y: 0.4225}
        - {x: 0.90, y: 0.81}
        - {x: 1.20, y: 1.20}
        - {x: 1.80, y: 1.80}
        - {x: 2.50, y: 2.50}
        - {x: 4.00, y: 4.00}
        - {x: 6.00, y: 6.00}
        - {x: 9.00, y: 9.00}
        - {x: 12.0, y: 12.0}
        - {x: 18.0, y: 18.0}
        - {x: 28.0, y: 24.4}
        - {x: 40.0, y: 31.0}
        - {x: 55.0, y: 39.25}
        - {x: 75.0, y: 50.25}
        - {x: 100,  y: 64.0}
    mark: {type: line, strokeWidth: 3, interpolate: monotone, color: "#0c5da5"}
    encoding:
      x: {field: x}
      y: {field: y}

  # |———| spines: parallel to local chord, offset off the curve (data coords)
  - data:
      values:
        - {x: 0.12363, y: 0.039095, x2: 0.44644, y2: 0.5098}
        - {x: 1.6347,  y: 2.9607,   x2: 8.9167,  y2: 16.15}
        - {x: 21.815,  y: 34.205,   x2: 62.329,  y2: 74.298}
    mark: {type: rule, strokeWidth: 2.2, color: "#18432b", strokeCap: butt}
    encoding:
      x: {field: x}
      y: {field: y}
      x2: {field: x2}
      y2: {field: y2}

  # end-caps ⊥ to each spine (same stroke)
  - data:
      values:
        - {x: 0.10715, y: 0.041995, x2: 0.14265, y2: 0.036395}
        - {x: 0.38691, y: 0.54762,  x2: 0.51513, y2: 0.4746}
        - {x: 1.4599,  y: 3.3154,   x2: 1.8305,  y2: 2.644}
        - {x: 7.9628,  y: 18.084,   x2: 9.9847,  y2: 14.422}
        - {x: 19.837,  y: 38.902,   x2: 23.991,  y2: 30.075}
        - {x: 56.676,  y: 84.501,   x2: 68.547,  y2: 65.327}
    mark: {type: rule, strokeWidth: 2.2, color: "#18432b"}
    encoding:
      x: {field: x}
      y: {field: y}
      x2: {field: x2}
      y2: {field: y2}

  # regime labels (Times roman)
  - data:
      values:
        - {x: 0.17646, y: 0.1629, label: "ballistic"}
        - {x: 3.0448,  y: 8.6706, label: "diffusive"}
        - {x: 30.489,  y: 65.209, label: "noisy"}
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

---

## 3. Pitfalls

1. Wrapped coordinates → MSD saturates at the box size.
2. Fitting the ballistic or noisy regime.
3. Thermostat-heavy sampling for comparison with NVE [VACF](vacf.md).

## See also

- [VACF](vacf.md) · [Onsager](onsager.md) · [PMSD](pmsd.md) · [Van Hove](van_hove.md)
- [API reference](../api/compute.md)
