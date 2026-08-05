# Van Hove

This page is a self-contained, textbook-style introduction to two time-resolved
correlation functions: the **Van Hove function** $G(r,t)$ — the time-dependent
generalization of the radial distribution function — and the **Legendre
reorientational correlations** $C_1(t)$, $C_2(t)$ that quantify how fast
molecular vectors lose their orientation. Together they bridge the static
structure of the [structural guide](rdf.md) and the transport coefficients
of the [transport guide](msd.md): they show *how structure decorrelates in
time*.

Correlation kernels run in the high-performance backend; the MolPy layer unwraps
trajectories where needed and returns a typed result.

!!! note "Conventions used throughout"
    - Distances in Å; time lags in **frames** (multiply by the dump interval for
      ps or fs).
    - $\langle\cdots\rangle$ averages over particles and time origins.
    - Self Van Hove needs **unwrapped** single-particle trajectories.

---

## 1. The Van Hove function is $g(r)$ resolved in time

Van Hove (1954) asked: given a particle at the origin at time $0$, what is the
probability density of finding a particle (the same one, or another) at distance
$r$ after time $t$? The answer splits into **self** and **distinct** parts:

$$
G(r,t)
= \underbrace{
    \frac{1}{N}\Big\langle
      \sum_i \delta\big(r - |\mathbf{r}_i(t)-\mathbf{r}_i(0)|\big)
    \Big\rangle
  }_{G_s(r,t)\ \text{(same particle)}
+ \underbrace{
    \frac{1}{N}\Big\langle
      \sum_{i\neq j}\delta\big(r - |\mathbf{r}_i(t)-\mathbf{r}_j(0)|\big)
    \Big\rangle
  }_{G_d(r,t)\ \text{(distinct particles)}.
$$

### 1.1 Self-part $G_s(r,t)$

$G_s$ is the distribution of single-particle displacements:

- **$t\to 0$**: a sharp peak at $r=0$ (particles have not moved).
- **Ballistic regime**: peak shifts as $\langle r\rangle\sim\langle v\rangle t$.
- **Cage regime** (dense liquids): probability piles up near the first-neighbour
  distance while the tagged particle rattles in its cage.
- **Diffusive regime**: $G_s$ broadens into a Gaussian whose second moment is the
  [MSD](msd.md).

Non-Gaussian shapes (shoulders, secondary peaks) flag hopping, dynamic
heterogeneity, or jump diffusion — central diagnostics in glass physics.

### 1.2 Distinct-part $G_d(r,t)$

At zero lag the distinct part recovers the RDF:

$$
\boxed{\;G_d(r,0)=\rho\,g(r)\;}
$$

As $t$ grows, the coordination shells wash out: neighbours leave and are
replaced. The time for the first peak of $G_d$ to decay is a structural
relaxation time complementary to density-density correlations in $k$-space.

### 1.3 Moments, MSD, and the non-Gaussian parameter

The second moment of the self-part **is** the mean-squared displacement:

$$
\big\langle r^2(t)\big\rangle
= 4\pi\int_0^\infty r^4\, G_s(r,t)\,\mathrm{d}r
= \mathrm{MSD}(t).
$$

(In 3-D with the radial measure $4\pi r^2\,\mathrm{d}r$ on the probability
density convention used here; implementations store a histogram consistent with
their binning.) The non-Gaussian parameter

$$
\alpha_2(t)
= \frac{3}{5}\frac{\langle r^4(t)\rangle}{\langle r^2(t)\rangle^2}-1
$$

vanishes for pure Fickian (Gaussian) diffusion and **peaks** when particles hop
between cages — a standard glass-physics diagnostic.

### 1.4 Intermediate scattering function (connection)

Scattering experiments measure the **intermediate scattering function**

$$
F_s(k,t)
= \big\langle e^{-i\mathbf{k}\cdot[\mathbf{r}(t)-\mathbf{r}(0)]}\big\rangle,
$$

which is the Fourier transform of $G_s(r,t)$. The same physics appears as a peak
broadening in $G_s$ or as a decay of $F_s(k,t)$ at the structure-factor peak
wavenumber $k^*$. MolPy’s Van Hove API works in $r$-space; use it when real-space
shells and hopping shoulders matter.

<figure id="fig-vanhove" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:9"
mark:
  type: line
  strokeWidth: 2.2
  interpolate: monotone
data:
  values:
    - {r: 0.2, Gs: 3.0}
    - {r: 0.5, Gs: 2.2}
    - {r: 1.0, Gs: 1.0}
    - {r: 1.5, Gs: 0.4}
    - {r: 2.0, Gs: 0.15}
    - {r: 3.0, Gs: 0.05}
    - {r: 5.0, Gs: 0.01}
encoding:
  x:
    field: r
    type: quantitative
    title: r (Å)
  y:
    field: Gs
    type: quantitative
    scale: {zero: false}
    title: G_s(r,t)
  color:
    value: "#0284c7"
```

</div>

**Figure 1.** Schematic self Van Hove $G_s(r,t)$ at fixed lag: a peak that broadens and shifts outward as $t$ increases (Fickian limit is Gaussian).
</figure>

---

## 2. Computing the Van Hove function

```python
import numpy as np
import molpy as mp

def _frame(step: int) -> mp.Frame:
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0.0, 20.0, size=(200, 3)) + 0.1 * step
    frame = mp.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.box = mp.Box.cubic(20.0)
    return frame

frames = [_frame(step) for step in range(20)]
```

```python
from molpy.compute import VanHove

vh = VanHove(n_rbins=200, r_max=15.0, lags=[1, 5, 10, 50, 100])
result = vh(frames)

result.r_centers   # radial grid, Å
result.lags        # time lags (frames)
result.g_self      # G_s(r, t): rows = lags, columns = radial bins
result.g_distinct  # G_d(r, t) when result.has_distinct
```

Choose `lags` to straddle the dynamics of interest — short lags for
ballistic/caging, longer ones for the diffusive broadening. Keep
`r_max ≤ L/2` for a clean distinct part under periodic boundaries.

---

## 3. Reorientation: how fast vectors forget their direction

### 3.1 Legendre correlations

For a unit vector $\mathbf{u}(t)$ rigidly attached to a molecule (bond, dipole,
symmetry axis), the **Legendre reorientational correlation functions** are

$$
\boxed{\;
C_\ell(t)
= \big\langle P_\ell\big(\mathbf{u}(0)\cdot\mathbf{u}(t)\big)\big\rangle,
\qquad
P_1(x)=x,\quad
P_2(x)=\tfrac12(3x^2-1)
\;}
$$

Both decay from $1$ (perfect memory) toward $0$ (fully randomized orientation).
Different experiments probe different $\ell$:

| Experiment / response | Order |
|---|---|
| Dielectric relaxation, IR | $C_1$ |
| NMR spin relaxation, fluorescence anisotropy, Raman | $C_2$ |

### 3.2 Correlation times

Fit the long-time exponential tail (not the librational head):

$$
C_\ell(t)\approx e^{-t/\tau_\ell}
\quad\Rightarrow\quad
\tau_\ell
= \int_0^\infty C_\ell(t)\,\mathrm{d}t
\quad\text{(or from the fit)}.
$$

In the **Debye rotational diffusion** limit,

$$
\tau_\ell = \frac{1}{\ell(\ell+1)D_R}
\quad\Rightarrow\quad
\frac{\tau_1}{\tau_2}=3.
$$

Large deviations of $\tau_1/\tau_2$ from 3 signal jump reorientation (e.g. water’s
large-amplitude H-bond exchanges) rather than small-step diffusion.

### 3.3 Link to spectra and dielectrics

- Collective $C_1$ of the total dipole underlies
  [dielectric spectroscopy](dielectric.md).
- Single-molecule $C_1$/$C_2$ set vibrational lineshape envelopes and NMR
  correlation times — complementary to the velocity-based
  [VDOS / IR](spectra.md) route.

---

## 4. Computing reorientational correlations

```python
import numpy as np
from molpy.compute import LegendreReorientation

# Bond endpoints come from the frame's `bonds` topology block.
for f in frames:
    f["bonds"] = {"atomi": np.array([0, 0]), "atomj": np.array([1, 2])}

reor = LegendreReorientation(max_lag=5)
result = reor(frames)

result.lags  # lags (frames)
result.c1    # C_1(t)
result.c2    # C_2(t)
```

---

## 5. Pitfalls checklist

1. **`r_max` beyond half the box** → distinct part corrupted by periodic images.
2. **Lags longer than the trajectory supports** → few time origins; noisy tails.
3. **Reading $\tau$ from a non-exponential head** → fit the long-time tail of
   $C_\ell$, not the librational sub-picosecond decay.
4. **Degenerate vectors** → identical head/tail atoms make $\mathbf{u}$ undefined.
5. **Wrapped coordinates for $G_s$** → self-part saturates at the box size;
   unwrap single-particle trajectories.
6. **Comparing $\tau_1$ and $\tau_2$** → ensure both use the same vector definition
   and fit window before quoting $\tau_1/\tau_2$.

---

## 6. References

- L. Van Hove, *Phys. Rev.* **95**, 249 (1954) — $G(r,t)$.
- B. J. Berne, R. Pecora, *Dynamic Light Scattering*, Wiley (1976) —
  reorientational correlations and the $C_1$/$C_2$ distinction.
- W. Kob, H. C. Andersen, *Phys. Rev. E* **51**, 4626 (1995) — non-Gaussian
  parameter and dynamical heterogeneity.
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed., ch. 7–8.
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — AIMD analysis stack.

## See also

- [Diffusion & Ionic Transport](msd.md) — MSD is the second moment of $G_s$.
- [Structural Analysis](rdf.md) — $G_d(r,0)=\rho\,g(r)$.
- [Dielectric](dielectric.md) — $C_1$ and collective dipoles.
- [Vibrational Spectra](spectra.md) — lineshapes and reorientation.
- [API reference: Compute](../api/compute.md).
