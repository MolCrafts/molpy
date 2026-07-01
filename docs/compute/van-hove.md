# Van Hove Correlations & Reorientational Dynamics

This page is a self-contained, textbook-style introduction to two time-resolved
correlation functions MolPy ports from TRAVIS: the **Van Hove correlation
function** $G(r, t)$ — the time-dependent generalization of the radial
distribution function — and the **Legendre reorientational correlation functions**
$C_1(t)$, $C_2(t)$ that quantify how fast molecular vectors lose their orientation.
Together they bridge the static structure of the [structural guide](structure.md)
and the transport coefficients of the [transport guide](transport.md): they show
*how structure decorrelates in time*.

As elsewhere, the correlation kernels run in Rust (`molrs`); the MolPy layer
unwraps trajectories and returns a typed result.

!!! note "Conventions used throughout"
    - Distances are in Å; time lags are given in **frames** (multiply by the frame
      spacing for ps).
    - $\langle\cdots\rangle$ averages over particles and time origins.

---

## 1. The Van Hove function is g(r) resolved in time

The Van Hove correlation function counts particles at separation $r$ after a delay
$t$, and splits naturally into a **self** and a **distinct** part:

$$
G(r,t) = \underbrace{\frac{1}{N}\Big\langle\sum_i \delta\big(r - |\mathbf r_i(t) - \mathbf r_i(0)|\big)\Big\rangle}_{G_s(r,t)\ \text{(same particle)}}
       + \underbrace{\frac{1}{N}\Big\langle\sum_{i\ne j}\delta\big(r - |\mathbf r_i(t) - \mathbf r_j(0)|\big)\Big\rangle}_{G_d(r,t)\ \text{(distinct particles)}}.
$$

- **$G_s(r,t)$** is the self-part: the distribution of single-particle
  displacements. At short time it is a sharp peak at $r=0$; at long time it
  broadens into the Gaussian of Fickian diffusion, and its second moment is the
  [mean-squared displacement](transport.md). Non-Gaussian shapes flag hopping or
  caging.
- **$G_d(r,t)$** is the distinct-part: at $t=0$ it is exactly $\rho\,g(r)$, and its
  decay shows how the structural shells around a particle wash out as neighbours
  diffuse away.

---

## 2. Computing the Van Hove function

```python
from molpy.compute import VanHove

vh = VanHove(n_rbins=200, r_max=15.0, lags=[1, 5, 10, 50, 100])
result = vh(frames)

result.r_centers    # radial grid, Å
result.lags         # the time lags (frames)
result.g_self       # G_s(r, t): rows are lags, columns radial bins
result.g_distinct   # G_d(r, t) (present when result.has_distinct)
```

Choose `lags` to straddle the dynamics of interest — a few short lags to resolve
the ballistic/caging regime and longer ones to reach the diffusive plateau.

---

## 3. Reorientation: how fast vectors forget their direction

For a unit vector $\mathbf u(t)$ rigidly attached to a molecule (a bond, a dipole,
a symmetry axis), the **Legendre reorientational correlation functions** are

$$
C_\ell(t) = \big\langle P_\ell\big(\mathbf u(0)\cdot\mathbf u(t)\big)\big\rangle,
\qquad P_1(x)=x,\quad P_2(x)=\tfrac{1}{2}(3x^2-1).
$$

Both decay from 1 (perfectly correlated) toward 0 (fully randomized) on the
reorientational time scale. The order matters because **different experiments probe
different $\ell$**: dielectric relaxation measures $C_1$, while NMR spin-relaxation,
fluorescence depolarization, and IR/Raman lineshapes measure $C_2$. Fitting an
exponential tail $C_\ell(t)\approx e^{-t/\tau_\ell}$ gives the reorientational
correlation time $\tau_\ell$; in the diffusive limit $\tau_1/\tau_2 = 3$.

---

## 4. Computing reorientational correlations

```python
import numpy as np
from molpy.compute import LegendreReorientation

pairs = np.array([[o, h1], [o, h2]], dtype=np.int64)   # O–H bond vectors
reor = LegendreReorientation(max_lag=500)
result = reor(frames, pairs)

result.lags   # lags (frames)
result.c1     # C_1(t)
result.c2     # C_2(t)
```

---

## 5. Pitfalls checklist

1. **`r_max` beyond half the box** → the distinct part is corrupted by periodic
   images; keep `r_max ≤ L/2`.
2. **Lags longer than the trajectory supports** → few time origins remain, so the
   long-lag tail is noisy; ensure many origins per lag.
3. **Reading $\tau$ from a non-exponential head** → fit $C_\ell$ on its long-time
   exponential tail, not the librational sub-picosecond decay.
4. **Unnormalized vectors** → supply genuine bond endpoints; the kernel forms unit
   vectors, but a degenerate pair (identical atoms) is undefined.
5. **Wrapped coordinates for $G_s$** → single-particle displacements require
   unwrapped trajectories, or the self-part saturates at the box size.

---

## 6. References

- L. Van Hove, *Phys. Rev.* **95**, 249 (1954) — the correlation function G(r, t).
- B. J. Berne, R. Pecora, *Dynamic Light Scattering*, Wiley (1976) — reorientational
  correlation functions and the $C_1$/$C_2$ distinction.
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — TRAVIS.

## See also

- [Diffusion & Ionic Transport](transport.md) — the MSD is the second moment of $G_s$.
- [Structural Analysis](structure.md) — $G_d(r, 0) = \rho\,g(r)$.
- [Dielectric Spectroscopy](dielectric.md) — $C_1$ underlies the dielectric response.
- [API reference: Compute](../api/compute.md).
