# Persist

This page is a self-contained, textbook-style introduction to **pair persistence
analysis** — how MolPy measures how long two particles stay associated (within a
distance cutoff) and turns that into a residence-time correlation function.
Canonical applications: hydrogen-bond dynamics in water, ion-pair lifetimes in
electrolytes, and solvation-shell exchange.

Bookkeeping is provided by `Persist`. You pass **coordinate arrays** (not a
tag-string recipe).

!!! note "Conventions used throughout"
    - A *pair* is one reference particle $i$ and one partner $j$.
    - Bonded if the minimum-image distance is within a cutoff. Inner $r_0$
      (formation) and outer $r_1\ge r_0$ (breaking) may differ.
    - $\langle\cdots\rangle_t$ averages over time origins; $\tau$ is the lag.
    - Length **Å**, time **fs**. $C(\tau)$ is dimensionless.
      `max_correlation_time` is in **frames**.

---

## 1. Why lifetimes need their own tool

Diffusion and conductivity ([transport](msd.md)) tell you how *far*
things move, not how *long* a contact survives. Proton transfer, ion-pair
stability, and shell exchange are governed by **lifetimes**. Near a sharp
cutoff, coordinates **rattle** in and out of the bonded region on a
sub-picosecond scale. A lifetime measure must define, carefully, what counts as
the same bond surviving through that noise.

Persistence analysis answers:

> Given that pair $(i,j)$ was associated at time origin $t$, what is the
> probability it is still considered associated at $t+\tau$?

---

## 2. The survival correlation function

Define a Boolean survival for one pair and one origin:

$$
S_{ij}(t, t+\tau) =
\begin{cases}
1 & \text{pair still “alive” at } t+\tau\text{ under the chosen definition},\\
0 & \text{otherwise.}
\end{cases}
$$

The **survival correlation** is the origin- and pair-averaged survival:

$$
\boxed{\;
C(\tau)
= \Big\langle
  \frac{1}{N_i}\sum_i\sum_j S_{ij}(t,\,t+\tau)
\Big\rangle_t
\;}
$$

Properties:

- **$C(0)$** equals the mean number of partners per reference particle at the
  formation criterion — a coordination number.
- **$C(\tau)$ decays** as associations break; the shape encodes the lifetime
  distribution.
- A single-exponential tail $C(\tau)\approx C(0)\,e^{-\tau/\tau_\mathrm{res}$
  defines a mean residence time $\tau_\mathrm{res}$.
- The integral estimator
  $\tau_\mathrm{res}=\int_0^\infty C(\tau)/C(0)\,\mathrm{d}\tau$ is robust when
  the tail is noisy but the decay is complete.

<figure id="fig-persist" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:9"
mark:
  type: line
  strokeWidth: 2.2
  interpolate: monotone
data:
  values:
    - {t: 0, C: 1.0}
    - {t: 0.5, C: 0.7}
    - {t: 1.0, C: 0.45}
    - {t: 2.0, C: 0.25}
    - {t: 4.0, C: 0.1}
    - {t: 8.0, C: 0.03}
    - {t: 12.0, C: 0.01}
encoding:
  x:
    field: t
    type: quantitative
    title: τ (ps)
  y:
    field: C
    type: quantitative
    scale: {zero: false}
    title: C(τ) / C(0)
  color:
    value: "#0284c7"
```

</div>

**Figure 1.** Schematic normalised pair-survival correlation: integral of $C(\tau)/C(0)$ estimates the mean residence time.
</figure>

---

## 3. Three definitions of survival

All three share the same **birth** condition (within $r_0$ at the time origin)
and differ in what keeps the bond alive. The choice is physics, not a
hyperparameter to “tune for nicer plots”.

### 3.1 Continuous (`continuous`, also `cr` / `rf`)

Strict definition (Rapaport, 1983): the pair must remain within $r_1$ at
**every** frame from $t$ to $t+\tau$. The first exit kills the bond for that
origin.

$$
S^\mathrm{cont}_{ij}(t,t+\tau)
= \prod_{s=0}^{n_\tau}
  \mathbf{1}\!\bigl[r_{ij}(t+s\Delta t)\le r_1\bigr],
$$

with birth requiring $r_{ij}(t)\le r_0$. Set $r_1=r_0$ for the classic form.
**Sensitive to rattling**: one brief excursion zeros $S$ forever for that
origin, so continuous lifetimes are short and dump-interval dependent.

### 3.2 Intermittent (`intermittent`, also `imm`)

Permissive definition (Luzar & Chandler, 1996): only the endpoints matter —
bonded at $t$ and bonded at $t+\tau$, regardless of intermediate breaks.

$$
S^\mathrm{int}_{ij}(t,t+\tau)
= \mathbf{1}\!\bigl[r_{ij}(t)\le r_0\bigr]\,
  \mathbf{1}\!\bigl[r_{ij}(t+\tau)\le r_1\bigr].
$$

This is the **structural** lifetime: it includes re-crossings and answers
“is the pair still associated after time $\tau$?”. Standard for hydrogen-bond
$\tau_\mathrm{HB}$.

### 3.3 Stable-states picture (`ssp`)

Laage & Hynes (2008): born within $r_0$, and remains alive while staying within
a larger outer cutoff $r_1\ge r_0$. The annulus $r_0 < r \le r_1$ is a buffer
that suppresses rattling without fully ignoring intermediate history the way
intermittent does.

$$
S^\mathrm{ssp}_{ij}(t,t+\tau)
= \mathbf{1}\!\bigl[r_{ij}(t)\le r_0\bigr]
  \prod_{s=1}^{n_\tau}
  \mathbf{1}\!\bigl[r_{ij}(t+s\Delta t)\le r_1\bigr].
$$

**Recommended default for ion pairs** and any system where a single hard cutoff
is noisy. Choose $r_0$ at the first $g(r)$ minimum and $r_1$ slightly larger
(or at a clear plateau of the potential of mean force).

### 3.4 Relation to older “tolerance time” recipes

Some codes allow intermittent bonds to be “dead” for at most a tolerance
$\Delta t_\mathrm{tol}$ without counting as broken. MolPy’s three methods are
explicit and time-tolerance-free; they match the physics layer used by the
`tame` / Luzar–Chandler / Laage–Hynes literature without a hidden grace period.

---

## 4. Choosing $r_0$ and $r_1$

| Rule | Practice |
|---|---|
| Inner $r_0$ | first minimum of the relevant $g_{ij}(r)$ |
| Outer $r_1$ | $=r_0$ (continuous/intermittent classic) or $r_0+\delta$ with $\delta\sim 0.5$–$1$ Å (SSP) |
| Validation | $C(0)$ should match the coordination number from integrating $g(r)$ to $r_0$ |
| Sensitivity | report $\tau$ at neighbouring cutoffs; large swings mean the basin is ill-defined |

For hydrogen bonds, prefer the geometric $(r,\theta)$ criterion of
[HBonds](hbond.md) for detection, then persistence on the accepted pairs — or
use a pure distance persistence when angle information is unavailable.

---

## 5. Using `Persist`

Distances use the orthorhombic minimum-image convention per axis.

| Argument | Type | Meaning |
|----------|------|---------|
| `coords_i`, `coords_j` | `(n_frames, n, 3)` | per-species coordinates (wrapped OK) |
| `box_lengths` | `(n_frames, 3)` | orthorhombic edges (≤ 0 disables an axis) |
| `r0`, `r1` | float | inner / outer cutoff, Å (`r0 > 0`, `r1 ≥ r0`) |
| `method` | str | `"continuous"`, `"intermittent"`, or `"ssp"` |
| `dt` | float | frame spacing, **fs** |
| `max_correlation_time` | int | longest lag in frames |
| `exclude_self` | bool | drop $i=j$ when both species are identical |

```python
import numpy as np
from molpy.compute import Persist

rng = np.random.default_rng(0)
# coords_cat, coords_an: (n_frames, n_ions, 3); box: (n_frames, 3)
coords_cat = np.ascontiguousarray(rng.random((30, 8, 3)) * 20.0)
coords_an = np.ascontiguousarray(rng.random((30, 8, 3)) * 20.0)
box = np.ascontiguousarray(np.full((30, 3), 20.0))
res = Persist.pair_survival_tcf(
    coords_cat,
    coords_an,
    box,
    r0=3.0,
    r1=4.0,
    method="ssp",
    dt=10.0,  # fs
    max_correlation_time=10,
    exclude_self=False,
)
C = res["correlation"]  # C(tau); C[0] = mean coordination number
tau = res["lag_times"]  # fs
```

When both species are the same set, pass `exclude_self=True` so $i=j$ is not
counted as a pair.

---

## 6. From persistence to pairing diffusion

Combining distinct-diffusion correlations
([Onsager](onsager.md) / distinct diffusion) with a survival
weight yields a **pairing contribution** to diffusion (Gudla et al., 2021):
only those pairs that are still alive contribute to a correlated displacement
term. Interpret only where **both** the persistence count has converged **and**
the displacement correlation is linear in time.

---

## 7. Pitfalls checklist

1. **Single cutoff with rattling** → continuous lifetime collapses toward the
   frame spacing. Prefer `ssp` with $r_1>r_0$, or `intermittent`.
2. **Cutoff off the RDF** → pick $r_0$ at the first $g(r)$ minimum.
3. **`max_correlation_time` shorter than the lifetime** → $C(\tau)$ never fully
   decays; $\tau_\mathrm{res}$ is truncated.
4. **Sparse sampling** → miss fast re-crossings; continuous is especially
   dump-dependent.
5. **Comparing definitions** → continuous / intermittent / SSP give different
   numbers by construction; always report which one.
6. **Self-pairs** → forget `exclude_self` on identical species and $C(0)$ is
   polluted by $i=j$.

---

## 8. References

- D. C. Rapaport, *Mol. Phys.* **50**, 1151 (1983).
- A. Luzar, D. Chandler, *Nature* **379**, 55 (1996); *Phys. Rev. Lett.* **76**,
  928 (1996).
- A. Luzar, *J. Chem. Phys.* **113**, 10663 (2000).
- R. W. Impey, P. A. Madden, I. R. McDonald, *J. Phys. Chem.* **87**, 5071
  (1983) — residence times of water around ions.
- D. Laage, J. T. Hynes, *J. Phys. Chem. B* **112**, 14230 (2008) — stable-states
  picture of H-bond exchange.
- H. Gudla, Y. Shao et al., *J. Phys. Chem. Lett.* **12**, 8460 (2021) — pairing
  contribution to diffusion.

## See also

- [Hydrogen-Bond Networks](hbond.md)
- [Diffusion & Ionic Transport](msd.md)
- [Structural Analysis](rdf.md) — $g(r)$ for choosing $r_0$
- [Compute overview](index.md)
- [API reference: Compute](../api/compute.md)
