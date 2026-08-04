# Pair Persistence & Residence Times

This page is a self-contained introduction to **pair persistence analysis** —
how MolPy / molrs measure how long two particles stay associated (within a
distance cutoff) and turn that into a residence-time correlation function.
Canonical applications: hydrogen-bond dynamics in water and ion-pair lifetimes
in electrolytes.

The per-pair, per-frame bookkeeping runs in **molrs**. MolPy re-exports
`Persist` identity-style. You pass **coordinate arrays** (not a
tag-string recipe).

!!! note "Conventions used throughout"
    - A *pair* is one reference particle $i$ and one partner $j$.
    - Bonded if the minimum-image distance is within a cutoff. Inner $r_0$
      (formation) and outer $r_1\ge r_0$ (breaking) may differ.
    - $\langle\cdots\rangle_t$ averages over time origins; $\tau$ is the lag.
    - Units: length **Å**, time **fs**. $C(\tau)$ is dimensionless.
      `max_correlation_time` is in **frames**.

---

## 1. Why lifetimes need their own tool

Diffusion and conductivity ([transport guide](transport.md)) tell you how *far*
things move, not how *long* a contact survives. Proton transfer, ion-pair
stability, and solvation-shell exchange are governed by **lifetimes**. Contacts
near a cutoff **rattle**; a lifetime measure must define what counts as the same
bond surviving.

---

## 2. The survival correlation function

$$
S_{ij}(t, t+\tau) =
\begin{cases}
1 & \text{pair still alive at } t+\tau,\\
0 & \text{otherwise.}
\end{cases}
$$

$$
\boxed{\;C(\tau) = \Big\langle\,\frac{1}{N_i}\sum_i\sum_j S_{ij}(t,\,t+\tau)\,\Big\rangle_t\;}
$$

- **$C(0)$** is the mean coordination number.
- **$C(\tau)$ decays** with the residence time; fit
  $C(\tau)\approx C(0)\,e^{-\tau/\tau_\text{res}}$ or integrate.

---

## 3. Three definitions of survival

All share the same *birth* condition (within $r_0$ at $t$) and differ in survival:

### 3.1 Continuous (`continuous`, also `cr` / `rf`)

Strict (Rapaport, 1983): within $r_1$ at **every** frame from $t$ to $t+\tau$.
First exit kills the bond for that origin. Use $r_1=r_0$ for the classic form.

### 3.2 Intermittent (`intermittent`, also `imm`)

Permissive (Luzar & Chandler, 1996): within $r_1$ **at** $t+\tau$ only; gaps
allowed. Structural lifetime including re-crossings.

### 3.3 Stable-states picture (`ssp`)

Laage & Hynes (2008): born within $r_0$, alive while staying within $r_1\ge r_0$.
Buffer between cutoffs suppresses rattling. **Recommended default for ion pairs.**

!!! note "Relation to tame"
    Mirrors the [tame](https://github.com/Roy-Kid/tame) `persist` physics with
    explicit, time-tolerance-free criteria. IMM's tolerance-time variant is not
    reproduced.

---

## 4. Using `Persist`

Call the static array API. Distances use the orthorhombic minimum-image
convention per axis.

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

When both species are the same set, pass `exclude_self=True` (or identical
arrays with self-pairs dropped) so $i=j$ is not counted as a pair.

---

## 5. From persistence to pairing diffusion

Combining distinct-diffusion correlations
([transport §2](transport.md#2-self-vs-distinct-diffusion)) with a survival
weight yields a **pairing contribution** to diffusion (Gudla et al., 2021).
Interpret only where **both** the persistence count has converged **and** the
displacement correlation is linear in time.

---

## 6. Pitfalls checklist

1. **Single cutoff with rattling** → continuous lifetime collapses to the frame
   spacing. Prefer `ssp` with $r_1 > r_0$, or `intermittent`.
2. **Cutoff off the RDF** → pick $r_0$ at the first $g(r)$ minimum.
3. **`max_correlation_time` shorter than the lifetime** → $C(\tau)$ never decays.
4. **Sparse sampling** → miss fast re-crossings.
5. **Comparing definitions** → continuous / intermittent / SSP give different
   numbers by construction; always report which one.

---

## 7. References

- D. C. Rapaport, *Mol. Phys.* **50**, 1151 (1983).
- A. Luzar, D. Chandler, *Nature* **379**, 55 (1996); *Phys. Rev. Lett.* **76**, 928 (1996).
- A. Luzar, *J. Chem. Phys.* **113**, 10663 (2000).
- R. W. Impey, P. A. Madden, I. R. McDonald, *J. Phys. Chem.* **87**, 5071 (1983).
- D. Laage, J. T. Hynes, *J. Phys. Chem. B* **112**, 14230 (2008).

[^gudla]: H. Gudla, Y. Shao et al., *J. Phys. Chem. Lett.* **12**, 8460 (2021).

## See also

- [Diffusion & Ionic Transport](transport.md)
- [Compute overview](index.md)
- [API reference: Compute](../api/compute.md)
