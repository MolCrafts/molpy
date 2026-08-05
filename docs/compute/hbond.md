# HBond

This page is a self-contained, textbook-style introduction to **hydrogen-bond
detection** in MolPy. A hydrogen bond is identified geometrically — by a
donor–acceptor distance and a donor–H···acceptor angle — and the per-frame bond
list it produces is the starting point for counting coordination, mapping
networks, and (combined with the [persistence](persist.md) analysis) measuring
hydrogen-bond lifetimes. Canonical applications: water, alcohols, amides, and
protic ionic liquids.

The geometric search runs in the high-performance backend; the MolPy layer feeds
it the donor/acceptor selections and returns a typed result.

!!! note "Conventions used throughout"
    - Distances are in Å, angles in **degrees**.
    - A **donor** is a `(D, H)` pair (heavy atom + its bonded hydrogen); an
      **acceptor** is a single heavy atom (often O, N, F, Cl).
    - Default geometric criterion is Luzar–Chandler: donor–acceptor distance
      $r_{D\cdots A}\le 3.5$ Å and $\angle(D\text{–}H\cdots A)\ge 150^\circ$.
    - Pair these counts with [Persist](persist.md) for lifetimes; detection
      alone is a *static* geometric event.

---

## 1. Physical picture: what is a hydrogen bond in MD?

There is no quantum-mechanical operator that is $1$ on a hydrogen bond and $0$
off it. In condensed-phase MD the bond is an **operational definition**: a
donor–hydrogen–acceptor geometry that lies in the populated basin of the joint
distance–angle distribution. That basin is real — it shows up as a peak in the
[combined distribution function](distribution.md) — but its boundary is a
modelling choice.

Why geometry works:

1. **Electrostatics + Pauli exclusion** create a preferred short $D\cdots A$
   contact with a near-linear $D$–$H\cdots A$ arrangement.
2. The first minimum of $g_{D A}(r)$ is a natural outer shell edge for
   “associated”.
3. An angle cutoff rejects accidental short contacts that are bent (not
   H-bond-like).

The goal of detection is therefore not a unique truth but a **reproducible,
reportable criterion** whose counts and lifetimes can be compared across systems.

---

## 2. Geometric criterion

A donor `(D, H)` and acceptor `A` form a hydrogen bond at a given frame when

$$
\boxed{\;
r_{DA} \le r_c
\quad\text{and}\quad
\theta_{DHA} \ge \theta_c
\;}
$$

with

$$
r_{DA} = \min_{\text{images}|\mathbf{r}_A - \mathbf{r}_D|,
\qquad
\theta_{DHA}
= \angle(\mathbf{r}_H-\mathbf{r}_D,\;\mathbf{r}_A-\mathbf{r}_H)
$$

(or the equivalent hydrogen-centred distance $r_{HA}$ when that convention is
chosen). MolPy’s `HBondCriterion` stores the cutoffs; defaults match the
Luzar–Chandler water literature:

| Parameter | Symbol | Default | Role |
|---|---|---|---|
| Distance cutoff | $r_c$ | $3.5$ Å | first-shell edge of $g_{\mathrm{OO}$ in water |
| Angle cutoff | $\theta_c$ | $150^\circ$ | near-linear $D$–$H\cdots A$ |

### 2.1 Distance convention: $D\cdots A$ vs $H\cdots A$

Two common practices:

- **Donor–acceptor** ($r_{DA}$): robust when H positions are noisy; standard for
  classical water models.
- **Hydrogen–acceptor** ($r_{HA}$): closer to the H-bond “length” of structural
  chemistry; more sensitive to librations of H.

They are **not interchangeable** at fixed numerical cutoffs. Pick one, document
it, and keep it fixed when comparing systems.

### 2.2 Why cutoffs must come from data

The “right” $(r_c,\theta_c)$ is the contour that encloses the associated basin
of the joint distribution $p(r,\theta)$. In practice:

1. Compute $g_{DA}(r)$ and read the **first minimum** for a candidate $r_c$.
2. Build the distance–angle [CDF](distribution.md) for donor–H–acceptor triples.
3. Draw $(r_c,\theta_c)$ so the bond region is a connected high-density patch,
   not an arbitrary rectangle through noise.

A criterion chosen from folklore without checking $p(r,\theta)$ will silently
mis-count mixed solvents, ionic liquids, and force fields with shifted
solvation shells.

<figure id="fig-hbond-geom" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:9"
mark:
  type: line
  strokeWidth: 2.2
  interpolate: monotone
data:
  values:
    - {r: 2.4, g: 0.0}
    - {r: 2.6, g: 0.3}
    - {r: 2.8, g: 2.8}
    - {r: 3.0, g: 1.2}
    - {r: 3.2, g: 0.7}
    - {r: 3.5, g: 0.5}
    - {r: 4.0, g: 0.9}
    - {r: 4.5, g: 1.05}
    - {r: 5.5, g: 1.0}
encoding:
  x:
    field: r
    type: quantitative
    title: r_DA (Å)
  y:
    field: g
    type: quantitative
    scale: {zero: false}
    title: g_DA(r)
  color:
    value: "#0284c7"
```

</div>

**Figure 1.** Schematic $g_{DA}(r)$: first peak (H-bonded shell) and first minimum (~3.5 Å in SPC water) that sets a natural $r_c$.
</figure>

---

## 3. From geometry to kinetics: Luzar–Chandler

A single-frame bond list answers *how many* bonds exist. Kinetics ask *how long*
a tagged donor–acceptor pair stays bonded. Define the indicator

$$
h_{ij}(t) =
\begin{cases}
1 & \text{pair }(i,j)\text{ satisfies the geometric criterion at }t,\\
0 & \text{otherwise.}
\end{cases}
$$

Two classical correlation functions follow (see also
[persistence](persist.md)):

$$
c(t) = \frac{\langle h(0)\,h(t)\rangle}{\langle h\rangle}
\qquad\text{(intermittent / structural)},
$$

$$
c_c(t) = \frac{\langle h(0)\,H(t)\rangle}{\langle h\rangle}
\qquad\text{(continuous / first-break)},
$$

where $H(t)=1$ only if the pair was bonded **at every** intermediate frame
between $0$ and $t$. Intermittent $c(t)$ allows reformation after a brief break;
continuous $c_c(t)$ dies at the first exit.

Luzar and Chandler showed that the reactive flux of the continuous population
separates into:

1. a **fast librational transient** (sub-picosecond rattling in the well), and
2. a slower **activated breaking rate** — the chemical lifetime of interest.

Reporting **both** continuous and intermittent lifetimes, with the geometric
criterion stated in full, is the standard characterization of H-bond dynamics.

### 3.1 Mean lifetime

Integrate or fit the intermittent correlation:

$$
\tau_\mathrm{HB}
= \int_0^\infty c(t)\,\mathrm{d}t
\quad\text{or}\quad
c(t)\approx e^{-t/\tau_\mathrm{HB}\ \text{(long-time tail)}.
$$

Do **not** fit the librational head of $c_c(t)$ and call it the chemical
lifetime.

---

## 4. Network observables from a bond list

Once each frame yields a set of edges $(D,H,A)$, the H-bond network is an
undirected graph on heavy atoms (or on molecules):

| Observable | Definition | Why it matters |
|---|---|---|
| Mean degree $\langle n_\mathrm{HB}\rangle$ | average bonds per donor/acceptor | bulk coordination |
| Per-molecule $n_\mathrm{HB}$ | bonds donated + accepted | local defects, interfaces |
| Shared pairs / rings | closed loops in the graph | water rings, ice-like order |
| Percolation | giant connected component | network spanning in mixtures |

MolPy’s `HBonds` result exposes `counts` and `per_frame` tuples
`(D, H, A, distance, angle)` so you can build these reductions in a few lines of
NumPy / NetworkX without re-running the geometric search.

---

## 5. Detecting hydrogen bonds

Supply donor `(D, H)` pairs and acceptor indices; tune geometry with
`HBondCriterion`:

```python
import numpy as np
import molpy as mp

def _frame(step: int) -> mp.Frame:
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0.0, 20.0, size=(30, 3)) + 0.1 * step
    frame = mp.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.box = mp.Box.cubic(20.0)
    return frame

frames = [_frame(step) for step in range(10)]
```

```python
from molpy.compute import HBonds, HBondCriterion

o1, h1, h2 = 0, 1, 2          # one donor water
o2, o3, o4 = 3, 6, 9          # three acceptor oxygens
donors = np.array([[o1, h1], [o1, h2]], dtype=np.int64)  # (D, H) pairs
acceptors = np.array([o2, o3, o4], dtype=np.int64)

hb = HBonds(
    donors,
    acceptors,
    HBondCriterion(dist_cutoff=3.5, angle_cutoff=150.0),
)
result = hb(frames)

result.counts      # number of H-bonds per frame
result.per_frame   # lists of (D, H, A, distance, angle) per frame
```

A heavy atom with two hydrogens contributes **two** donor rows. Exclude
intramolecular $(D,A)$ combinations when you want intermolecular bonds only.

---

## 6. From a bond list to lifetimes

Treat each detected donor–acceptor pair as an association and run
[pair-persistence](persist.md) survival analysis:

- **`intermittent`** → structural $\tau_\mathrm{HB}$ (Luzar–Chandler).
- **`continuous`** → first-break time (much shorter under rattling).
- **`ssp`** → stable-states picture with $r_1 > r_0$ buffer (recommended for
  noisy cutoffs / ion pairs).

Feed the same $r_c$ that defined the geometric bond (or a slightly larger outer
$r_1$ for SSP). Always report definition + criterion together.

---

## 7. Pitfalls checklist

1. **Criterion sensitivity** → counts and lifetimes depend strongly on
   $(r_c,\theta_c)$; choose them from the distance–angle CDF and state them.
2. **Donor list must pair D with its H** → each entry is `(heavy, hydrogen)`.
3. **Self-pairs** → drop intramolecular donor/acceptor if only intermolecular
   bonds are wanted.
4. **Distance convention** → $D\cdots A$ vs $H\cdots A$ cutoffs are not
   interchangeable.
5. **Lifetime ≠ count** → a high instantaneous count can coexist with a short
   lifetime.
6. **Comparing definitions** → continuous / intermittent / SSP are different
   numbers by construction.
7. **Sparse dump interval** → miss sub-picosecond re-crossings; dump denser
   than the lifetime you claim.

---

## 8. References

- A. Luzar, D. Chandler, *Nature* **379**, 55 (1996); *Phys. Rev. Lett.* **76**,
  928 (1996) — geometric criterion and hydrogen-bond kinetics.
- D. C. Rapaport, *Mol. Phys.* **50**, 1151 (1983) — continuous vs intermittent
  bond correlation functions.
- A. Luzar, *J. Chem. Phys.* **113**, 10663 (2000) — resolving H-bond kinetics.
- F. H. Stillinger, *Adv. Chem. Phys.* **31**, 1 (1975) — network picture of
  water connectivity.
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — AIMD analysis feature set (distributions, H-bonds, spectra).

## See also

- [Pair Persistence](persist.md) — turn the bond list into a lifetime.
- [Distribution Functions](distribution.md) — distance–angle CDF that defines
  the criterion.
- [Compute overview](index.md) — the Compute → Result pattern.
- [API reference: Compute](../api/compute.md).
