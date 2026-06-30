# Hydrogen-Bond Networks

This page is a self-contained, textbook-style introduction to **hydrogen-bond
detection** in MolPy. A hydrogen bond is identified geometrically — by a
donor–acceptor distance and a donor–H···acceptor angle — and the per-frame bond
list it produces is the starting point for counting coordination, mapping
networks, and (combined with the [persistence](persistence.md) analysis) measuring
hydrogen-bond lifetimes. The canonical applications are water, alcohols, and
protic ionic liquids.

As elsewhere, the geometric search runs in Rust (`molrs`); the MolPy layer feeds
it the donor/acceptor selections and returns a typed result.

!!! note "Conventions used throughout"
    - Distances are in Å, angles in degrees.
    - A **donor** is a `(D, H)` pair (the heavy atom and its bonded hydrogen); an
      **acceptor** is a single heavy atom.
    - The default criterion is the Luzar–Chandler geometry: donor–acceptor
      distance ≤ 3.5 Å and D–H···A angle ≥ 150°.

---

## 1. A hydrogen bond is a geometric event

There is no quantum-mechanical observable that switches on at a hydrogen bond, so
simulations use a **geometric criterion**: a donor `(D, H)` and an acceptor `A`
are hydrogen-bonded at a frame when

$$
r_{D\cdots A} \le r_c \qquad\text{and}\qquad \angle(D\text{–}H\cdots A) \ge \theta_c.
$$

The distance can be measured donor-to-acceptor or hydrogen-to-acceptor (selectable
through the criterion). The cutoffs are not universal constants — they should be
read off the first minimum of the relevant $g(r)$ and the
[distance–angle CDF](distributions.md), which is exactly the joint distribution
whose populated region *defines* the bond.

---

## 2. Detecting hydrogen bonds

Supply the donor `(D, H)` pairs and the acceptor indices; tune the geometry with
an optional `HBondCriterion`:

```python
import numpy as np
from molpy.compute import HBonds, HBondCriterion

donors = np.array([[o1, h1], [o1, h2]], dtype=np.int64)   # (D, H) pairs
acceptors = np.array([o2, o3, o4], dtype=np.int64)

hb = HBonds(donors, acceptors, HBondCriterion(dist_cutoff=3.5, angle_cutoff=150.0))
result = hb(frames)

result.counts      # number of H-bonds per frame
result.per_frame   # lists of (D, H, A, distance, angle) per frame
```

The per-frame tuples let you build the bond network (degree distribution,
ring statistics) or feed specific donor–acceptor pairs into a lifetime analysis.

---

## 3. From a bond list to lifetimes

A single-frame count tells you *how many* hydrogen bonds exist, not *how long* they
last. To get the lifetime, treat each detected donor–acceptor pair as an
association and run the [pair-persistence](persistence.md) survival analysis on it:
the **intermittent** correlation gives the structural hydrogen-bond lifetime
$\tau_\text{HB}$ of Luzar–Chandler, while the **continuous** correlation gives the
much shorter "first-break" time. Reporting both, with the geometric criterion used,
is the standard way to characterize hydrogen-bond dynamics.

---

## 4. Pitfalls checklist

1. **Criterion sensitivity** → counts and lifetimes depend strongly on $r_c$ and
   $\theta_c$; choose them from the distance–angle CDF and state them explicitly.
2. **Donor list must pair D with its H** → each donor entry is `(heavy, hydrogen)`;
   a heavy atom with two hydrogens contributes two donor rows.
3. **Self-pairs** → exclude intramolecular donor/acceptor combinations if you only
   want intermolecular bonds.
4. **Distance convention** → donor–acceptor vs. hydrogen–acceptor cutoffs are not
   interchangeable; pick one and keep it consistent across systems.
5. **Lifetime ≠ count** → a high instantaneous count can coexist with a short
   lifetime; the two answer different questions.

---

## 5. References

- A. Luzar, D. Chandler, *Nature* **379**, 55 (1996); *Phys. Rev. Lett.* **76**,
  928 (1996) — geometric criterion and hydrogen-bond kinetics.
- D. C. Rapaport, *Mol. Phys.* **50**, 1151 (1983) — continuous vs. intermittent
  bond correlation functions.
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — TRAVIS.

## See also

- [Pair Persistence](persistence.md) — turn the bond list into a lifetime.
- [Distribution Functions](distributions.md) — the distance–angle CDF that defines the criterion.
- [Compute overview](index.md) — the Compute → Result pattern.
- [API reference: Compute](../api/compute.md).
