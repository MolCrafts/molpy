# Bond-Orientational Order & Local Environment

This page is a self-contained, textbook-style introduction to MolPy's
**bond-orientational order parameters** — the tools that quantify how *ordered* a
local environment is, and of *what kind*. Where the [structural guide](structure.md)
asks how many neighbours sit at a given distance, these operators ask how those
neighbours are **arranged in angle**: are they on a crystal lattice, a hexagonal
film, or pointing along a common nematic axis? The canonical applications are
crystallization and melting, 2-D phase transitions, and liquid-crystal alignment.

As with every `compute` operator, the spherical-harmonic sums run in Rust
(`molrs`); MolPy supplies coordinates, the box, and a neighbor list and returns a
typed result. Each of these analyses takes **two inputs** — frames and a
`NeighborList` — because "order" is defined relative to a particle's neighbours.

!!! note "Conventions used throughout"
    - Length is in Å. Order parameters are dimensionless.
    - A particle's **bonds** are the vectors to the neighbours in its
      `NeighborList`; the cutoff defining those neighbours is the single most
      consequential choice (see Pitfalls).
    - $Y_{\ell m}$ are the spherical harmonics; $\ell$ is the harmonic degree.

---

## 1. Local order lives in the spherical harmonics of the bonds

Steinhardt, Nelson and Ronchetti's insight (1983) is to expand the directions of
a particle's bonds in spherical harmonics. For particle $i$ with $N_b(i)$
neighbours at directions $(\theta_{ij}, \phi_{ij})$,

$$
q_{\ell m}(i) = \frac{1}{N_b(i)}\sum_{j=1}^{N_b(i)} Y_{\ell m}\big(\theta_{ij}, \phi_{ij}\big).
$$

These complex coefficients rotate with the coordinate frame, so they are not
observables on their own. The physics is in their **rotational invariants**.

---

## 2. q_ℓ and w_ℓ are the rotationally invariant order parameters

The second- and third-order invariants are frame-independent fingerprints of the
local symmetry:

$$
q_\ell(i) = \sqrt{\frac{4\pi}{2\ell+1}\sum_{m=-\ell}^{\ell}\big|q_{\ell m}(i)\big|^2},
\qquad
w_\ell(i) = \sum_{m_1+m_2+m_3=0}
   \begin{pmatrix}\ell&\ell&\ell\\ m_1&m_2&m_3\end{pmatrix}
   q_{\ell m_1}q_{\ell m_2}q_{\ell m_3}.
$$

The degree $\ell$ selects the symmetry you are sensitive to:

- **$q_6$** is the workhorse for close-packed order — it takes large, distinct
  values for fcc, hcp, and bcc, and small values in a liquid.
- **$q_4$** together with $q_6$ separates fcc from hcp from bcc.
- **$w_\ell$** (the third-order invariant) sharpens that discrimination further;
  its sign distinguishes crystal structures that share similar $q_\ell$.

The **locally averaged** variant of Lechner & Dellago (2008) first averages
$q_{\ell m}$ over a particle *and its neighbours* before taking the invariant,
which dramatically sharpens the solid/liquid separation at the cost of one extra
neighbour shell of range.

---

## 3. Computing Steinhardt order with `Steinhardt`

```python
from molpy.compute import NeighborList, Steinhardt

nlist = NeighborList(cutoff=1.5)(frame)        # first-shell neighbours
q = Steinhardt(l=[4, 6])([frame], [nlist])     # per-particle q_4 and q_6
```

Switch to the averaged and third-order variants through the constructor flags:

```python
q_avg = Steinhardt(l=[6], average=True)          # Lechner–Dellago averaged q_6
w = Steinhardt(l=[6], wl=True, wl_normalize=True)  # normalized w_6
```

A per-particle order parameter becomes a phase diagnostic by histogramming it: a
bimodal $q_6$ distribution is the signature of solid and liquid coexisting.

---

## 4. Two-dimensional order: the hexatic parameter

In a 2-D film the relevant symmetry is six-fold, and the order parameter is the
**hexatic** $\psi_k$ (with $k=6$):

$$
\psi_k(i) = \frac{1}{N_b(i)}\sum_{j=1}^{N_b(i)} e^{\,i k\,\theta_{ij}},
$$

where $\theta_{ij}$ is the in-plane bond angle. $|\psi_6|\to 1$ for a perfect
triangular lattice and $\to 0$ in an isotropic liquid; its spatial correlations
are the order parameter of the KTHNY melting scenario.

```python
from molpy.compute import Hexatic

psi6 = Hexatic(k=6)([frame], [nlist])
```

---

## 5. Distinguishing solid from liquid particle-by-particle

To **label** each particle as solid- or liquid-like, `SolidLiquid` implements the
ten Wolde–Ruiz-Montero–Frenkel criterion. Two particles share a *solid-like bond*
when the normalized complex vectors $\mathbf q_\ell$ of their environments are
sufficiently aligned,

$$
s_{ij} = \frac{\sum_m q_{\ell m}(i)\,q_{\ell m}^{*}(j)}
              {\big|\mathbf q_\ell(i)\big|\,\big|\mathbf q_\ell(j)\big|} > q_\text{threshold},
$$

and a particle is **solid** when it has at least `n_threshold` such bonds.

```python
from molpy.compute import SolidLiquid

sl = SolidLiquid(l=6, q_threshold=0.7, n_threshold=6)([frame], [nlist])
```

This is the standard way to track a growing crystalline nucleus inside a melt.

---

## 6. Orientational order of anisotropic particles: the nematic Q-tensor

When particles carry an intrinsic direction $\mathbf u_i$ (rods, mesogens, bonded
segments), collective alignment is measured by the **nematic order tensor**

$$
Q = \frac{1}{N}\sum_i \Big(\tfrac{3}{2}\,\mathbf u_i\otimes\mathbf u_i - \tfrac{1}{2}\,\mathbf I\Big).
$$

Its largest eigenvalue is the scalar nematic order parameter $S$ (0 isotropic, 1
perfectly aligned) and the corresponding eigenvector is the **director**.
`Nematic` reads each particle's orientation axis from the frame's `orientations`
topology block (one `(head, tail)` atom pair per particle; the director is the
unit `head - tail` vector) and returns the order, the eigenvalues, the director,
and the full $Q$-tensor:

```python
from molpy.compute import Nematic

# `frame` must carry an `orientations` block, e.g. one (head, tail) row per particle.
order, eigenvalues, director, q_tensor = Nematic()([frame])
```

---

## 7. The bond-orientational diagram visualizes the local geometry

Where the invariants compress a local environment to a number, **`BondOrder`**
keeps the full picture: it histograms every bond direction onto a spherical
$(\theta, \phi)$ grid, accumulated over the chosen particles and frames. The
resulting diagram shows the angular signature of the coordination shell directly —
the four lobes of a tetrahedral environment, the six of an octahedral one.

```python
from molpy.compute import BondOrder

diagram = BondOrder(n_theta=80, n_phi=160)([frame], [nlist])
```

---

## 8. Pitfalls checklist

1. **Neighbor cutoff sets the answer.** $q_\ell$ depends strongly on which bonds
   are counted. Choose the cutoff at the first minimum of $g(r)$ (see the
   [structural guide](structure.md)), or use a fixed neighbour count, and keep it
   consistent across systems you compare.
2. **Wrong $\ell$ for the symmetry** → $q_6$ for close packing, $\psi_6$ for 2-D
   hexagonal, $q_4\!+\!q_6$ to separate fcc/hcp/bcc. A single $\ell$ rarely
   resolves everything.
3. **Skipping the averaged variant** → unaveraged $q_6$ has broad, overlapping
   solid/liquid peaks; the Lechner–Dellago average is usually worth its extra range.
4. **Normalization conventions** → $w_\ell$ values differ between normalized and
   unnormalized definitions; report which (`wl_normalize`).
5. **Finite size and surfaces** → particles near a free surface or interface have
   truncated neighbour shells and artificially low order; exclude or flag them.
6. **Nematic axis endpoints reversed** → the director is `head - tail` (block
   columns `atomi`/`atomj`); the $Q$-tensor is sign-independent so $S$ is
   unaffected, but the reported director points along the head-to-tail sense.

---

## 9. References

- P. J. Steinhardt, D. R. Nelson, M. Ronchetti, *Phys. Rev. B* **28**, 784 (1983)
  — bond-orientational order parameters $q_\ell$, $w_\ell$.
- D. R. Nelson, B. I. Halperin, *Phys. Rev. B* **19**, 2457 (1979) — hexatic order
  and 2-D melting (KTHNY).
- P. R. ten Wolde, M. J. Ruiz-Montero, D. Frenkel, *J. Chem. Phys.* **104**, 9932
  (1996) — solid-like bond criterion for nucleation.
- W. Lechner, C. Dellago, *J. Chem. Phys.* **129**, 114707 (2008) — locally
  averaged order parameters.
- P. G. de Gennes, J. Prost, *The Physics of Liquid Crystals*, 2nd ed. (1993) —
  the nematic $Q$-tensor and order parameter.
- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) — the
  freud library, on which these kernels are modelled.

## See also

- [Structural Analysis](structure.md) — $g(r)$ to choose the neighbour cutoff,
  plus $S(k)$ and density fields.
- [Molecular Shape, Clustering & Decomposition](descriptors.md) — grouping the
  ordered particles into nuclei and domains.
- [Compute overview](index.md) — the Compute → Result pattern.
- [API reference: Compute](../api/compute.md).
