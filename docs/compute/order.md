# Order parameters

Is this configuration a crystal?

You cannot answer from the coordinates, and — perhaps surprisingly — not
reliably from [$g(r)$](rdf.md) either. A supercooled liquid and a defective
crystal can have similar radial distributions, and averaging over all atoms
hides the case that usually matters most: a system that is *part* solid and
*part* liquid, with an interface in between.

What you need is a number for **each atom**, describing how symmetric its own
neighbourhood is. That is what an order parameter is.

## Why spherical harmonics

Start from the obvious idea and watch it fail. You could describe an atom's
environment by the list of directions to its neighbours. But rotate the whole
crystal and every direction changes, while the crystal is obviously still a
crystal. A useful order parameter must be **invariant under global rotation**.

The standard construction gets there in two steps. First expand the bond
directions in spherical harmonics — for atom $i$ with $N_b(i)$ neighbours,

$$
q_{\ell m}(i) = \frac{1}{N_b(i)}\sum_{j=1}^{N_b(i)}
Y_{\ell m}\big(\theta_{ij},\varphi_{ij}\big),
$$

where $(\theta_{ij},\varphi_{ij})$ are the polar angles of the bond from $i$ to
$j$. These coefficients still change under rotation, but in a controlled way:
rotating the system mixes the $2\ell+1$ values of $m$ among themselves without
changing their total magnitude. So take that magnitude:

$$
\boxed{\;q_\ell(i) = \sqrt{\frac{4\pi}{2\ell+1}
\sum_{m=-\ell}^{\ell}\big|q_{\ell m}(i)\big|^{2}}\;}
$$

Now the answer depends on the *shape* of the neighbourhood, not on how it is
oriented in the box. This is the **Steinhardt order parameter**.

The choice of $\ell$ selects which symmetry you are sensitive to. $\ell = 6$ is
the workhorse for close-packed structures; $\ell = 4$ helps separate FCC from
HCP and BCC. Odd $\ell$ vanishes for centrosymmetric environments and is rarely
used.

Two variants appear in the literature and both are available here. The
**third-order invariant** $w_\ell$ is built from triple products of the same
$q_{\ell m}$ rather than their squared magnitude; it is also rotationally
invariant, and it is *signed*, which is what makes it able to separate FCC from
HCP where $q_6$ alone struggles. Ask for it with `wl=True`, and normalize it
with `wl_normalize=True`. The **locally averaged** variant of Lechner and
Dellago first averages $q_{\ell m}$ over an atom and its neighbours before
taking the magnitude, which suppresses thermal noise dramatically; ask for it
with `average=True`. Both change the numbers, so neither is comparable with the
plain values tabulated below.

## The reference values are exact — and depend on the cutoff

Order parameters are worth learning because ideal lattices have known values you
can check against:

| Structure | neighbours | $q_4$ | $q_6$ |
|---|---|---|---|
| FCC | 12 | 0.1909 | 0.5745 |
| BCC | 14 | 0.0364 | 0.5107 |
| HCP | 12 | 0.0972 | 0.4848 |
| Liquid | first shell | small | small |

Running `Steinhardt` on a perfect FCC lattice reproduces $q_4 = 0.1909$ and
$q_6 = 0.5745$ to four decimal places, and on BCC with a 14-neighbour cutoff,
$0.0364$ and $0.5107$. When a compute has an analytic answer available, check it
there before trusting it anywhere else.

Now the part that ruins reproducibility if you skip it. Those BCC numbers assume
the conventional **14**-neighbour definition — the 8 nearest plus the 6
next-nearest, which in BCC are only 15 % further away. Tighten the cutoff to
capture just the 8 nearest and the same perfect lattice gives

$$
q_4 = 0.5092, \qquad q_6 = 0.6285,
$$

nothing like the tabulated values. Same crystal, same code, different neighbour
definition. **A $q_\ell$ value without its cutoff is not comparable to
anything** — not to a table, not to a paper, not to your own earlier run.

## What a liquid looks like

Liquid argon at its first-shell cutoff gives $q_6 = 0.340 \pm 0.064$.

<figure id="fig-q6" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
config:
  legend:
    orient: bottom
    direction: horizontal
    title: null
data: {$file: data/order/steinhardt_q6.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: q6
    type: quantitative
    title: "q₆"
    scale: {domain: [0, 0.7]}
  y:
    field: p
    type: quantitative
    title: "pdf"
  color:
    field: phase
    type: nominal
    title: null
```

</div>

**Figure 1.** Per-atom $q_6$ for a perfect FCC crystal (a single spike at
0.5745) and for liquid argon at 85 K (broad, centred on 0.34). The gap between
them is what makes per-atom classification possible.
</figure>

Two things to read off. The crystal is a **delta function**: every atom has an
identical environment, so there is no distribution at all. The liquid is broad
and does **not** reach zero — a disordered environment still gives a small
$q_6$, because a finite number of neighbours never averages perfectly to
nothing. That residual is of order $1/\sqrt{N_b}$, so it shrinks as coordination
grows, which is another reason $q_\ell$ from different cutoffs cannot be
compared.

The two distributions barely overlap, and that is precisely what you need in
order to label individual atoms.

## Computing it

`Steinhardt` takes a **list** of degrees — `l=[4, 6]`. Passing `l=6` raises
`TypeError: argument 'l': 'int' object is not an instance of 'Sequence'`.

```python
import numpy as np
import molpy as mp
from molpy.compute import NeighborList, Steinhardt

a = 5.26
basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
xyz = np.array(
    [(np.array([i, j, k]) + b) * a
     for i in range(5) for j in range(5) for k in range(5) for b in basis]
)
crystal = mp.Frame()
crystal["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
crystal.box = mp.Box.cubic(5 * a)
```

The result is a list with one **dict** per frame, holding the degrees you asked
for and a `(n_degrees, n_atoms)` array of per-atom values:

```python
nlist = NeighborList(cutoff=4.5)(crystal)
result, = Steinhardt(l=[4, 6])([crystal], [nlist])

print(sorted(result))                        # -> ['l', 'ql']
q = np.asarray(result["ql"])
print(q.shape)                               # -> (2, 500)
print(round(float(q[0].mean()), 4), round(float(q[1].mean()), 4))  # -> 0.1909 0.5745
```

Confirm the cutoff caught the shell you meant, using the identity from
[NeighborList](neighborlist.md):

```python
print(2 * nlist.n_pairs / crystal["atoms"].nrows)   # -> 12.0
```

Twelve neighbours per atom — the FCC first shell, so the reference values apply.

### Classifying atoms one by one

`SolidLiquid` builds on $q_\ell$: two neighbouring atoms form a *solid-like
bond* when their $q_{\ell m}$ vectors are sufficiently aligned, and an atom
counts as solid when it has enough such bonds.

Note the argument, because it is genuinely inconsistent with `Steinhardt`:
`SolidLiquid` computes one bond correlation at a single degree, so its `l` is a
plain **int**. `Steinhardt` evaluates several degrees at once, so its `l` is a
**Sequence**. `SolidLiquid(l=[6])` is as much an error as `Steinhardt(l=6)`.

```python
from molpy.compute import SolidLiquid

n_solid_bonds, is_solid = SolidLiquid(l=6)([crystal], [nlist])[0]
n_solid_bonds, is_solid = np.asarray(n_solid_bonds), np.asarray(is_solid)
print(int(n_solid_bonds[0]), bool(is_solid.all()))    # -> 12 True
```

Every atom in the perfect crystal has all 12 of its bonds solid-like. The same
calculation on liquid argon averages 0.28 solid-like bonds per atom and
classifies **no** atom as solid. The defaults (`q_threshold=0.7`,
`n_threshold=6`) separate those two cases with an enormous margin — which is
exactly why you must retune them on a real system, where the margin is narrow
and the answer depends on where you put the line.

### Orientational order of anisotropic particles

`Nematic` asks whether elongated particles point the same way, regardless of
where they sit. It reads per-particle directors from the frame's `orientations`
block — the same `(atomi, atomj)` schema and the same **one row per atom** rule
described on the [PMFT](pmft.md) page.

It also breaks the pattern of every other compute here: it returns a **single
tuple aggregated over all frames**, not one entry per frame.

```python
from molpy.compute import Nematic

rng = np.random.default_rng(0)
n_rods = 400
centres = rng.uniform(0.0, 30.0, size=(n_rods, 3))

# Each rod is two atoms joined by an `orientations` pair (atomi, atomj).
isotropic = rng.normal(size=(n_rods, 3))
isotropic /= np.linalg.norm(isotropic, axis=1, keepdims=True)
pos = np.concatenate([centres + 0.5 * isotropic, centres - 0.5 * isotropic])
frame = mp.Frame()
frame["atoms"] = {"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]}
frame.box = mp.Box.cubic(30.0)
frame["orientations"] = {
    "atomi": np.arange(n_rods),
    "atomj": np.arange(n_rods, 2 * n_rods),
}
order, eigenvalues, director, q_tensor = Nematic()([frame])
print(round(float(order), 3))                # -> 0.058
```

$S \approx 0$ for randomly oriented rods, as it must be. Align them along $z$
with a little jitter and it jumps:

```python
aligned = np.array([0.0, 0.0, 1.0]) + rng.normal(0.0, 0.15, size=(n_rods, 3))
aligned /= np.linalg.norm(aligned, axis=1, keepdims=True)
pos = np.concatenate([centres + 0.5 * aligned, centres - 0.5 * aligned])
frame["atoms"] = {"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]}
order, _, director, _ = Nematic()([frame])
print(round(float(order), 3))                # -> 0.934
print(np.round(np.abs(director), 2).tolist())  # -> [0.01, 0.01, 1.0]
```

$S = 0.934$, and the recovered `director` is the $z$ axis — the compute found
the alignment direction without being told it. $S$ runs from 0 (isotropic) to 1
(perfectly aligned); real nematic liquid crystals sit around 0.4–0.7.

Note the 0.058 for the isotropic case. That is not zero, and with $N$ particles
it never will be: $S$ has a finite-size floor of roughly $1/\sqrt{N}$. Always
compare weak order against a randomized control at the same $N$.

### Two-dimensional systems

`Hexatic(k=6)` is the 2-D analogue, measuring six-fold bond-orientational order
in a monolayer or slab:

$$
\psi_k(i) = \frac{1}{N_b(i)}\sum_{j=1}^{N_b(i)} e^{\,i k\,\theta_{ij}},
$$

with $\theta_{ij}$ the in-plane bond angle. $|\psi_6| \to 1$ for a triangular
lattice, $\to 0$ for a 2-D liquid. The intermediate hexatic phase — orientational
order surviving after positional order is lost — is the reason the parameter
exists at all.

## When it goes wrong

**Your crystal does not reproduce the reference $q_\ell$.**
Almost always the cutoff. Check `2 * n_pairs / n_atoms` against the coordination
number of your lattice before anything else.

**`TypeError: argument 'l': 'int' object is not an instance of 'Sequence'`.**
Pass `l=[6]`, not `l=6` — but only to `Steinhardt`. `SolidLiquid` takes a bare
int, and giving *it* a list fails the other way. The two differ because one
evaluates several degrees and the other exactly one.

**Liquid $q_6$ is not zero and you expected zero.**
It should not be zero. Finite coordination leaves a residual of order
$1/\sqrt{N_b}$. Compare against a liquid reference, not against 0.

**Everything classifies as solid, or nothing does.**
The `SolidLiquid` thresholds are defaults, not physical constants. Plot the
distribution of solid-like bond counts and put the threshold in the valley
between the peaks.

**Your $q_\ell$ disagrees with a published value.**
Check three things in order: the cutoff, whether the paper used the locally
averaged variant (`average=True`, Lechner–Dellago), and whether $w_\ell$ was
normalized. All three change the numbers, and papers often omit which was used.

## Check yourself

- Run `Steinhardt(l=[4, 6])` on a perfect FCC lattice and confirm 0.1909 and
  0.5745. Then build BCC and reproduce 0.0364 and 0.5107 — but only with a
  14-neighbour cutoff.
- Take that same BCC lattice, tighten the cutoff to 8 neighbours, and watch the
  numbers become 0.5092 and 0.6285. Nothing about the crystal changed.
- Shuffle your rod directors and confirm the nematic $S$ falls to the
  finite-size floor rather than to zero.

## References

- P. J. Steinhardt, D. R. Nelson, M. Ronchetti, *Phys. Rev. B* **28**, 784
  (1983) — the original bond-orientational order parameters and the reference
  table above.
- W. Lechner, C. Dellago, *J. Chem. Phys.* **129**, 114707 (2008) — the locally
  averaged variant, which separates FCC/HCP/BCC far better.
- P. R. ten Wolde, M. J. Ruiz-Montero, D. Frenkel, *J. Chem. Phys.* **104**,
  9932 (1996) — the solid-liquid bond criterion.
- D. R. Nelson, B. I. Halperin, *Phys. Rev. B* **19**, 2457 (1979) — hexatic
  order.
- P. G. de Gennes, J. Prost, *The Physics of Liquid Crystals*, 2nd ed. (1993) —
  the nematic order parameter.

## See also

- [Environment](environment.md) — the bond-orientational diagram behind $q_\ell$
- [RDF](rdf.md) — where the first-shell cutoff comes from
- [Cluster](cluster.md) — grouping the atoms an order parameter has labelled
- [PMFT](pmft.md) — the `orientations` block contract
- [API reference](../api/compute.md)
