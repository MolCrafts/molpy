# Diffraction

An X-ray or neutron experiment never sees atomic positions. It shines a beam at
a sample and records how much radiation scatters into each angle. The quantity
that comes out is the **static structure factor** $S(k)$, and if you want to
compare a simulation against an experiment, this is the language you have to
speak.

$S(k)$ carries the same information as [$g(r)$](rdf.md). It is that information
expressed in reciprocal space, where lengths become inverse lengths and "atoms
3.7 Å apart" becomes "a density wave with wavenumber $k$".

## Where the formula comes from

Two atoms separated by $\mathbf{r}_{ij}$ scatter waves that arrive at the
detector with a phase difference $\mathbf{k}\cdot\mathbf{r}_{ij}$. A liquid or
powder has no preferred orientation, so every orientation of $\mathbf{r}_{ij}$
occurs with equal probability. Averaging $e^{i\mathbf{k}\cdot\mathbf{r}}$ over
all directions of $\mathbf{r}$ gives $\sin(kr)/(kr)$, and summing over all pairs
gives the **Debye scattering equation**:

$$
S(k) = \frac{1}{N}\left\langle \sum_{i}\sum_{j}
\frac{\sin(k r_{ij})}{k r_{ij}} \right\rangle .
$$

MolPy evaluates this directly from coordinates — no histogram and no Fourier
transform of $g(r)$.

One property of that evaluation matters more than it first appears: the sum runs
over **literal pair distances, not minimum-image distances**. The estimator
treats your coordinates as an isolated cluster in open space, even when
`frame.box` is periodic. This matches freud's Debye implementation, and it has
two consequences. Pairs are counted out to the full diagonal of the box rather
than being capped at $L/2$, and the configuration has a surface, so the
outermost atoms have fewer neighbours than bulk ones. For the 500-atom argon box
used below, that surface is a real finite-size effect on peak heights — it is
part of why the main peak comes out lower than the experimental value quoted
later. Peak *positions* are robust; absolute intensities from a small box are
not.

The equivalent statement in terms of $g(r)$ makes the connection explicit:

$$
S(k) = 1 + 4\pi\rho\int_{0}^{\infty} r^{2}\,[g(r)-1]\,
\frac{\sin(kr)}{kr}\,\mathrm{d}r .
$$

Note what is being transformed: $g(r) - 1$, the *deviation* from randomness. A
structureless fluid has $g = 1$ everywhere, the integral vanishes, and
$S(k) = 1$ at every $k$. Structure in real space is what puts features in
reciprocal space.

## Reading a real curve

<figure id="fig-sk-argon" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
data: {$file: data/diffraction/argon_sk.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: k
    type: quantitative
    title: "k (Å⁻¹)"
  y:
    field: S
    type: quantitative
    title: "S(k)"
    scale: {domain: [0, 2.4]}
```

</div>

**Figure 1.** $S(k)$ of liquid argon at 85 K from the Debye equation, averaged
over 50 configurations of a 500-atom box. The main peak sits at 1.98 Å⁻¹ and
the oscillations damp out by about 6 Å⁻¹.
</figure>

The **main peak at $k = 1.98$ Å⁻¹** is the dominant density wave of the liquid —
the reciprocal-space fingerprint of the first coordination shell. Its position is
the number a diffraction experiment reports; neutron diffraction on liquid argon
at 85 K puts it close to 2.0 Å⁻¹. Its height here, 2.12, falls short of the
experimental ≈ 2.7 for the finite-size reason given above; do not read that as
the force field failing.

The **damped oscillations** past the main peak are the second and third shells.
They die out around 6 Å⁻¹, and beyond that $S(k) \to 1$: at short enough
wavelengths the liquid looks structureless.

Two cautions come with this figure, and both catch people.

**Do not convert the peak with $2\pi/k$.** That gives $2\pi/1.98 = 3.18$ Å,
while the actual first-neighbour distance from [$g(r)$](rdf.md) is 3.68 Å. The
$2\pi/k$ rule works for Bragg planes in a crystal, not for the main peak of a
liquid, because that peak is a broad superposition of many pair distances rather
than one repeat spacing. To go from $S(k)$ to distances, transform the whole
curve; do not read one point off it.

**The figure starts at 1 Å⁻¹ on purpose.** The Debye sum above includes the
$i = j$ terms, each contributing 1, so as $k \to 0$ every term approaches 1 and
$S(k) \to N$. That runaway is bookkeeping, not the thermodynamic limit
$S(0) = \rho k_B T \chi_T$. Reading compressibility off small-$k$ values from
this estimator is wrong, and in a finite box $k < 2\pi/L$ has no meaning anyway
(0.22 Å⁻¹ for the 28.9 Å argon box).

## Computing it

A crystal is the honest way to test the machinery, because you can predict the
answer. FCC reflections appear at $k = 2\pi\sqrt{h^2+k^2+l^2}/a$ with $h,k,l$
all even or all odd.

```python
import numpy as np
import molpy as mp

a = 5.26  # argon FCC lattice constant, Å
basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
xyz = np.array(
    [(np.array([i, j, k]) + b) * a
     for i in range(4) for j in range(4) for k in range(4) for b in basis]
)
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(4 * a)
```

The compute takes the $k$ grid up front and returns, **per frame**, a tuple of
`(k_values, S, n_particles)`:

```python
from molpy.compute import StaticStructureFactorDebye

k = np.linspace(1.0, 4.5, 140)
(k_out, s_k, n_particles), = StaticStructureFactorDebye(k)([frame])

print(n_particles)                        # -> 256
print(round(float(k_out[np.argmax(s_k)]), 2))   # -> 2.06
```

The strongest peak lands at 2.06 Å⁻¹; the (111) reflection is predicted at
$2\pi\sqrt{3}/5.26 = 2.07$ Å⁻¹. Peaks for (200), (220), and (311) follow at
2.39, 3.38, and 3.96 Å⁻¹.

They are broad rather than sharp because the crystal is only four unit cells
across. Peak width in a diffraction pattern is set by the size of the ordered
domain — the same physics behind the Scherrer equation used to size
nanocrystals. Enlarge the lattice and the peaks narrow.

Average over a trajectory by passing more frames; `S(k)` is an ensemble average
like $g(r)$:

```python
per_frame = StaticStructureFactorDebye(k)([frame, frame])
s_mean = np.mean([s for _, s, _ in per_frame], axis=0)
print(s_mean.shape)                       # -> (140,)
```

**Cost.** Every $k$ point sums over every pair: $\mathcal{O}(N^2 n_k)$ per
frame. Doubling the atom count quadruples the work, and a 1000-point $k$ grid
costs ten times a 100-point one. Choose the coarsest grid that resolves your
peaks.

## When it goes wrong

**$S(k)$ shoots up at small $k$.**
Expected, not a bug — the self terms, as above. Start your grid near
$2\pi/L$ at the very lowest, and do not interpret the rise.

**Division-by-zero or `nan` at the first point.**
Your grid includes $k = 0$, where $\sin(kr)/(kr)$ is evaluated directly.
Start above zero.

**$S(k)$ does not approach 1 at large $k$.**
Either far too few frames, or the coordinates are not what you think — check for
a mangled box or duplicated atoms.

**The peak position disagrees with experiment but $g(r)$ looks fine.**
Check units. A GROMACS trajectory is in nm; a $k$ grid in Å⁻¹ against
coordinates in nm gives a peak off by a factor of 10.

**The calculation takes forever.**
$\mathcal{O}(N^2 n_k)$. Reduce the $k$ grid first, then the number of frames,
then the system size.

## Check yourself

- Scatter points at random in a box and compute $S(k)$. You should get 1 at
  every $k$ except the small-$k$ self-term rise — the reciprocal-space statement
  of "no structure".
- Compare the main peak position of argon (1.98 Å⁻¹) with $2\pi/r_1$ using the
  first $g(r)$ peak (3.68 Å). They disagree, and now you know why.
- Double the FCC lattice to $8^3$ cells and watch the Bragg peaks narrow while
  their positions stay put.

## References

- P. Debye, *Ann. Phys.* **351**, 809 (1915) — the scattering equation.
- J. L. Yarnell, M. J. Katz, R. G. Wenzel, S. H. Koenig, *Phys. Rev. A* **7**,
  2130 (1973) — neutron diffraction from liquid argon at 85 K.
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed. (2013) —
  chapter 4 for $S(k)$, its $k\to 0$ limit, and the link to $g(r)$.

## See also

- [RDF](rdf.md) — the real-space partner of this page
- [NeighborList](neighborlist.md) · [Density](density.md)
- [API reference](../api/compute.md)
