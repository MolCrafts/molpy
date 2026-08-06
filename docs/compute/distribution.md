# Distributions

[$g(r)$](rdf.md) histograms distances between *any* pair of atoms that happen to
be near each other. Often you want the opposite: distances between two atoms you
have specifically named — the C–C bond, the O–H–O angle, the backbone torsion.
Those are the internal degrees of freedom, and their distributions are how you
check a force field, spot a strained geometry, or find the rotameric states of a
molecule.

Three computes cover them, one per arity: `DistanceDistribution` (pairs),
`AngleDistribution` (triplets), `DihedralDistribution` (quadruplets), plus
`CombinedDistribution` for a joint histogram over several at once.

## The selection comes from the topology, not from arguments

None of these takes a list of atom indices. Each reads its tuples from the
matching **topology block** on every frame:

| Compute | Block | Endpoint columns |
|---|---|---|
| `DistanceDistribution` | `bonds` | `atomi`, `atomj` |
| `AngleDistribution` | `angles` | `atomi`, `atomj`, `atomk` |
| `DihedralDistribution` | `dihedrals` | `atomi`, `atomj`, `atomk`, `atoml` |

The angle is measured at the **middle** atom `atomj`. Endpoint columns are read
as unsigned integers, so build them from an integer array.

This is a deliberate design: the selection lives with the structure, so the same
compute runs unchanged over a trajectory whose topology you defined once.

!!! warning "The angular ranges are documented in degrees but binned in radians"
    `AngleDistribution(n_bins, min=0.0, max=180.0)` and
    `DihedralDistribution(n_bins, min=-180.0, max=180.0)` carry degree-valued
    defaults, but the kernel bins the angle in **radians**. With the defaults,
    every physically possible angle (0 to $\pi \approx 3.14$) falls into the
    first two or three bins out of `n_bins`, and the rest of the histogram is
    empty. Pass the range explicitly in radians:

    ```text
    AngleDistribution(n_bins=90, min=0.0, max=np.pi)
    DihedralDistribution(n_bins=90, min=-np.pi, max=np.pi)
    ```

    `bin_centers` comes back in radians too. `DistanceDistribution` has no
    defaults and is unaffected.

## The sin θ trap

Before reading any angular distribution, there is a piece of geometry to divide
out, and it catches nearly everyone the first time.

Suppose the two bonds at an atom point in completely uncorrelated directions —
no chemistry at all. Is the angle between them uniformly distributed? It is not.
There are far more ways to be at 90° than at 5°, because the set of directions
making an angle $\theta$ with a fixed axis is a cone whose circumference goes as
$\sin\theta$. For isotropic directions,

$$
p(\theta) = \tfrac12\sin\theta .
$$

So a peak at 90° in a raw angular histogram may mean nothing whatsoever. Here is
that measured on 200 000 constructed random triplets, where the correct answer
is known in advance:

<figure id="fig-solid-angle" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
config:
  legend:
    orient: bottom
    direction: horizontal
    title: null
data: {$file: data/distribution/solid_angle.json}
mark: {type: line, strokeWidth: 2.2, interpolate: monotone}
encoding:
  x:
    field: theta
    type: quantitative
    title: "θ (degrees)"
  y:
    field: p
    type: quantitative
    title: "pdf"
  color:
    field: series
    type: nominal
    title: null
```

</div>

**Figure 1.** Angle distribution of triplets whose two bond directions are
independent and isotropic. The measured curve peaks at 91° and follows
$\tfrac12\sin\theta$ to within 6 %; the sin-corrected curve is flat to 1.8 %,
which is the correct answer for "no angular structure".
</figure>

The result object hands you both. `density` is the raw histogram;
`density_sin_corrected` has the $\sin\theta$ weight divided out, so a
structureless distribution comes out **flat** and any departure from flat is
real. Use the corrected one for interpretation, and quote which you plotted.

## Computing it

Build the frame and its topology block together:

```python
import numpy as np
import molpy as mp
from molpy.compute import DistanceDistribution

rng = np.random.default_rng(0)
n_bonds = 5000
lengths = rng.normal(1.53, 0.03, n_bonds)          # a C–C bond, Å
direction = rng.normal(size=(n_bonds, 3))
direction /= np.linalg.norm(direction, axis=1, keepdims=True)

first = np.full((n_bonds, 3), 50.0)
second = first + direction * lengths[:, None]
xyz = np.empty((2 * n_bonds, 3))
xyz[0::2], xyz[1::2] = first, second

frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(100.0)
index = np.arange(2 * n_bonds, dtype=np.uint32).reshape(n_bonds, 2)
frame["bonds"] = {"atomi": index[:, 0], "atomj": index[:, 1]}
```

```python
result = DistanceDistribution(n_bins=60, min=1.35, max=1.75)([frame])

centers = np.asarray(result.bin_centers)
density = np.asarray(result.density)
print(result.n_raw_samples, result.angular)        # -> 5000 False
print(round(float((centers * density).sum() / density.sum()), 3))   # -> 1.53
```

Bonds drawn with a mean of 1.53 Å come back with a mean of 1.53 Å. Run that
check on a distribution whose answer you know before trusting one whose answer
you do not.

`n_raw_samples` counts the tuples found; if it is zero, the block is missing or
its columns are misnamed. `angular` tells you whether the sin correction
applies — `False` here, `True` for angles and dihedrals.

For an angular distribution, the same pattern with the range in radians:

```python
from molpy.compute import AngleDistribution

triples = np.arange(3 * 1000, dtype=np.uint32).reshape(1000, 3)
angles = mp.Frame()
pts = rng.normal(size=(3000, 3)) * 5.0 + 50.0
angles["atoms"] = {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}
angles.box = mp.Box.cubic(100.0)
angles["angles"] = {
    "atomi": triples[:, 0], "atomj": triples[:, 1], "atomk": triples[:, 2]
}

adf = AngleDistribution(n_bins=90, min=0.0, max=float(np.pi))([angles])
print(adf.angular, np.asarray(adf.bin_centers).max() <= np.pi)   # -> True True
```

`CombinedDistribution` takes one `(kind, n_bins, min, max, sin_weight)` tuple
per axis, with `kind` one of `"distance"`, `"angle"`, `"dihedral"`, and builds
the joint histogram — useful when a bond length and a torsion are correlated and
the marginals hide it.

## When it goes wrong

**The histogram is empty except for the first few bins.**
The radians-versus-degrees range. See the warning above.

**`n_raw_samples` is 0.**
No `bonds` / `angles` / `dihedrals` block on the frame, or the endpoint columns
are not named `atomi`, `atomj`, … Check `list(frame.keys())`.

**A peak at 90° that disappears when you think about it.**
The $\sin\theta$ weight. Look at `density_sin_corrected`.

**The distribution is cut off at one end.**
Your `min`/`max` window clips real samples. Samples outside the range are simply
not binned, and nothing warns you — compare `n_binned` with `n_raw_samples`.

**Bond lengths look right but angles are nonsense.**
Check the endpoint order. The angle is taken at `atomj`; if your block lists the
central atom first, every angle is wrong in a way that still looks plausible.

**Distributions from two runs disagree slightly.**
Compare bin widths before anything else. A histogram is not a function; its
apparent peak height depends on binning.

## Check yourself

- Generate bonds with a known mean and width, then confirm the measured mean
  and standard deviation match. This catches unit and indexing errors instantly.
- Compare `n_binned` with `n_raw_samples`. If they differ, your range is
  clipping data.
- Histogram random isotropic triplets and confirm the raw curve is
  $\tfrac12\sin\theta$ and the corrected one is flat. If the raw curve is
  already flat, you are looking at the corrected array.

## References

- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.
  (2017) — internal-coordinate distributions and their Jacobians.
- A. K. Soper, *Chem. Phys.* **202**, 295 (1996) — angular distribution
  functions and the solid-angle weighting.

## See also

- [RDF](rdf.md) — the unselected, all-pairs version
- [Spatial](spatial.md) — the full 3-D body-fixed distribution
- [PMFT](pmft.md) — turning a distribution into a free energy
- [API reference](../api/compute.md)
