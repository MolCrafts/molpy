# Distributions

[$g(r)$](rdf.md) histograms distances between *any* pair of atoms that happen to
be near each other. Often you want the opposite: distances between two atoms you
have specifically named — the C–C bond, the O–H–O angle, the backbone torsion.
Those are the internal degrees of freedom, and their distributions are how you
check a force field, spot a strained geometry, or find the rotameric states of a
molecule.

One compute covers them, `DistributionFunction`, with the observable named
per arity: `"distance"` (pairs), `"angle"` (triplets), `"dihedral"`
(quadruplets); `CombinedDistribution` builds a joint histogram over several at
once.

## The selection is an index array you read from the topology

The compute does not know what a bond is. You tell it which atoms to measure by
handing over one row of atom indices per sample — a pair for a distance, a
triple for an angle, a quadruple for a dihedral. Those rows normally already
exist on the frame, as the endpoint columns of its topology blocks, and reading
them out is your step:

| Observable | Block you read | Endpoint columns |
|---|---|---|
| `DistributionFunction("distance", ...)` | `bonds` | `atomi`, `atomj` |
| `DistributionFunction("angle", ...)` | `angles` | `atomi`, `atomj`, `atomk` |
| `DistributionFunction("dihedral", ...)` | `dihedrals` | `atomi`, `atomj`, `atomk`, `atoml` |

The angle is measured at the **middle** atom of the triple. Because the
selection is just an array, nothing ties it to the frame's own topology: a
hand-picked list of O–H···O triples or every C–C pair of one residue type is
measured the same way.

The constructor holds what defines the measurement — the observable and its
binning — and `compute` takes the data as parallel per-frame lists:
`positions`, optionally `simbox` (to measure across periodic faces by the
minimum image), and `tuples`. One configured instrument therefore runs
unchanged over a whole trajectory.

!!! warning "Angular ranges are in radians"
    The kernel bins angles in **radians**. Give only `n_bins` and an angle is
    binned over its natural range $[0, \pi]$, a dihedral over $(-\pi, \pi]$
    (kept signed). An explicit `axis=Axis(min, max, n_bins)` given in degrees
    puts every physically possible angle (0 to $\pi \approx 3.14$) into the
    first two or three bins, and the rest of the histogram is empty:

    ```text
    DistributionFunction("angle", n_bins=90)                       # [0, pi]
    DistributionFunction("dihedral", axis=Axis(-np.pi, np.pi, 90))
    ```

    The bin centres come back in radians too. `"distance"` has no natural
    range: it needs an explicit `axis`, in the coordinates' length unit.

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
data: {$file: series/distribution/solid_angle.json}
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

Build a frame and its topology block together — here 5000 isolated C–C bonds
with lengths drawn around 1.53 Å, so the answer is known:

```python
import numpy as np
import molpy as mp
from molpy.compute import DistributionFunction

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
frame.box = mp.Box(mp.core.Cell.lengths([100.0] * 3))
index = np.arange(2 * n_bonds, dtype=np.uint32).reshape(n_bonds, 2)
frame["bonds"] = {"atomi": index[:, 0], "atomj": index[:, 1]}
```

Read the pair rows out of the `bonds` block, then measure:

```python
from molpy.compute import Axis

bonds = frame["bonds"]
pairs = np.column_stack([bonds["atomi"], bonds["atomj"]])

lengths_of = DistributionFunction("distance", axis=Axis(1.35, 1.75, 60))
result = lengths_of.compute(
    positions=[frame.coords], simbox=[frame.box], tuples=[pairs]
)

centers = result.histogram.axes[0].centers
density = result.density
print(result.n_raw_samples, result.density_sin_corrected)   # -> 5000 None
print(round(float((centers * density).sum() / density.sum()), 3))   # -> 1.53
```

Bonds drawn with a mean of 1.53 Å come back with a mean of 1.53 Å. Run that
check on a distribution whose answer you know before trusting one whose answer
you do not.

`n_raw_samples` counts the tuples you passed and `histogram.binned` how many of
them fell inside the axis. `density_sin_corrected` is `None` for a distance,
because the sin correction applies only to angles: a signed dihedral has no
solid-angle Jacobian, so it is `None` for a dihedral too.

For an angular distribution, the same pattern over the natural range in radians:

```python
triples = np.arange(3 * 1000, dtype=np.uint32).reshape(1000, 3)
angles = mp.Frame()
pts = rng.normal(size=(3000, 3)) * 5.0 + 50.0
angles["atoms"] = {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}
angles.box = mp.Box(mp.core.Cell.lengths([100.0] * 3))
angles["angles"] = {
    "atomi": triples[:, 0], "atomj": triples[:, 1], "atomk": triples[:, 2]
}

block = angles["angles"]
triplets = np.column_stack([block["atomi"], block["atomj"], block["atomk"]])
adf = DistributionFunction("angle", n_bins=90).compute(
    positions=[angles.coords], tuples=[triplets]
)
axis = adf.histogram.axes[0]
print(axis.min, round(axis.max, 4), adf.density_sin_corrected.shape)
# -> 0.0 3.1416 (90,)
```

When a bond length and an angle are correlated, their marginals hide it, and
you want the joint histogram. `CombinedDistribution` takes one observable and
one `Axis` per dimension; `compute` takes, per frame, one tuple array per
observable, row $k$ of each describing the same sample. Angular axes are
sin-corrected automatically, `sin_weight=[i]` asks for it on axis `i` as well,
and `marginal(i)` integrates the others out:

```python
from molpy.compute import CombinedDistribution

joint = CombinedDistribution(
    ["distance", "angle"], [Axis(0.0, 30.0, 15), Axis(0.0, np.pi, 18)]
).compute(positions=[angles.coords], tuples=[[triplets[:, :2], triplets]])
print(joint.density.shape, joint.sin_corrected.tolist())   # -> (15, 18) [False, True]
print(joint.marginal(0).density.shape)                     # -> (15,)
```

## When it goes wrong

**The histogram is empty except for the first few bins.**
A range given in degrees. See the warning above.

**`n_raw_samples` is 0.**
The tuple array you passed is empty — usually the block you read it from was
missing or had no rows. Check `list(frame.keys())` and the block's `n_rows`.

**A peak at 90° that disappears when you think about it.**
The $\sin\theta$ weight. Look at `density_sin_corrected`.

**The distribution is cut off at one end.**
Your `Axis` window clips real samples. Samples outside the range are simply
not binned, and nothing warns you — compare `histogram.binned` with
`n_raw_samples`.

**Bond lengths look right but angles are nonsense.**
Check the column order of your tuples. The angle is taken at the middle
index; if your rows list the central atom first, every angle is wrong in a way
that still looks plausible.

**Distributions from two runs disagree slightly.**
Compare bin widths before anything else. A histogram is not a function; its
apparent peak height depends on binning.

## Check yourself

- Generate bonds with a known mean and width, then confirm the measured mean
  and standard deviation match. This catches unit and indexing errors instantly.
- Compare `histogram.binned` with `n_raw_samples`. If they differ, your range
  is clipping data.
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
