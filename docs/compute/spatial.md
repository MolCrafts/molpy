# Spatial distribution

Where does the sodium ion sit relative to a carbonate group — above the plane,
or in it? Where does the next water molecule sit relative to this one: along the
O–H bonds, or near the lone pairs?

[$g(r)$](rdf.md) cannot answer either. It averages over all directions, so
"2.8 Å away" is all you get. [PMFT](pmft.md) recovers two dimensions. The
**spatial distribution function** recovers all three: a density cloud in the
body-fixed frame of a chosen molecule.

## Freezing the tumbling out

The obstacle is that molecules rotate. A neighbour sitting permanently above the
plane of a reference molecule appears, in lab coordinates, to be everywhere —
because the reference has been turning the whole time. Averaging lab-frame
positions therefore reproduces $g(r)$ and nothing more.

So the reference molecule is held still, mathematically, before anything is
binned. Each frame:

1. Take the reference atoms — three or more atoms that define the molecule's
   geometry.
2. Superimpose them onto a fixed **template** by a rigid rotation and
   translation. This is the Kabsch algorithm, which finds the rotation
   minimizing the RMS deviation from the template.
3. Apply that same rotation to every target atom nearby, after
   minimum-image unwrapping about the reference centre of mass.
4. Add the rotated target positions to a 3-D voxel grid.

The result, divided by the bulk density,

$$
g_{\mathrm{SDF}}(\mathbf{x}) =
\frac{\rho(\mathbf{x}_{\text{body}})}{\rho_{\text{bulk}}},
$$

reads exactly like $g(r)$: 1 means "as much density as chance", and lobes are
where the neighbour genuinely prefers to be. Integrating the whole thing over
angles at fixed radius recovers $g(r)$, which is a useful consistency check and
a good way to see how much information the angular resolution added.

The template is what fixes the orientation convention, so **the order of the
reference atoms matters**. Swap two of them and the body frame is a different
one; every lobe moves, and nothing warns you.

!!! note "No figure on this page yet — TODO"
    An SDF is only meaningful for a molecular liquid — water around water,
    an ion around a solvent molecule — sampled over thousands of frames, because
    a 3-D histogram is hungry: a $32^3$ grid is 32 768 voxels competing for
    every neighbour you observe. The reference trajectory behind the other
    compute pages is monatomic argon, which has no body frame at all. A sketched
    lobe diagram would teach the wrong thing about how much sampling this needs.
    Add a figure when a molecular trajectory exists under `scripts/docs_data/`.

## Computing it

Everything is configured up front; the call takes only frames.

```python
import numpy as np
import molpy as mp
from molpy.compute import SpatialDistribution

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(300, 3))
# Atoms 0-2 are the reference molecule, placed at the box centre.
xyz[:3] = np.array([[10.0, 10, 10], [10.76, 10.59, 10], [9.24, 10.59, 10]])

frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(20.0)
```

```python
template = np.array([[0.0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])

sdf = SpatialDistribution(
    reference=[0, 1, 2],              # atoms defining the body frame
    template=template,                # their canonical geometry, same order
    target=list(range(3, 300)),       # atoms whose density is binned
    n=(16, 16, 16),                   # voxels per axis
    extent=(6.0, 6.0, 6.0),           # half-width of the grid, Å
    bulk_density=300 / 20.0**3,       # for the g_SDF normalization
)
result = sdf([frame])

print(np.asarray(result.density).shape)     # -> (16, 16, 16)
print(int(np.asarray(result.counts).sum())) # -> 6
```

Six target atoms fell inside a $\pm 6$ Å box in this single frame, which is the
honest scale of the problem: one frame gives you almost nothing, and `g_sdf`
computed from it is meaningless. Real use means thousands of frames.

Note that `target` must not be the reference atoms. Pointing `target` at an atom
that is part of `reference` bins nothing, silently.

The result carries `counts` (raw), `density` (per unit volume),
`g_sdf` (normalized by `bulk_density`), plus `extent`, `n`, `voxel_volume`, and
`n_frames` for bookkeeping. An optional `orientations` block adds the per-voxel
mean orientation of a head–tail vector on the target species, using the same
per-atom contract described on the [PMFT](pmft.md) page.

## When it goes wrong

**`g_sdf` is enormous — hundreds or thousands.**
Too few samples in too many voxels. With one frame and a $16^3$ grid, a single
neighbour in a 0.05 Å³ voxel produces a huge ratio. Add frames, or coarsen the
grid.

**Everything is zero.**
`target` overlaps `reference`, or the extent is smaller than the distance to the
nearest target atom.

**The lobes are in the wrong place, or rotate between runs.**
Reference-atom order does not match the template order. This is the most common
error and the hardest to spot, because the output still looks like a plausible
SDF.

**Lobes are smeared radially but sharp angularly.**
The reference molecule is flexible, so the Kabsch fit onto a rigid template
leaves residual distortion. Use a more rigid subset of atoms as the reference.

**The grid clips the first solvation shell.**
`extent` is a half-width, not a diameter. To see a shell at 5 Å you need extent
above 5, and more if you want the second.

## Check yourself

- Integrate `g_sdf` over angles at fixed radius and compare with
  [$g(r)$](rdf.md) for the same pair of species. They must agree; if they do
  not, the body frame is wrong.
- Confirm `g_sdf` tends to 1 far from the reference. If it tends to something
  else, `bulk_density` is wrong.
- Permute two reference atoms deliberately and watch the map change. That is how
  much the atom order matters.

## References

- W. Kabsch, *Acta Crystallogr. A* **32**, 922 (1976) — the optimal
  superposition used for the body frame.
- A. K. Soper, *Chem. Phys.* **202**, 295 (1996) — spatial distribution
  functions of molecular liquids.
- I. M. Svishchev, P. G. Kusalik, *J. Chem. Phys.* **99**, 3049 (1993) — SDFs of
  water, the canonical example.

## See also

- [PMFT](pmft.md) — the 2-D, free-energy version of the same idea
- [Distribution](distribution.md) — 1-D internal coordinates
- [RDF](rdf.md) — what an SDF reduces to when you average over angles
- [API reference](../api/compute.md)
