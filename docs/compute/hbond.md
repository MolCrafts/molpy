# Hydrogen bonds

A hydrogen bond is not a term in your force field. In a classical simulation
there is no "H-bond" interaction — only Coulomb and Lennard-Jones — so a
hydrogen bond is something *you* define, by drawing a box in geometry space and
declaring everything inside it bonded.

That makes this page unusual. The compute is simple; the definition is the hard
part, and almost every disagreement in the literature about hydrogen-bond
numbers is a disagreement about the definition rather than about the physics.

## The geometric criterion, and why it is a choice

The standard criterion puts two conditions on a donor–hydrogen–acceptor triple
$D{-}H\cdots A$:

$$
r_{DA} < r_c, \qquad \theta_{DHA} > \theta_c ,
$$

close enough, and straight enough. The Luzar–Chandler values, $r_c = 3.5$ Å and
$\theta_c = 150°$, are the defaults here and the most widely used — but they are
conventions calibrated on SPC water, not constants of nature.

Two details trip people up.

**Which distance.** $r_{DA}$ (donor to acceptor, used here) and $r_{HA}$
(hydrogen to acceptor) differ by roughly an O–H bond length, so a criterion
quoted as "3.5 Å" means different things depending on which was meant. Always
say which.

**Where the cutoff should come from.** Not from folklore — from your own
$g_{DA}(r)$. The first minimum of the donor–acceptor radial distribution is the
defensible boundary, exactly as for [Cluster](cluster.md), and it moves between
force fields, between solvents, and with temperature. A criterion transplanted
from a water paper into an ionic liquid will silently miscount.

Because the definition is binary, the count is discontinuous: a pair at 3.49 Å
is bonded, at 3.51 Å it is not, and nothing physical happens in between. That is
why instantaneous H-bond counts are noisier than they look, and why lifetimes
need the two-threshold treatment on [Persist](persist.md).

!!! note "No figure on this page yet — TODO"
    The figure this section needs is your own measured $g_{DA}(r)$ with the
    first minimum marked, because reading the cutoff off a real curve is the
    entire argument. It cannot be produced here: hydrogen bonding requires a
    molecular liquid with donors and acceptors, and the reference trajectory
    behind the other compute pages is monatomic argon. A sketched curve would be
    exactly the folklore this section warns against. Produce it for your own
    system with [RDF](rdf.md) restricted to donor and acceptor atoms, following
    the partial-distribution recipe in the [compute overview](index.md).

## Computing it

`HBonds` is configured with the criterion and nothing else: `dist_cutoff`
($r_c$, 3.5 Å), `angle_cutoff` ($\theta_c$, 150°) and `dist_kind`, which says
which distance the cutoff applies to — `"donor_acceptor"` (the default) or
`"hydrogen_acceptor"`. The chemistry is data, passed to `compute`: an array of
`(D, H)` index pairs and an array of acceptor indices. It does not guess which
atoms are donors — that is your topology's job, and being made to state it is a
feature.

The candidates come from the core neighbour search, one table per frame: the
donor sites are the query points, the acceptors the reference points, searched
out to at least the cutoff. The table also carries the box it was searched
in, which is how the criterion gets minimum-image distances.

```python
import numpy as np
import molpy as mp
from molpy.compute import HBonds

# One ideal water dimer: O–H pointing straight at a second oxygen 2.8 Å away.
xyz = np.array([
    [0.00, 0.0, 0.0],    # 0: donor O
    [0.96, 0.0, 0.0],    # 1: its H
    [2.80, 0.0, 0.0],    # 2: acceptor O
    [3.20, 0.9, 0.0],    # 3: an H on the acceptor
]) + 10.0
box = mp.Box.cube(30.0)
donors = np.array([[0, 1]])
acceptors = np.array([2])

table = mp.core.NeighborQuery(box, xyz[acceptors], 3.5).query(xyz[donors[:, 0]])
result = HBonds().compute(
    positions=xyz[None], donors=donors, acceptors=acceptors, pairs=[table]
)
print(result.counts.tolist())               # -> [1]
```

The positions are a trajectory, `(n_frames, n_atoms, 3)` — here one frame,
hence `xyz[None]` — and `pairs` holds one table per frame. On your own data
both come out of the frames: `frame.coords` for the positions, and the donor
and acceptor indices from your topology's atom types.

One frame, one hydrogen bond. The geometry of each detection comes back too,
which is how you check the criterion is doing what you think:

```python
bond = result.per_frame[0][0]
print(bond.donor, bond.hydrogen, bond.acceptor)        # -> 0 1 2
print(round(bond.distance, 2), round(bond.angle, 1))   # -> 2.8 180.0
```

2.8 Å and 180° — a perfectly linear bond, comfortably inside both cutoffs.
Build this dimer, tilt it, and watch the detection switch off; that is the
fastest way to see where the boundary actually sits.

A different criterion is a different configured instrument; the data is
unchanged:

```python
strict = HBonds(dist_cutoff=3.0, angle_cutoff=160.0)
tighter = strict.compute(
    positions=xyz[None], donors=donors, acceptors=acceptors, pairs=[table]
)
print(tighter.counts.tolist())              # -> [1]
```

`counts` is the per-frame bond count and `per_frame` the detected `HBond`s of
each frame. Averaging `counts` over a trajectory and dividing by the number of
donors gives hydrogen bonds per molecule — about 3.5 for bulk water with the
Luzar–Chandler criterion, which is the number to sanity-check against.
`components(frame)` labels every atom by the hydrogen-bonded network it belongs
to in that frame — a partition, ready for the [Cluster](cluster.md) and
[Shape](shape.md) computes.

## From a bond list to lifetimes

Counting bonds is the easy half; how long they last is the interesting half, and
it is the same machinery as [Persist](persist.md). The result already knows the
indicator $h(t)$ — `presence()` returns it, one column per `(donor, acceptor)`
pair that was ever bonded — and `Survival` correlates it with itself.

Here the dimer's acceptor slides off-axis for two frames out of twelve, which
is enough to show the two definitions part ways:

```python
from molpy.compute import Survival

trajectory = np.repeat(xyz[None], 12, axis=0)
trajectory[4:6, 2] += [0.0, 1.6, 0.0]       # frames 4 and 5: the bond breaks
tables = [
    mp.core.NeighborQuery(box, frame[acceptors], 3.5).query(frame[donors[:, 0]])
    for frame in trajectory
]
bonds = HBonds().compute(
    positions=trajectory, donors=donors, acceptors=acceptors, pairs=tables
)
presence = bonds.presence()
print(presence.pairs.tolist(), bonds.counts.tolist())
# -> [[0, 2]] [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]

for kind in ("continuous", "intermittent"):
    curve = Survival(kind, 6).compute(presence=presence.values, dt=100.0)
    print(kind, np.round(curve.values[:, 0], 2).tolist())
# -> continuous [0.83, 0.73, 0.6, 0.44, 0.25, 0.14, 0.0]
# -> intermittent [0.83, 0.73, 0.6, 0.56, 0.5, 0.57, 0.67]
```

`presence()` is `None` when no bond formed in any frame, so a typo in the donor
list fails loudly instead of producing a flat zero curve. The continuous curve
reaches zero at the sixth lag — no stretch of six frames is unbroken — while
the intermittent one recovers, because the bond re-formed.

The continuous-versus-intermittent distinction matters more here than for plain
contacts, because hydrogen bonds break and re-form constantly at the threshold.
Luzar and Chandler's reactive-flux treatment exists precisely to separate
genuine breaking from threshold flicker, and a lifetime quoted without saying
which definition produced it is not comparable with anything. The lifetime
itself is the area under $C(\tau)/C(0)$: `CumulativeTrapezoid` on the survival
curve, read at its plateau, divided by $C(0)$.

## When it goes wrong

**Zero bonds detected.**
Check the donor array shape — it must be `(n_donor, 2)` pairs of `(D, H)`, not a
flat list of donor atoms. Then check the angle convention: 150° means nearly
linear, so if the geometry looks right but the reported angle is near 30° you
are measuring the supplement.

**The count is far above literature values.**
Usually the distance convention: a 3.5 Å cutoff applied to $r_{HA}$ rather than
$r_{DA}$ admits many more pairs.

**The count jumps between frames.**
Real, and inherent to a binary criterion. Average over many frames and do not
over-interpret the fluctuations of an instantaneous count.

**Lifetimes come out implausibly short.**
Threshold flicker. Use the two-radius treatment from [Persist](persist.md).

**Bonds are missed across a periodic boundary.**
Search with the frame's real box. The criterion uses minimum-image distances in
the box each table was searched in, so a table searched without one is refused.

**`compute` refuses the tables as the wrong shape.**
The donor sites must be the query points and the acceptors the reference
points, one table per frame, with as many query points as donor pairs.

## Check yourself

- Build the linear dimer above and confirm 2.8 Å / 180°. Then rotate the
  acceptor until the angle falls below the cutoff and confirm the bond vanishes.
- Compute $g_{DA}(r)$ for your own system and find the first minimum. If it is
  not near 3.5 Å, do not use 3.5 Å.
- Count bonds per donor in bulk water; you should get roughly 3.5 with the
  default criterion.

## References

- A. Luzar, D. Chandler, *Nature* **379**, 55 (1996) — the geometric criterion
  and reactive-flux lifetimes used as defaults here.
- A. Luzar, D. Chandler, *Phys. Rev. Lett.* **76**, 928 (1996) — hydrogen-bond
  kinetics in water.
- R. Kumar, J. R. Schmidt, J. L. Skinner, *J. Chem. Phys.* **126**, 204107
  (2007) — how much the answer depends on which definition you pick.

## See also

- [RDF](rdf.md) — where a defensible cutoff comes from
- [Persist](persist.md) — lifetimes, and the two-threshold treatment
- [Distribution](distribution.md) — the underlying angle histograms
- [API reference](../api/compute.md)
