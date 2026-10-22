# Reorientation

A molecule in a liquid does two things: it moves, and it turns. [MSD](msd.md)
measures the moving. This page measures the turning.

Rotational relaxation is what NMR spin-lattice relaxation, dielectric
spectroscopy, and fluorescence anisotropy all probe, so it is one of the few
dynamical quantities you can compare against experiment fairly directly — but
only if you use the same $\ell$ the experiment does.

## Two correlation functions, not one

Track a unit vector $\mathbf{u}(t)$ fixed in the molecule — an O–H bond, a
dipole, the long axis of a rod. Its reorientation is measured by the Legendre
correlation functions

$$
C_\ell(t) = \big\langle P_\ell\big(\mathbf{u}(t)\cdot\mathbf{u}(0)\big)\big\rangle,
$$

with $P_1(x) = x$ and $P_2(x) = \tfrac12(3x^2-1)$. Both start at 1 and decay to
0 as the molecule forgets its initial orientation.

Why two? Because different experiments couple to different ranks. $C_1$ is what
dielectric relaxation and infrared absorption see, since those depend on the
dipole direction. $C_2$ is what NMR and depolarized light scattering see,
because those depend on the *axis* rather than its sign — $P_2$ is unchanged by
$\mathbf{u} \to -\mathbf{u}$. Quoting "the rotational correlation time" without
saying which $\ell$ is a common and avoidable ambiguity.

For simple rotational diffusion with rotational diffusion constant $D_r$ there
is an exact answer:

$$
C_\ell(t) = e^{-\ell(\ell+1)D_r t}, \qquad
\tau_\ell = \frac{1}{\ell(\ell+1)D_r},
$$

so

$$
\boxed{\;\frac{\tau_1}{\tau_2} = 3\;}
$$

independently of $D_r$, of temperature, and of the molecule. That ratio is the
most useful diagnostic on this page: it holds whenever reorientation proceeds by
many small random steps, and it *breaks* when it does not — jump reorientation,
strong hydrogen-bond networks, and glassy systems all push it away from 3.

## Testing it where the answer is known

Here are 300 rods undergoing pure rotational diffusion — small random rotations,
nothing else — with $C_1$ and $C_2$ computed from the resulting trajectory.

<figure id="fig-legendre" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
config:
  legend:
    orient: bottom
    direction: horizontal
    title: null
data: {$file: series/reorientation/rod_legendre.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: t
    type: quantitative
    title: "lag (frames)"
  y:
    field: c
    type: quantitative
    title: "C_ℓ(t)"
    scale: {domain: [0, 1]}
  color:
    field: order
    type: nominal
    title: null
```

</div>

**Figure 1.** $C_1$ and $C_2$ for rods undergoing rotational diffusion. Both
decay exponentially; the fitted relaxation times are 276 and 91 frames, a ratio
of **3.04** against the exact 3.
</figure>

That agreement is the validation — of the compute, of the estimator, and of the
sampling. On a real molecular liquid the same measurement becomes physics: a
ratio near 3 says diffusive reorientation, and a ratio far from it says the
molecule is turning by some other mechanism.

## The correlation is measured on unit vectors you choose

`ReorientationAcf` — the orientational autocorrelation function, or OACF —
correlates an `(n_frames, n_vectors, 3)` array of vectors.
The vector that matters is rarely given to you by a file: it is the O–H
direction of every water molecule under the minimum image, a molecular dipole,
the normal of an aromatic ring, the long axis of a chain. Building it is a few
lines of NumPy you want to see; the correlation is the part you do not want to
write twice. Each vector is normalised inside the kernel, so only its direction
counts, and a zero vector — almost always the wrong pair of atoms — is refused
with a `ValueError`.

The rods below are stored the way a simulation would store them: two atoms per
rod in a frame, with a `bonds` block naming the pair.

```python
import numpy as np
import molpy as mp
from molpy.compute import ReorientationAcf

rng = np.random.default_rng(0)
n_rods, n_frames = 300, 400
axis = rng.normal(size=(n_rods, 3))
axis /= np.linalg.norm(axis, axis=1, keepdims=True)
centres = rng.uniform(0.0, 100.0, size=(n_rods, 3))
bonds = {
    "atomi": np.arange(n_rods, dtype=np.uint32),
    "atomj": np.arange(n_rods, 2 * n_rods, dtype=np.uint32),
}

frames = []
for _ in range(n_frames):
    kick = rng.normal(0.0, 0.06, size=(n_rods, 3))
    kick -= (kick * axis).sum(axis=1, keepdims=True) * axis   # rotation only
    axis = axis + kick
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    pos = np.concatenate([centres - 0.5 * axis, centres + 0.5 * axis])
    frame = mp.Frame()
    frame["atoms"] = {"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]}
    frame.box = mp.Box(mp.core.Cell.lengths([200.0] * 3))
    frame["bonds"] = bonds
    frames.append(frame)
```

Subtracting the component along `axis` before adding the kick is what keeps the
motion a *rotation*: a displacement along the axis would change the rod's length
instead of its direction.

Reading the tracked vectors is your step, through the frame's own blocks: the
bond endpoints index the coordinates, and the box gives the minimum-image
displacement from atom *i* to atom *j*.

```python
def bond_vectors(frame):
    xyz = frame.coords
    bonds = frame["bonds"]
    return frame.box.delta(
        xyz[bonds["atomi"]], xyz[bonds["atomj"]], minimum_image=True
    )

u = np.stack([bond_vectors(frame) for frame in frames])   # (n_frames, n_rods, 3)
```

One instance measures one order, so $C_1$ and $C_2$ are two configured
correlators applied to the same vectors. `max_lag` is in **frames**; `dt` is the
time between frames, and sets the unit of `lag_times`:

```python
c1 = ReorientationAcf(1, 200).compute(vectors=u, dt=1.0)
c2 = ReorientationAcf(2, 200).compute(vectors=u, dt=1.0)
print(round(float(c1.values[0, 0]), 3), round(float(c2.values[0, 0]), 3))   # -> 1.0 1.0
print(c1.values.shape)                                                       # -> (201, 1)
```

Both start at exactly 1, as any normalised correlation function must — the
first thing to check. The result is a `TimeCorrelation`: `values[lag, channel]`
at `lag_times[lag]`, here a single channel because the curve is averaged over
every rod and every time origin.

### The relaxation time is a fit, or an integral

For a single exponential the relaxation time is the decay constant of the
curve. `DebyeFit` fits $C(t) = A\,e^{-t/\tau}$ to one channel of a correlation:

```python
from molpy.compute import DebyeFit

tau1 = DebyeFit().compute(correlation=c1, channel=0).tau
tau2 = DebyeFit().compute(correlation=c2, channel=0).tau
print(round(tau1 / tau2, 2))                             # -> 3.02
```

The fit assumes the decay *is* a single exponential. The integral
$\tau_\ell = \int_0^\infty C_\ell(t)\,\mathrm{d}t$ does not, and it is the number
to quote when the decay is stretched, as it is in water. The running integral
of a correlation is `CumulativeTrapezoid`, itself a `TimeCorrelation`; its last
value is the integral over the computed lags, meaningful only once the curve has
reached zero inside them:

```python
from molpy.compute import CumulativeTrapezoid

running = CumulativeTrapezoid().compute(correlation=c2, channel=0)
print(running.n_lags == c2.n_lags)                       # -> True
```

### How it is computed

$C_1$ is the vector autocorrelation itself, one FFT per vector and component.
$C_2$ looks like it should not go through an FFT, because the square sits inside
the average — but $(\hat{\mathbf{u}}\cdot\hat{\mathbf{v}})^2$ is the dot
product of the six independent components of
$\hat{\mathbf{u}}\otimes\hat{\mathbf{u}}$ (the three squares once, the three cross
terms twice), so
$C_2 = \tfrac32\langle Q(\tau)\cdot Q(\tau+t)\rangle - \tfrac12$ is six FFTs per
vector. Both orders are exact, $\mathcal{O}(T\log T)$, and run on the same
estimator as every other correlation in molpy, `Correlator` with the
`legendre` kernel; orders above 2 fall back to the direct sum over origins.

### Per-vector curves and fewer origins

`reduction="per_item"` keeps one curve per vector instead of the mean — one
channel per rod — and `stride` takes every *s*-th frame as a time origin: four
times fewer origins, four times faster, and the same curve to within noise. It
is the way to make a long trajectory affordable.

```python
per_rod = ReorientationAcf(2, 50, reduction="per_item", stride=4).compute(
    vectors=u, dt=1.0
)
print(per_rod.values.shape)                              # -> (51, 300)
```

### O–H vectors in water are built the same way

For water the vector is the H → O direction, one per hydrogen, under the
minimum image. With your own trajectory loaded as frames, the vectors and the
correlation are:

```python
# docs: skip — needs your own trajectory
def oh_vectors(frame, h_idx, o_of_h):
    """H -> bonded O vectors for one frame, minimum image, shape (n_H, 3)."""
    xyz = frame.coords
    return frame.box.delta(xyz[h_idx], xyz[o_of_h], minimum_image=True)

u = np.stack([oh_vectors(f, h_idx, o_of_h) for f in frames])
c2 = ReorientationAcf(2, 2000).compute(vectors=u, dt=0.05)   # dt in ps
tau2 = CumulativeTrapezoid().compute(correlation=c2, channel=0).values[-1, 0]
```

## Rigid-body orientations correlate through Wigner functions

When the simulation stores each molecule's orientation as a unit quaternion —
rigid bodies, coarse-grained ellipsoids — the vector is implicit, and the
rank-$\ell$ correlation is the average of the Wigner $D$-matrix of the relative
rotation between $t_0$ and $t_0 + t$. `RotationalAutocorrelation` takes the
`(n_frames, n_bodies, 4)` quaternions, and follows freud's convention of a
single time origin, the first frame. Rotational diffusion gives the same
$\ell(\ell+1)$ ratio:

```python
from molpy.compute import RotationalAutocorrelation

def rotate(q, w):
    """Compose each quaternion q with the small rotation of angle vector w."""
    dq = np.concatenate([np.ones((len(w), 1)), 0.5 * w], axis=1)
    dq /= np.linalg.norm(dq, axis=1, keepdims=True)
    a0, a, b0, b = dq[:, :1], dq[:, 1:], q[:, :1], q[:, 1:]
    out = np.concatenate(
        [a0 * b0 - (a * b).sum(1, keepdims=True), a0 * b + b0 * a + np.cross(a, b)],
        axis=1,
    )
    return out / np.linalg.norm(out, axis=1, keepdims=True)

rng = np.random.default_rng(1)
q = np.tile([1.0, 0.0, 0.0, 0.0], (100, 1))
quaternions = np.empty((300, 100, 4))
for t in range(300):
    q = rotate(q, rng.normal(0.0, 0.03, size=(100, 3)))
    quaternions[t] = q

r1 = RotationalAutocorrelation(1, 100).compute(orientations=quaternions, dt=1.0)
r2 = RotationalAutocorrelation(2, 100).compute(orientations=quaternions, dt=1.0)
ratio = (
    DebyeFit().compute(correlation=r1, channel=0).tau
    / DebyeFit().compute(correlation=r2, channel=0).tau
)
print(round(ratio, 1))                                   # -> 3.0
```

## When it goes wrong

**`ValueError` about a zero vector.**
Two atoms of a pair coincide — almost always the wrong pairing. Check the
indices that built the vectors.

**$C_\ell(0) \ne 1$.**
The vectors are not the ones you meant — a wrong column, or vectors mixed
between frames. Stop here.

**$C_2$ decays more slowly than $C_1$.**
Impossible for real reorientation — $\tau_1 > \tau_2$ always. Suspect a swapped
assignment.

**$\tau_1/\tau_2$ is far from 3.**
Sometimes physics, sometimes not. Genuine causes: jump reorientation, anisotropic
rotation, hydrogen-bond-mediated turning. Artefacts: too short a trajectory, or
fitting a stretched exponential with a single time. Check the fit residuals
before claiming the physics.

**The curves are noisy at long lag.**
Same time-origin problem as [MSD](msd.md): few independent origins remain out
there. Fit the early, well-sampled decay.

**The correlation does not reach zero.**
Something is restricting rotation — a frozen system, a rod stuck in a
crystalline environment, or a constraint you forgot. In a liquid it must decay.

## Check yourself

- Confirm both curves start at 1.00.
- Fit $\tau_1$ and $\tau_2$ and check the ratio. On a diffusively reorienting
  system you should get 3 to within a few percent, as above.
- Halve the angular step in the example. Both times should quadruple (the
  rotational diffusion constant scales with the step squared) and the ratio
  should not move.
- Compare the `DebyeFit` time with the integral. If they disagree, the decay is
  not a single exponential and the fitted $\tau$ is not the number to quote.

## References

- P. Debye, *Polar Molecules*, Chemical Catalog Company (1929) — rotational
  diffusion and $C_1$.
- D. A. McQuarrie, *Statistical Mechanics*, Harper & Row (1976), ch. 21 —
  Legendre reorientational correlation functions.
- A. Luzar, D. Chandler, *J. Chem. Phys.* **98**, 8160 (1993) — reorientation in
  hydrogen-bonded liquids, where the ratio departs from 3.
- B. J. Berne, R. Pecora, *Dynamic Light Scattering*, Wiley (1976), ch. 7 —
  which spectroscopy sees which $\ell$.
- G. Lipari, A. Szabo, *J. Am. Chem. Soc.* **104**, 4546 (1982) — $C_2$ as the
  NMR relaxation correlation function.
- D. Laage, J. T. Hynes, *Science* **311**, 832 (2006) — O–H reorientation in
  water by large-angle jumps.

## See also

- [VACF](vacf.md) — the `TimeCorrelation` a correlation returns, and its fits
- [Van Hove](van_hove.md) — the translational partner of this measurement
- [Dielectric](dielectric.md) — where $C_1$ becomes a measurable spectrum
- [Persist](persist.md) — how long the neighbours it turns among survive
- [API reference](../api/compute.md)
