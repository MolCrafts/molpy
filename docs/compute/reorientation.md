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
data: {$file: data/reorientation/rod_legendre.json}
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
decay exponentially; the fitted relaxation times are 294 and 97 frames, a ratio
of **3.02** against the exact 3.
</figure>

That agreement is the validation — of the compute, of the estimator, and of the
sampling. On a real molecular liquid the same measurement becomes physics: a
ratio near 3 says diffusive reorientation, and a ratio far from it says the
molecule is turning by some other mechanism.

## Computing it

`LegendreReorientation` reads the tracked vectors from each frame's **`bonds`**
topology block — `(atomi, atomj)`, giving the vector from atom *i* to atom *j*.
There is no separate vector argument. Whatever you want to track, express it as
a bond.

```python
import numpy as np
import molpy as mp
from molpy.compute import LegendreReorientation

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
    frame.box = mp.Box.cubic(200.0)
    frame["bonds"] = bonds
    frames.append(frame)
```

Subtracting the component along `axis` before adding the kick is what keeps the
motion a *rotation*: a displacement along the axis would change the rod's length
instead of its direction.

```python
result = LegendreReorientation(max_lag=200)(frames)
c1, c2 = np.asarray(result.c1), np.asarray(result.c2)
print(round(float(c1[0]), 3), round(float(c2[0]), 3))   # -> 1.0 1.0
print(c1.shape)                                          # -> (201,)
```

Both start at exactly 1, as any normalized correlation function must — the first
thing to check. Extract relaxation times by fitting the logarithm over the range
where the curve is neither saturated nor buried in noise:

```python
lag = np.asarray(result.lags, dtype=float)
# Single-exponential τ from log-linear fit on the decaying part of C_ℓ(t).
keep1 = (c1 > 0.05) & (lag > 0)
keep2 = (c2 > 0.05) & (lag > 0)
tau1 = -1.0 / np.polyfit(lag[keep1], np.log(c1[keep1]), 1)[0]
tau2 = -1.0 / np.polyfit(lag[keep2], np.log(c2[keep2]), 1)[0]
print(round(tau1 / tau2, 2))                             # -> 3.02
```

`max_lag` is in **frames**, not time, and `stride` sets the spacing between time
origins — raise it to trade statistics for speed on long trajectories.

## When it goes wrong

**$C_\ell(0) \ne 1$.**
The bond block is wrong, or the vectors are not being normalized as you assume.
Stop here.

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
- Halve the angular step in the example. Both times should double and the ratio
  should not move.

## References

- P. Debye, *Polar Molecules*, Chemical Catalog Company (1929) — rotational
  diffusion and $C_1$.
- D. A. McQuarrie, *Statistical Mechanics*, Harper & Row (1976), ch. 21 —
  Legendre reorientational correlation functions.
- A. Luzar, D. Chandler, *J. Chem. Phys.* **98**, 8160 (1993) — reorientation in
  hydrogen-bonded liquids, where the ratio departs from 3.

## See also

- [Van Hove](van_hove.md) — the translational partner of this measurement
- [Dielectric](dielectric.md) — where $C_1$ becomes a measurable spectrum
- [Persist](persist.md) — how long the neighbours it turns among survive
- [API reference](../api/compute.md)
