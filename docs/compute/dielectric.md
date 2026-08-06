# Dielectric response

Put a polar liquid in an electric field and its molecules turn to align with it.
Reverse the field slowly and they follow. Reverse it fast enough and they cannot
keep up, so the material stops responding.

The frequency where "keeping up" fails is a direct measurement of how fast
molecules reorient, and the whole curve — the **dielectric spectrum**
$\varepsilon^*(\omega)$ — is one of the few quantities that can be compared
against experiment over ten decades of frequency. It is also, for an
electrolyte, entangled with the ionic conductivity in a way that catches people
out, which is most of what this page is about.

## You never simulate the field

The central idea is that you do not have to apply a field at all.

The **fluctuation–dissipation theorem** says a system's response to a small
perturbation is completely determined by its spontaneous fluctuations at
equilibrium. So instead of switching on a field and watching the polarization
build up, you run an ordinary equilibrium trajectory and watch the total dipole

$$
\mathbf{M}(t) = \sum_i q_i \mathbf{r}_i(t)
$$

wander on its own. How fast $\mathbf{M}$ forgets its own direction *is* the
dielectric relaxation.

Everything on this page follows from that: static permittivity from the size of
the fluctuations, the spectrum from their correlation in time.

### Unwrapping is not optional

$\mathbf{M}$ is a sum over $q_i\mathbf{r}_i$, so a single molecule crossing a
periodic boundary shifts it by $qL$ — an enormous artefact next to the real
fluctuations. Worse, the standard fix of unwrapping every atom independently
breaks a molecule apart across the boundary and gives an equally wrong dipole.

The rule: **unwrap each molecule as a unit, relative to its own centre**, not
atom by atom, and not by folding the final coordinates. Get this wrong and
nothing downstream is salvageable, however good the statistics.

## Static permittivity from the size of the fluctuations

For a system simulated under conducting (tin-foil) boundary conditions, the
Neumann fluctuation formula gives

$$
\varepsilon(0) = \varepsilon_\infty +
\frac{\langle \mathbf{M}^2\rangle - \langle \mathbf{M}\rangle^2}
{3\,\epsilon_{\text{vac}} V k_B T}.
$$

($\epsilon_{\text{vac}}$ is the vacuum permittivity; $\varepsilon(0)$ with a
zero argument is the static relative permittivity this page is about. The
literature writes both as $\varepsilon_0$, which is unhelpful, so they are kept
apart here.)

Read it as: **permittivity is variance**. A liquid whose dipole fluctuates
wildly is easy to polarize. $\varepsilon_\infty$ accounts for the electronic
polarizability a fixed-charge force field cannot represent — set it to 1 for a
non-polarizable model, and be explicit that you did.

A dipole trajectory for a worked example has to be **bounded** — that is the
whole point of the section on electrolytes below. An Ornstein–Uhlenbeck process
is the simplest thing that both fluctuates around zero and has a known
relaxation time, which makes every number here checkable:

```python
import numpy as np
from molpy.compute import Dielectric

rng = np.random.default_rng(0)
n_frames, dt, tau_true = 20000, 10.0, 500.0        # frames, fs, fs
decay = np.exp(-dt / tau_true)

M = np.zeros((n_frames, 3))
for i in range(1, n_frames):                        # bounded, not a random walk
    M[i] = decay * M[i - 1] + np.sqrt(1 - decay**2) * rng.normal(0.0, 0.6, 3)
M = np.ascontiguousarray(M)

volume, temperature = 2.69e4, 298.15                # Å³, K
eps_static = Dielectric.static_dielectric_constant(
    M, volume=volume, temperature=temperature, epsilon_inf=1.0
)
print(round(float(eps_static), 3))                  # -> 1.1
```

Note `epsilon_inf` is passed by name. It is the electronic polarizability a
fixed-charge force field cannot represent; 1.0 is the right value for a
non-polarizable model, and you should say in your methods section that you used
it.

The 1.1 is not water — the dipole amplitude here is arbitrary. Real SPC/E water
gives about 71, and getting it needs nanoseconds, because $\varepsilon(0)$ is a
*variance* and variances converge slowly. A run whose $g(r)$ is beautifully
converged can be nowhere near converged in $\varepsilon(0)$.

Which is exactly why the halves test below is worth running, and why this
example passes it:

```python
half = n_frames // 2
first = Dielectric.static_dielectric_constant(
    np.ascontiguousarray(M[:half]), volume=volume,
    temperature=temperature, epsilon_inf=1.0
)
second = Dielectric.static_dielectric_constant(
    np.ascontiguousarray(M[half:]), volume=volume,
    temperature=temperature, epsilon_inf=1.0
)
print(round(float(first), 3), round(float(second), 3))   # -> 1.096 1.103
```

Under a percent apart. Run the identical test on a `np.cumsum` random walk and
the two halves disagree wildly and grow with length — the pathology this page
attributes to un-decomposed ions.

Two practical notes. Compute the variance in two passes (mean first, then
deviations) rather than as $\langle M^2\rangle - \langle M\rangle^2$, which
cancels catastrophically when the mean is large. And check the three Cartesian
components separately: in an isotropic liquid they must agree, and a persistent
difference means the box is not equilibrated or a field is leaking in from
somewhere.

## The spectrum, by two routes

The frequency-resolved response comes from the *time correlation* of
$\mathbf{M}$ rather than its variance. There are two equivalent routes, and
which you use depends on what your system contains.

**Einstein–Helfand** works from the dipole autocorrelation
$\Phi(t) = \langle\mathbf{M}(0)\cdot\mathbf{M}(t)\rangle$. It is the natural
route for a neutral polar liquid.

**Green–Kubo** works from the current-density autocorrelation, with
$\mathbf{J} = \dot{\mathbf{M}}/V$. It is the natural route when charges migrate,
because a current is what they produce.

The subtlety worth internalizing: in a numerical implementation you transform
the *derivative* of the correlation function rather than the function itself.
$\Phi(t)$ does not decay to zero within any affordable window, and a truncated
non-decaying signal produces the ringing described on the [Signal](signal.md)
page. Its derivative does decay, so transforming that and anchoring the DC value
separately is numerically far better behaved. The computes already do this; it
explains why the API asks for an ACF and hands back a spectrum rather than
exposing an intermediate you might be tempted to plot.

```python
from molpy.compute import DebyeRelaxation, EinsteinHelfandSpectrum

relax = DebyeRelaxation(volume, temperature).compute(
    M, dt=dt, max_correlation_time=100      # lags in FRAMES, so 1 ps here
)
print(sorted(relax))
# -> ['acf', 'boundary', 'lag_times', 'temperature', 'volume', 'zero_lag_variance']

spectrum = EinsteinHelfandSpectrum(
    dt,                            # frame spacing, fs
    volume,                        # Å³
    temperature,                   # K
    1.0,                           # epsilon_inf
    relax["zero_lag_variance"],    # from the DebyeRelaxation result
).fit(relax["acf"])
print(sorted(spectrum))                   # -> ['eps_imag', 'eps_real', 'frequencies']
```

`EinsteinHelfandSpectrum` takes five bare positional arguments and it is worth
naming them in a comment as above, because transposing `volume` and
`temperature` produces a plausible-looking spectrum that is wrong by orders of
magnitude.

`max_correlation_time` is a count of **frames**, not a time — 100 frames at
`dt = 10` fs is a 1 ps window.

`frequencies` are in **rad fs⁻¹** when `dt` is in fs. To get to the wavenumbers
an experimentalist quotes, divide by $2\pi c$: multiply by
$10^{15}/(2\pi \times 2.998\times10^{10})$ to reach cm⁻¹. `eps_real` is the
in-phase response and `eps_imag` the loss — the part that dissipates energy.

## What the spectrum looks like

The simplest model of relaxation, and the shape you should be able to recognize
instantly, is **Debye**: a single exponential decay of $\Phi(t)$ with time
constant $\tau$, giving

$$
\varepsilon^*(\omega) = \varepsilon_\infty +
\frac{\varepsilon(0)-\varepsilon_\infty}{1 + i\omega\tau}.
$$

<figure id="fig-debye-spectrum" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
config:
  legend:
    orient: bottom
    direction: horizontal
    title: null
data: {$file: data/dielectric/debye_reference.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: omega
    type: quantitative
    scale: {type: log}
    title: "ω (arb.)"
  y:
    field: eps
    type: quantitative
    title: "ε′, ε″"
  color:
    field: part
    type: nominal
    title: null
```

</div>

**Figure 1.** The closed-form Debye response, with $\varepsilon(0) = 54$,
$\varepsilon_\infty = 1$, $\tau = 6.5$. Unlike every other figure in these pages
this is **not a measurement** — it is the analytic formula evaluated on a grid,
shown so the shape is recognizable when you meet it in your own data.
</figure>

Three things to read off, because they generalize far beyond the Debye model.
$\varepsilon'$ **steps down** from $\varepsilon(0)$ to $\varepsilon_\infty$: the
liquid stops responding as the field outruns the molecules. $\varepsilon''$
**peaks**, and it peaks exactly at $\omega\tau = 1$ — so the loss maximum
locates the relaxation time by inspection. And the step in $\varepsilon'$ and the
peak in $\varepsilon''$ are not independent; they are linked by the
Kramers–Kronig relations, which is a strong consistency check on any measured
spectrum.

Real liquids are broader than Debye. Water is close to Debye with
$\tau \approx 8.3$ ps at 25 °C, which by $\omega\tau = 1$ puts its loss peak at
$1/(2\pi\tau) \approx 19$ GHz, plus faster processes above it; most other
liquids need Cole–Cole,
Cole–Davidson, or Havriliak–Negami forms, which are the same expression with
one or two stretching exponents added. Fitting those is deliberately *not* part
of the compute layer — it is a `scipy.optimize` recipe in your analysis script,
because the choice of model is a physical claim you should be making explicitly.

For a quick time-domain estimate, fit the normalized ACF directly:

```python
from molpy.compute import DebyeFit

phi = np.asarray(relax["acf"]) / np.asarray(relax["acf"])[0]
fit = DebyeFit().fit(phi, dt=dt)
print(sorted(fit))                        # -> ['amplitude', 'n_samples', 'tau']
print(round(float(fit["tau"])))           # -> 516
```

516 fs against the 500 fs built into the process — a 3 % recovery, and the
check that the whole chain is wired correctly. `tau` comes back in the same
units as `dt`.

Now widen the correlation window and watch it break:

```python
long_window = DebyeRelaxation(volume, temperature).compute(
    M, dt=dt, max_correlation_time=300      # 3 ps, six relaxation times
)
long_phi = np.asarray(long_window["acf"]) / np.asarray(long_window["acf"])[0]
print(round(float(DebyeFit().fit(long_phi, dt=dt)["tau"])))   # -> 909
```

Nearly double the true value, from *more* data. Past about 2 ps the correlation
function is buried in noise, but it is still fitted with equal weight, and the
noisy tail drags the exponential out. Fit the window where the signal is, not
everything you have — the same discipline as the fitting window on
[MSD](msd.md) and the plateau on [VACF](vacf.md).

## Electrolytes: decompose before you interpret

This is where dielectric analysis most often goes wrong, and the error is
conceptual rather than numerical.

In a solution containing ions, $\mathbf{M}(t)$ has two contributions that behave
completely differently. Solvent molecules **rotate**, and their dipole
contribution decorrelates and settles — that is dielectric relaxation. Ions
**translate**, and their contribution to $\mathbf{M}$ grows without bound,
diffusing like the collective charge displacement on the [PMSD](pmsd.md) page.
That is conductivity, not permittivity.

Feed the combined $\mathbf{M}$ into the Neumann formula and the variance is
dominated by unbounded ionic drift, producing an $\varepsilon(0)$ that grows
with trajectory length — a number with no physical meaning that nonetheless
looks like a converging quantity if you do not plot it against time.

So **split the dipole first**: solvent series into the static formula and the
dielectric spectrum, ion series into the conductivity routes on
[PMSD](pmsd.md) and [JACF](jacf.md).

The split is by atom, so you can do it directly from your charges and a mask
naming which atoms belong to ions:

```python
charges = np.array([-0.8476, 0.4238, 0.4238, 1.0, -1.0])   # water + Na+ + Cl-
is_ion = np.array([False, False, False, True, True])
positions = rng.normal(0.0, 5.0, size=(200, 5, 3))          # (frames, atoms, 3)

def dipole(pos, q, mask):
    """Sum q_i r_i over the selected atoms, per frame."""
    return np.einsum("i,tij->tj", q[mask], pos[:, mask, :])

M_solvent = dipole(positions, charges, ~is_ion)
M_ions = dipole(positions, charges, is_ion)
print(M_solvent.shape, M_ions.shape)          # -> (200, 3) (200, 3)
```

`M_solvent` goes into `static_dielectric_constant` and the spectrum;
`M_ions` goes into [`EinsteinConductivity`](pmsd.md).
`Dielectric.compute_dipole_moment` does the same sum for a whole frame, and
`Dielectric.decompose_current` splits an already-assembled current series the
same way.

The positions above must be **unwrapped per molecule**, as described earlier —
that is the step this snippet assumes you have already done, and the one most
likely to be wrong.

Even after splitting, the two remain coupled in the measured spectrum: a
conductive sample contributes $\sigma/(\epsilon_{\text{vac}}\omega)$ to $\varepsilon''$,
which diverges as $\omega \to 0$ and buries the relaxation peak. Subtract that
term using an independently determined $\sigma$ before fitting relaxation times,
and say that you did.

Be honest about uncertainty too. A simulation box typically holds tens of ions,
not Avogadro's number, so $\sigma$ from an MD box carries error bars of tens of
percent. Quote them.

## The pieces, and what feeds what

| Step | Entry point |
|---|---|
| Dipole $\mathbf{M}=\sum q_i\mathbf{r}_i$ | `Dielectric.compute_dipole_moment` |
| Current density $\mathbf{J}=\dot{\mathbf{M}}/V$ | `Dielectric.compute_current_density` |
| Split solvent / ion series | `Dielectric.decompose_current` |
| Static $\varepsilon(0)$ | `Dielectric.static_dielectric_constant` |
| Raw dipole ACF | `DebyeRelaxation` |
| $\varepsilon^*(\omega)$, dipole route | `EinsteinHelfandSpectrum` |
| Raw current ACF | `GreenKuboConductivity` |
| $\varepsilon^*(\omega)$, current route | `GreenKuboSpectrum` |
| Time-domain $\tau$ | `DebyeFit` |
| DC $\sigma$ | [`EinsteinConductivity`](pmsd.md) → `LinearFit` |

One trap in that table. `GreenKuboSpectrum` needs the ACF of the **current
density** $\mathbf{J} = \dot{\mathbf{M}}/V$, while the DC conductivity on
[JACF](jacf.md) needs the ACF of the **total current** $\dot{\mathbf{M}}$. The
two currents differ by a factor $V$ — but a correlation function is *quadratic*
in its input, so the two ACFs differ by $V^2$. For the 2.69 × 10⁴ Å³ box used
above that is $7.2\times10^{8}$, nine orders of magnitude. Feed one where the
other belongs and the answer is not subtly wrong, it is unrecognizable, which is
at least easy to spot.

!!! note "No measured spectrum on this page yet — TODO"
    Figure 1 is the analytic Debye form, not data. A real
    $\varepsilon^*(\omega)$ needs a polar molecular liquid with charges, run for
    nanoseconds with Ewald electrostatics; the reference trajectory behind these
    pages is monatomic argon, for which $\mathbf{M} \equiv 0$. Add a measured
    spectrum when a polar reference trajectory exists under
    `scripts/docs_data/`.

## When it goes wrong

**$\varepsilon(0)$ keeps growing as you add trajectory.**
Ions are in your dipole. Decompose. If there are no ions, the run is simply not
converged — plot $\varepsilon(0)$ against trajectory length and see whether it
is drifting or wandering.

**$\varepsilon(0)$ is wildly too large, with jumps in $\mathbf{M}(t)$.**
Molecules are being unwrapped atom-by-atom, or not at all. Plot $\mathbf{M}(t)$
directly and look for box-sized steps.

**$\varepsilon''$ diverges at low frequency.**
The conductivity contribution. Expected in an electrolyte; subtract
$\sigma/(\epsilon_{\text{vac}}\omega)$.

**The spectrum is covered in ripples.**
Spectral leakage from an ACF truncated before it decayed. See
[Signal](signal.md).

**$\varepsilon(0)$ from the fluctuation formula disagrees with the $\omega\to 0$
limit of your spectrum.**
They must agree — it is the same quantity. A mismatch usually means the DC
anchoring of the spectrum used a different $\varepsilon_\infty$, or the ACF was
truncated.

**The per-axis components differ.**
The system is not isotropic or not equilibrated. Check before averaging them.

**$\tau$ from `DebyeFit` disagrees with the loss-peak position.**
Only equal for a genuinely single-exponential process. A discrepancy is
information: your relaxation is not Debye.

## Check yourself

- Plot $\mathbf{M}(t)$ before anything else. It should look like a bounded
  fluctuating signal for a neutral liquid, with no steps.
- Compute $\varepsilon(0)$ on the first and second halves of your trajectory
  separately. If they differ by more than a few percent, you are not converged.
- Check $\varepsilon'(\omega \to 0)$ against the fluctuation-formula
  $\varepsilon(0)$.
- Confirm the $\varepsilon''$ peak sits at $\omega = 1/\tau$ for the fitted
  $\tau$.

## References

- M. Neumann, *Mol. Phys.* **50**, 841 (1983) — the fluctuation formula and
  boundary-condition dependence.
- D. Braun, S. Boresch, O. Steinhauser, *J. Chem. Phys.* **140**, 064107 (2014)
  — computing dielectric spectra from MD, including the derivative trick.
- C. Schröder, O. Steinhauser, *J. Chem. Phys.* **132**, 244109 (2010) —
  separating dielectric and conductive contributions in electrolytes.
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed. (2013),
  ch. 11 — linear response and the fluctuation–dissipation theorem.

## See also

- [PMSD](pmsd.md) · [JACF](jacf.md) — the conductivity half, in full
- [Reorientation](reorientation.md) — $C_1$, the single-molecule counterpart
- [Signal](signal.md) — windowing and the frequency grid
- [Spectra](spectra.md) — the same machinery at vibrational frequencies
- [API reference](../api/compute.md)
