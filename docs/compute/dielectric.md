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

`StaticDielectric` is configured with the unit system the dipoles and volume
are in — `"real"` here: dipoles in e·Å, volume in Å³ — and computed on the
`(n_frames, 3)` dipole series together with the state it was sampled at:
volume, temperature, and the Ewald boundary of the simulation.

```python
import numpy as np
from molpy.compute import StaticDielectric

rng = np.random.default_rng(0)
n_frames, dt, tau_true = 20000, 10.0, 500.0        # frames, fs, fs
decay = np.exp(-dt / tau_true)

M = np.zeros((n_frames, 3))
for i in range(1, n_frames):                        # bounded, not a random walk
    M[i] = decay * M[i - 1] + np.sqrt(1 - decay**2) * rng.normal(0.0, 0.6, 3)
M = np.ascontiguousarray(M)

volume, temperature = 2.69e4, 298.15                # Å³, K
state = {"volume": volume, "temperature": temperature, "boundary": "tin_foil"}

static = StaticDielectric("real").compute(dipoles=M, **state)
print(round(static.epsilon, 3), static.epsilon_inf)  # -> 1.1 1.0
```

`epsilon_inf` defaults to 1.0, the right value for a non-polarizable model; a
polarizable one configures it, `StaticDielectric("real", epsilon_inf=1.8)`.
Either way, say in your methods section which you used. The boundary is
`"tin_foil"` because that is what the fluctuation formula above assumes;
`"vacuum"` is refused with a `ValueError` rather than silently treated as
tin-foil, because the vacuum-boundary formula is a different expression.

The 1.1 is not water — the dipole amplitude here is arbitrary. Real SPC/E water
gives about 71, and getting it needs nanoseconds, because $\varepsilon(0)$ is a
*variance* and variances converge slowly. A run whose $g(r)$ is beautifully
converged can be nowhere near converged in $\varepsilon(0)$.

Which is exactly why the halves test below is worth running, and why this
example passes it:

```python
half = n_frames // 2
static_eps = StaticDielectric("real")
first = static_eps.compute(dipoles=M[:half], **state).epsilon
second = static_eps.compute(dipoles=M[half:], **state).epsilon
print(round(first, 3), round(second, 3))            # -> 1.096 1.103
```

Under a percent apart. Run the identical test on a `np.cumsum` random walk and
the two halves disagree wildly and grow with length — the pathology this page
attributes to un-decomposed ions.

Two practical notes. `StaticDielectric` computes the variance in two passes
(mean first, then deviations) rather than as
$\langle M^2\rangle - \langle M\rangle^2$, which cancels catastrophically
when the mean is large; do the same in any variance you compute yourself. And
check the three Cartesian
components separately — `epsilon_axes` holds them: in an isotropic liquid they
must agree, and a persistent difference means the box is not equilibrated or a
field is leaking in from somewhere.

```python
print(static.epsilon_axes.round(2))                 # -> [1.11 1.09 1.1 ]
```

## The spectrum, by two routes

The frequency-resolved response comes from the *time correlation* of
$\mathbf{M}$ rather than its variance. There are two equivalent routes, and
which you use depends on what your system contains.

**Einstein–Helfand** works from the fluctuation dipole autocorrelation
$\Phi(t) = \langle\delta\mathbf{M}(0)\cdot\delta\mathbf{M}(t)\rangle$. It is the
natural route for a neutral polar liquid.

**Green–Kubo** works from the current-density autocorrelation, with
$\mathbf{J} = \dot{\mathbf{M}}/V$. It is the natural route when charges migrate,
because a current is what they produce.

The subtlety worth internalizing: in a numerical implementation you transform
the *derivative* of the correlation function rather than the function itself.
$\Phi(t)$ does not decay to zero within any affordable window, and a truncated
non-decaying signal produces the ringing described on the [Spectra](spectra.md)
page. Its derivative does decay, so transforming that and anchoring the DC value
separately at the exact static limit is numerically far better behaved.

Each route is two computes: a correlation, and a transform that reads it.
`DipoleAcf` is the dipole autocorrelation — mean-subtracted, summed over the
three Cartesian components, unbiased. `EinsteinHelfandSpectrum` is configured
with the unit system (and optionally the window, the zero padding and
$\varepsilon_\infty$), and computes on the correlation plus the same state the
static formula took:

```python
from molpy.compute import DipoleAcf, EinsteinHelfandSpectrum

phi = DipoleAcf(100).compute(dipoles=M, dt=dt)      # lags in FRAMES: 1 ps here
print(phi.n_lags, phi.lag_times[-1])                # -> 101 1000.0

spectrum = EinsteinHelfandSpectrum("real").compute(correlation=phi, **state)
print(spectrum.frequencies.shape, round(spectrum.eps_real[0], 3))   # -> (257,) 1.1
```

`max_lag` is a count of **frames**, not a time — 100 frames at `dt = 10` fs is
a 1 ps window. The transform reads the time step off the correlation's own lag
grid, so there is no second `dt` to get wrong. The zero-frequency value is the
static $\varepsilon(0)$ from the fluctuation formula, exactly: the two routes
share one anchor. `DipoleAcf(..., normalization="none")` selects the raw lagged
sum of the older convention instead of the unbiased estimator; the default is
the one the spectrum routes are written for.

`frequencies` are angular, in radians per unit of `dt` — rad fs⁻¹ here. To get
to the wavenumbers an experimentalist quotes, divide by $2\pi c$ (see
[Spectra](spectra.md)). `eps_real` is the in-phase response and `eps_imag` the
loss — the part that dissipates energy, stored positive.

The same dipole correlation feeds a second dipole route,
`DipoleAutocorrelationSpectrum`, which transforms $\Phi$ itself through
$\chi(\omega) = A\,[\Phi(0) - i\omega\hat\Phi(\omega)]$ rather than its
derivative. For an ionic conductor $\Phi$ does not decay at all — it plateaus,
because the cell dipole drifts with the ions — and the bare formula then
produces a huge false static response. Configured with a `plateau=(start, end)`
window (fractions of the lag range), the route subtracts the plateau value
$\Phi(\infty)$ first; `plateau_value="mean"` (the default) takes the window's
mean, `"end"` its last sample. Whether to subtract is your decision, stated in
your script; the route never guesses it from the data.

The two current routes are the same composition with a different correlation.
`Derivative` turns the dipole series into its rate $\dot{\mathbf M}$;
`CurrentAcf` of $\dot{\mathbf M}/V$ feeds `GreenKuboSpectrum`, and the
cross-correlation of $\dot{\mathbf M}$ against $\mathbf M$ — a `Correlator` with
the product kernel, a second series `b`, and the mean subtracted — feeds
`DipoleRateCrossSpectrum`:

```python
from molpy.compute import (
    Correlator,
    CurrentAcf,
    Derivative,
    DipoleAutocorrelationSpectrum,
    DipoleRateCrossSpectrum,
    GreenKuboSpectrum,
)

rate = Derivative("gradient").compute(series=M, dt=dt)            # dM/dt
current_acf = CurrentAcf(100).compute(current=rate / volume, dt=dt)
rate_cross = Correlator("product", 100, mean="subtract").compute(
    a=rate[:, None, :], b=M[:, None, :], dt=dt
)

routes = {
    "einstein-helfand": spectrum,
    "dipole acf": DipoleAutocorrelationSpectrum("real").compute(
        correlation=phi, **state
    ),
    "green-kubo": GreenKuboSpectrum("real").compute(correlation=current_acf, **state),
    "rate cross": DipoleRateCrossSpectrum("real").compute(
        correlation=rate_cross, **state
    ),
}
print(len({r.frequencies.shape for r in routes.values()}))       # -> 1
```

## A spectrum is checked, not trusted

Four routes to one quantity is the strongest check there is: on a converged
trajectory they must agree. `RouteAgreement` compares one part of several
spectra on the same frequency grid — the loss, by default — and passes when
every pairwise relative RMS difference is under its `tolerance` (0.1):

```python
from molpy.compute import KramersKronig, RouteAgreement

dipole_routes = [routes["einstein-helfand"], routes["dipole acf"]]
check = RouteAgreement().compute(spectra=dipole_routes)
print(check.passed, round(check.max_rms, 2))        # -> False 0.14
```

The two dipole routes disagree by 14 %, and the check is right to fail. The
Ornstein–Uhlenbeck dipole is a Markov process: its correlation
$e^{-|t|/\tau}$ has a cusp at $t = 0$, while the Einstein–Helfand route assumes
$\Phi'(0) = 0$, which holds for every real dipole because a physical
correlation function is even and smooth. A real liquid passes where this toy
does not. The current routes need a longer window still: $\sigma(\omega)/\omega$
is noisy at the lowest frequencies, and their agreement improves as `max_lag`
grows.

The other check needs only one spectrum. $\varepsilon'$ and $\varepsilon''$ are
not independent — causality ties them through the Kramers–Kronig relations —
so `KramersKronig` recovers $\varepsilon'$ from $\varepsilon''$ and compares:

```python
kk = KramersKronig().compute(spectrum=spectrum)
print(kk.passed, kk.mae < kk.tolerance)             # -> True True
```

A spectrum that fails it was truncated, mis-anchored, or built from the wrong
correlation. For a conductor the third check is the conductivity sum rule,
$\int_0^\infty\sigma(\omega)\,\mathrm{d}\omega = \tfrac{\pi}{2}\langle J^2\rangle/(3Vk_BT)$,
`ConductivitySumRule("real")`, computed on a frequency grid and the real
conductivity $\sigma'(\omega)$ on it, with $\langle J^2\rangle$, the volume and
the temperature.

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
data: {$file: series/dielectric/debye_reference.json}
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

For a quick time-domain estimate, fit a single exponential
$\Phi(t) = A\,e^{-t/\tau}$ to the dipole correlation directly. `DebyeFit` reads
one channel of a correlation and its lag times:

```python
from molpy.compute import DebyeFit

fit = DebyeFit().compute(correlation=phi, channel=0)
print(round(fit.tau), fit.n_samples)      # -> 516 101
```

516 fs against the 500 fs built into the process — a 3 % recovery, and the
check that the whole chain is wired correctly. `tau` comes back in the units of
the lag times, which are the units of `dt`.

Now widen the correlation window and watch it break:

```python
long_window = DipoleAcf(300).compute(dipoles=M, dt=dt)   # 3 ps, six relaxation times
print(round(DebyeFit().compute(correlation=long_window, channel=0).tau))   # -> 909
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

The split is by atom. `DipoleMoment` sums $q_i\mathbf{r}_i$ for every frame;
given a partition — one label per atom, here `0` for solvent and `1` for ions —
it sums each group separately, so both series come out of one call. The
charges and the labels are columns you read from your frame's atoms block:

```python
from molpy.compute import DipoleMoment

charges = np.array([-0.8476, 0.4238, 0.4238, 1.0, -1.0])   # water + Na+ + Cl-
species = np.array([0, 0, 0, 1, 1], dtype=np.uint32)       # 0 solvent, 1 ion
positions = rng.normal(0.0, 5.0, size=(200, 5, 3))          # (frames, atoms, 3)

parts = DipoleMoment().compute(positions=positions, charges=charges, partition=species)
M_solvent, M_ions = parts.series(0), parts.series(1)
print(parts.groups, M_solvent.shape, M_ions.shape)   # -> [0 1] (200, 3) (200, 3)
```

`M_solvent` goes into `StaticDielectric` and the spectrum routes; `M_ions` goes
into the conductivity routes, the [MSD of the collective dipole](pmsd.md) or
the [current correlation](jacf.md). Without a partition `DipoleMoment` sums the
whole system into one series. Any label array works as the partition — a
`mol_id` column, a species column, or the result of a [Cluster](cluster.md)
analysis.

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

| Step | Compute |
|---|---|
| Dipole $\mathbf{M}=\sum q_i\mathbf{r}_i$, whole or per group | `DipoleMoment` |
| Rate $\dot{\mathbf{M}}$ (divide by $V$ for $\mathbf{J}$) | `Derivative` |
| Static $\varepsilon(0)$ | `StaticDielectric` |
| Dipole ACF | `DipoleAcf` |
| $\varepsilon^*(\omega)$, dipole routes | `EinsteinHelfandSpectrum`, `DipoleAutocorrelationSpectrum` |
| Current ACF | `CurrentAcf` |
| $\varepsilon^*(\omega)$, current routes | `GreenKuboSpectrum`; `DipoleRateCrossSpectrum` on a `Correlator` cross |
| Checks | `KramersKronig`, `RouteAgreement`, `ConductivitySumRule` |
| Time-domain $\tau$ | `DebyeFit` |
| DC $\sigma$ | [`Msd`](pmsd.md) of the ion dipole → `LinearFit`, or [`CurrentAcf`](jacf.md) → `CumulativeTrapezoid` |

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
    `molpy/scripts/docs_series/`.

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
[Spectra](spectra.md) on windows.

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
- Run `KramersKronig` on every spectrum, and `RouteAgreement` across the routes
  you can compute.

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
- [Spectra](spectra.md) — the same machinery at vibrational frequencies
- [API reference](../api/compute.md)
