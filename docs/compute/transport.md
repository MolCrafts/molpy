# Diffusion & Ionic Transport

This page is a self-contained, textbook-style introduction to how MolPy turns an
equilibrium molecular-dynamics (MD) trajectory into **transport coefficients** —
diffusion coefficients, Onsager phenomenological coefficients, and ionic
conductivity. It starts from the random walk and builds up to the collective
correlation functions used in modern electrolyte analysis. No background beyond
undergraduate statistical mechanics and a little linear algebra is assumed.

It is the conceptual companion to the [Dielectric Spectroscopy](dielectric.md)
guide: that page derives the *frequency-dependent* response and the two
conductivity routes in depth; this page focuses on *diffusion* and the
*displacement* picture, and points back to the dielectric page for the spectral
machinery.

As with all MolPy analyses, the heavy numerics run in Rust (`molrs`); the MolPy
layer extracts coordinates, unwraps periodic images, builds the collective
quantities, and returns typed result objects.

!!! note "Conventions used throughout"
    - Position of atom $i$ at time $t$: $\mathbf{r}_i(t)$; displacement over a lag
      $\tau$: $\Delta\mathbf{r}_i(\tau) = \mathbf{r}_i(t+\tau) - \mathbf{r}_i(t)$.
    - $\langle\cdots\rangle_t$ denotes an average over **time origins** $t$.
    - Units (LAMMPS *real*): length Å, time ps, charge $e$, volume Å³,
      temperature K. Diffusion coefficients come out in Å²·ps⁻¹; conductivity in
      S·m⁻¹.
    - $d = 3$ spatial dimensions; the Einstein factor $1/(2d) = 1/6$.

---

## 1. The random walk and the Einstein relation

A particle in a liquid is kicked around by its neighbours. Over a short time it
moves *ballistically* (it remembers its velocity), but after many uncorrelated
kicks its motion becomes a **random walk**: the *direction* is forgotten while
the *spread* keeps growing. The natural measure of that spread is the
**mean-squared displacement (MSD)**:

$$
\mathrm{MSD}(\tau) = \big\langle\,|\mathbf{r}_i(t+\tau) - \mathbf{r}_i(t)|^2\,\big\rangle_{i,t}.
$$

For a diffusive process the MSD grows **linearly** in time, and the slope defines
the **self-diffusion coefficient** $D$ (Einstein, 1905):

$$
\boxed{\;D = \lim_{\tau\to\infty}\frac{1}{2d\,\tau}\,\mathrm{MSD}(\tau)
       = \lim_{\tau\to\infty}\frac{1}{6\tau}\big\langle|\Delta\mathbf{r}(\tau)|^2\big\rangle\;}
$$

The factor $1/6 = 1/(2d)$ with $d=3$ counts the three independent directions the
walker can spread into.

### 1.1 The three regimes

A real MSD curve has three parts, and only the middle one is physical diffusion:

- **Ballistic** (short $\tau$): $\mathrm{MSD}\propto\tau^2$ — the particle still
  moves at roughly constant velocity.
- **Diffusive** (intermediate $\tau$): $\mathrm{MSD}\propto\tau$ — the linear
  regime; **fit the slope here**.
- **Noisy** (long $\tau$): few time origins remain, so the estimate is dominated
  by statistical scatter.

Choosing the linear window is the single most important judgement call in a
diffusion calculation. MolPy's transport computes expose `fit_min`/`fit_max` (or
fraction-based windows) for exactly this reason.

### 1.2 Averaging over time origins

At equilibrium the dynamics are stationary, so every frame can serve as a time
origin $t$. Averaging over **all** origins (a "windowed" MSD) uses the data far
more efficiently than measuring displacement from frame 0 only:

$$
\mathrm{MSD}(\tau) = \frac{1}{N_\text{origins}}\sum_{t} |\mathbf{r}(t+\tau)-\mathbf{r}(t)|^2,
\qquad N_\text{origins} = N_\text{frames}-\tau.
$$

The number of usable origins shrinks as $\tau$ grows — which is exactly why the
long-$\tau$ tail is noisy.

### 1.3 Minimum-image unwrapping (the usual trap)

MD runs in a periodic box: when an atom leaves one face it re-enters the
opposite one, so its stored coordinate jumps by a box length $L$. A naive MSD
built from such coordinates registers a spurious $L$-sized displacement at every
crossing. The fix is to accumulate **minimum-image** steps,

$$
\Delta\mathbf{r}_k = \mathbf{r}(t_k)-\mathbf{r}(t_{k-1}) - L\,\mathrm{round}\!\Big(\tfrac{\mathbf{r}(t_k)-\mathbf{r}(t_{k-1})}{L}\Big),
\qquad
\mathbf{r}_\text{unwrap}(t_k) = \mathbf{r}_\text{unwrap}(t_{k-1}) + \Delta\mathbf{r}_k,
$$

valid as long as a particle moves less than $L/2$ per frame. MolPy does this with
`Box.delta(p1, p2, minimum_image=True)`; it is shared by every displacement-based
transport compute. (See [Dielectric §2.1](dielectric.md#21-a-critical-detail-minimum-image-unwrapping)
for the same argument in the dipole context.)

---

## 2. Self vs distinct diffusion: the MDC

A single diffusion coefficient hides a lot of physics. In a multi-component
system the displacements of *different* particles are correlated — ions drag
their counter-ions, solvent flows around them. The **mean displacement
correlation (MDC)** generalises the MSD to expose this.[^gudla]

**Self (tag `"3"`)** — the ordinary MSD of one species, giving the self-diffusion
coefficient $D^\mathrm{s}_\alpha$:

$$
D^\mathrm{s}_{\alpha} = \lim_{\tau\to\infty}\frac{1}{6\tau N}
   \sum_i\big\langle|\Delta\mathbf{r}_{i,\alpha}(\tau)|^2\big\rangle.
$$

**Distinct (tag `"3,4"`)** — the *cross*-correlation between the displacements of
species $\alpha$ and $\beta$, giving the distinct-diffusion coefficient
$D^\mathrm{d}_{\alpha\beta}$:

$$
D^\mathrm{d}_{\alpha\beta} = \lim_{\tau\to\infty}\frac{1}{6\tau N}
   \sum_i\sum_{j\ne i}\big\langle\Delta\mathbf{r}_{i,\alpha}(\tau)\cdot\Delta\mathbf{r}_{j,\beta}(\tau)\big\rangle.
$$

The distinct term is a **collective** quantity: it is dominated by how the
species move *together*, not by any single particle.

!!! note "Normalization convention (MolPy vs tame)"
    For different species, MolPy evaluates the distinct term as the collective
    cross-correlation $\big\langle(\sum_i\Delta\mathbf{r}_i)\cdot(\sum_j\Delta\mathbf{r}_j)\big\rangle$
    — the physically meaningful form that feeds directly into the Onsager
    coefficients of [§3](#3-onsager-phenomenological-coefficients). The original
    [tame](https://github.com/Roy-Kid/tame) `mdc` recipe averages (rather than
    sums) over the reference species $i$, i.e. it carries an extra factor
    $1/N_i$; MolPy deliberately uses the un-normalized collective form. For fully
    normalized Onsager coefficients use [`Onsager`](#3-onsager-phenomenological-coefficients).

```python
from molpy.compute import MCDCompute

mdc = MCDCompute(tags=["3", "4", "3,4"], max_dt=20.0, dt=0.01)
result = mdc(trajectory)
result.correlations["3"]    # self MSD of species 3, vs lag time
result.correlations["3,4"]  # distinct cross-correlation of 3 and 4
```

---

## 3. Onsager phenomenological coefficients

The cleanest way to describe coupled transport in an electrolyte is Onsager's
**phenomenological coefficients** $\Omega_{\alpha\beta}$ (also written
$L_{\alpha\beta}$). They are the proper, normalized version of the collective
displacement correlation: $\Omega_{\alpha\beta}$ relates the flux of species
$\alpha$ to the thermodynamic driving force on species $\beta$.

Define the **collective coordinate** of a species — the summed (unwrapped)
position of all its atoms,

$$
\mathbf{P}_\alpha(t) = \sum_{i\in\alpha}\mathbf{r}_i(t),
\qquad
\Delta\mathbf{P}_\alpha(\tau) = \mathbf{P}_\alpha(t+\tau)-\mathbf{P}_\alpha(t).
$$

The collective displacement correlation and the Onsager coefficient are then

$$
\mathrm{corr}_{\alpha\beta}(\tau) = \big\langle\,\Delta\mathbf{P}_\alpha(\tau)\cdot\Delta\mathbf{P}_\beta(\tau)\,\big\rangle_t,
\qquad
\boxed{\;\Omega_{\alpha\beta} = \lim_{\tau\to\infty}\frac{\mathrm{corr}_{\alpha\beta}(\tau)}{6\,k_B T\,V\,N_A\,\tau}\;}
$$

- The **diagonal** $\Omega_{\alpha\alpha}$ is the collective MSD of species
  $\alpha$ — it contains the self term **plus** the like-ion cross terms.
- The **off-diagonal** $\Omega_{\alpha\beta}$ ($\alpha\ne\beta$) captures
  cation–anion coupling. A negative value (anticorrelated drift) is the signature
  of ion pairing.

`Onsager` returns the correlation curves $\mathrm{corr}_{\alpha\beta}(\tau)$; you
take the long-time slope and apply the $1/(6 k_B T V N_A)$ prefactor for the
coefficient itself.

```python
from molpy.compute import Onsager

ons = Onsager(tags=["1,1", "1,2", "2,2"], max_dt=20.0, dt=0.01)
result = ons(trajectory)
result.correlations["1,2"]  # cation-anion collective correlation L_12(tau)
```

### 3.1 From Onsager coefficients to conductivity

The ionic conductivity is a weighted sum of the Onsager coefficients over the
ion charges $z_\alpha$:

$$
\sigma = \frac{e^2}{V k_B T}\sum_{\alpha\beta} z_\alpha z_\beta\,\Omega_{\alpha\beta}.
$$

If the off-diagonal (distinct) terms vanish — ions moving independently — this
collapses to the **Nernst–Einstein** estimate $\sigma_\text{NE}$ built from the
self-diffusion coefficients alone. The ratio $\sigma/\sigma_\text{NE}$ (the
*ionicity* or *Haven ratio*) measures how strongly ion correlations suppress (or
enhance) conduction — a number you cannot get from single-particle diffusion
alone, which is the whole reason the Onsager picture exists.

---

## 4. Ionic conductivity: the two equivalent routes

Conductivity can be obtained from the **collective charge transport** directly,
without going through individual $\Omega_{\alpha\beta}$. There are two equivalent
routes (a general consequence of the fluctuation–dissipation theorem; see
[Dielectric §1.3](dielectric.md#13-the-fluctuationdissipation-theorem)).

### 4.1 Einstein route — polarization MSD (PMSD)

Build the **collective charge displacement** (a.k.a. the translational dipole)
of the ions,

$$
\mathbf{P}(t) = \sum_\text{cations}\mathbf{r}_i(t) - \sum_\text{anions}\mathbf{r}_j(t),
$$

and measure its MSD. Its long-time slope gives the conductivity by an Einstein
relation:

$$
\mathrm{PMSD}(\tau) = \big\langle|\mathbf{P}(t+\tau)-\mathbf{P}(t)|^2\big\rangle_t,
\qquad
\sigma = \lim_{\tau\to\infty}\frac{1}{6\,V k_B T}\,\frac{d}{d\tau}\,\mathrm{PMSD}(\tau).
$$

```python
from molpy.compute import PMSDCompute, IonicConductivity

# The PMSD curve itself:
pmsd = PMSDCompute(cation_type=1, anion_type=2, max_dt=30.0, dt=0.01)(trajectory)

# The conductivity (Einstein-Helfand), fitted and converted to S/m:
sigma = IonicConductivity(dt=0.01, temperature=298.15, max_correlation_time=1000)(ion_trajectory)
sigma.sigma  # S/m
```

`PMSDCompute` returns the curve; `IonicConductivity` does the fit and unit
conversion. See [Dielectric §7](dielectric.md#7-ionic-conductivity-einsteinhelfand)
for the full derivation, the diffusive-window caveat, and the SI prefactor.

### 4.2 Green–Kubo route — current autocorrelation (JACF)

Equivalently, integrate the autocorrelation of the **charge current**
$\mathbf{J}(t)=\sum_a q_a\mathbf{v}_a(t)$:

$$
\boxed{\;\sigma = \frac{1}{3\,V k_B T}\int_0^\infty \big\langle\mathbf{J}(0)\cdot\mathbf{J}(t)\big\rangle\,dt\;}
$$

The integrand $C(\tau)=\langle\mathbf{J}(0)\cdot\mathbf{J}(\tau)\rangle$ is the
current autocorrelation function (JACF); the factor $1/3$ is the Green–Kubo
analogue of the Einstein $1/6$ (one comes from integrating the ACF, the other
from differentiating the MSD).

```python
from molpy.compute import JACF

jacf = JACF(cation_type=1, anion_type=2, max_dt=30.0, dt=0.01, temperature=298.15)
result = jacf(trajectory)        # requires per-atom velocities vx, vy, vz
result.jacf                      # <J(0).J(t)>
result.sigma                     # DC conductivity, S/m (the GK integral)
result.sigma_running             # running integral sigma(tau), to check convergence
```

The Einstein (PMSD) and Green–Kubo (JACF) routes are mathematically identical;
in practice the Einstein route is more robust at coarse sampling, while the JACF
exposes the current memory directly and lets you watch the integral converge.
[Dielectric §8](dielectric.md#8-spectrum-route-ii-greenkubo-current-autocorrelation)
derives the frequency-dependent generalisation $\sigma(\omega)$.

---

## 5. Reading the results

| Quantity | Compute | Physical meaning |
|---|---|---|
| $\mathrm{MSD}(\tau)$ / $D^\mathrm{s}$ | `MCDCompute` (single tag) | single-particle diffusion |
| $D^\mathrm{d}_{\alpha\beta}$ | `MCDCompute` (pair tag) | distinct (collective) diffusion |
| $\mathrm{corr}_{\alpha\beta}(\tau)$ / $\Omega_{\alpha\beta}$ | `Onsager` | coupled transport; ion pairing (off-diagonal) |
| $\mathrm{PMSD}(\tau)$ | `PMSDCompute` | collective charge transport |
| $\sigma$ (Einstein) | `IonicConductivity` | DC conductivity, S/m |
| $C(\tau)$, $\sigma$ (Green–Kubo) | `JACF` | current memory + DC conductivity |

**Cross-checks.** The Einstein (`IonicConductivity`/`PMSDCompute`) and
Green–Kubo (`JACF`) conductivities must agree within statistics. The
Nernst–Einstein estimate (from `MCDCompute` self terms) should exceed the
correlated conductivity from `Onsager`/`JACF` when ion pairing is significant —
their ratio is the ionicity.

---

## 6. Pitfalls checklist

1. **No unwrapping** → boundary crossings inject $L$-sized jumps; every MSD is
   garbage. (MolPy unwraps automatically via `Box.delta`.)
2. **Fitting outside the diffusive window** → the ballistic head or the noisy
   tail biases $D$ and $\sigma$. Always inspect the curve first.
3. **Too few carriers / too short a trajectory** → collective quantities
   (PMSD, Onsager off-diagonal, JACF) are intrinsically noisy because there is
   only *one* collective coordinate per species. Report a range, not a digit.
4. **Wrong velocity units in `JACF`** → the current must be in $e\cdot$Å·ps⁻¹
   (velocities in Å/ps); $\sigma$ scales linearly, so a unit slip rescales the
   answer.
5. **Ignoring distinct diffusion** → quoting only Nernst–Einstein conductivity
   ignores ion correlations and typically overestimates $\sigma$.

---

## 7. References

- A. Einstein, *Ann. Phys.* **322**, 549 (1905) — the diffusion/MSD relation.
- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed. (2017)
  — MSD, time-origin averaging, transport coefficients.
- J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids*, 4th ed. — Green–Kubo
  relations and the current correlation function.
- D. Frenkel, B. Smit, *Understanding Molecular Simulation*, 2nd ed. (2002),
  §4.4 — Einstein relations for transport coefficients.
- L. Onsager, *Phys. Rev.* **37**, 405 (1931); **38**, 2265 (1931) —
  reciprocal relations and phenomenological coefficients.

[^gudla]: H. Gudla, Y. Shao et al., *J. Phys. Chem. Lett.* **12**, 8460 (2021) —
    distinct diffusion combined with a persistence function to extract the
    pairing contribution to transport.

## See also

- [Dielectric Spectroscopy](dielectric.md) — the spectral machinery
  ($\varepsilon^*(\omega)$, autocorrelation, FFT) and the full conductivity
  derivations.
- [Pair Persistence](persistence.md) — residence times and the survival
  functions that resolve the *pairing* contribution to diffusion.
- [Compute overview](index.md) — the Compute → Result pattern.
- [API reference: Compute](../api/compute.md).
