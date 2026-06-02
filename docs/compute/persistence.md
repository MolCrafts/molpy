# Pair Persistence & Residence Times

This page is a self-contained, textbook-style introduction to **pair persistence
analysis** — how MolPy measures how long two particles stay "bonded" (within a
distance cutoff) and turns that into a residence-time correlation function. The
canonical applications are hydrogen-bond dynamics in water and ion-pair lifetimes
in electrolytes. No background beyond the idea of a time-correlation function is
assumed; if you have read the [Diffusion & Ionic Transport](transport.md) guide
you already have everything you need.

As always, the per-pair, per-frame bookkeeping runs in Rust (`molrs`); the MolPy
layer extracts per-species coordinates and box dimensions and returns a typed
result.

!!! note "Conventions used throughout"
    - A *pair* is one reference particle $i$ and one partner particle $j$.
    - A pair is **bonded** at a frame if their minimum-image distance is within a
      cutoff. Two cutoffs may be used: an inner $r_0$ (formation) and an outer
      $r_1\ge r_0$ (breaking).
    - $\langle\cdots\rangle_t$ is an average over time origins; $\tau$ is the lag.
    - Units: length Å, time ps. The correlation $C(\tau)$ is dimensionless.

---

## 1. Why lifetimes need their own tool

Diffusion and conductivity ([transport guide](transport.md)) tell you how *far*
things move, but not how *long* a specific association survives. Yet many
properties — proton transfer, ion-pair stability, solvation-shell exchange — are
governed by the **lifetime** of a contact, not by bulk mobility.

The difficulty is that a contact is rarely clean: two particles sitting near the
cutoff will **rattle** in and out many times per picosecond without the pair
meaningfully breaking. A good lifetime measure has to take a stand on what
counts as "the same bond surviving". That choice is the whole subject of this
page.

---

## 2. The survival correlation function

Define a **survival indicator** for the pair $(i,j)$ that is born at time $t$ and
observed at time $t+\tau$:

$$
S_{ij}(t, t+\tau) =
\begin{cases}
1 & \text{the pair is considered still alive at } t+\tau,\\
0 & \text{otherwise.}
\end{cases}
$$

The **persistence (residence-time) correlation function** averages this over all
reference particles and time origins, normalized per reference particle:

$$
\boxed{\;C(\tau) = \Big\langle\,\frac{1}{N_i}\sum_i\sum_j S_{ij}(t,\,t+\tau)\,\Big\rangle_t\;}
$$

Two limits make this concrete:

- **$C(0)$ is the mean coordination number** — the average number of partners
  within the formation cutoff. (At $\tau=0$ every freshly born pair is alive.)
- **$C(\tau)$ decays** as bonds break; its decay time is the **residence time**.
  Fitting $C(\tau)\approx C(0)\,e^{-\tau/\tau_\text{res}}$ (or integrating it)
  gives a single lifetime.

The only thing left to pin down is the rule inside $S_{ij}$.

---

## 3. Three definitions of survival

The literature offers several survival rules; MolPy implements the three that are
well defined and widely used. All share the same *birth* condition — a pair is
born at $t$ only if it is within the inner cutoff $r_0$ — and differ in how they
decide a born pair is still alive at $t+\tau$.

### 3.1 Continuous survival

The strictest rule (Rapaport, 1983): the pair must stay within the survival
cutoff $r_1$ at **every** frame from $t$ to $t+\tau$. The first time it leaves,
it is dead **forever** for that origin:

$$
S^\text{cont}_{ij}(t,t+\tau) = \prod_{s=0}^{\tau}\Theta\!\big(r_1 - r_{ij}(t+s)\big),
$$

with $\Theta$ the step function. This yields the **continuous** residence time —
short, because every transient excursion ends the bond. Use a single cutoff
($r_1=r_0$) for the classic definition.

### 3.2 Intermittent survival

The most permissive rule (Luzar & Chandler, 1996): the pair is alive at $t+\tau$
if it is within $r_1$ **at that frame**, regardless of whether it left in
between. Re-formation is allowed:

$$
S^\text{int}_{ij}(t,t+\tau) = \Theta\!\big(r_1 - r_{ij}(t+\tau)\big).
$$

This yields the **intermittent** correlation, whose decay reflects the
*structural* lifetime including re-crossings. The continuous and intermittent
correlations bracket the truth, and their ratio characterises how much a bond
rattles.

### 3.3 Stable-states picture (SSP)

A middle ground (Laage & Hynes, 2008) that suppresses rattling without
discarding genuine survival: use **two** cutoffs. A pair is born within the inner
$r_0$ ("stable reactant") and is counted alive as long as it stays within the
outer $r_1$ ("stable product"); it must cross the *outer* cutoff to be declared
broken:

$$
S^\text{SSP}_{ij}(t,t+\tau) = \Theta\!\big(r_0 - r_{ij}(t)\big)\prod_{s=0}^{\tau}\Theta\!\big(r_1 - r_{ij}(t+s)\big),
\qquad r_1\ge r_0.
$$

The gap between $r_0$ and $r_1$ is a buffer: a particle wandering just past the
contact distance is not immediately counted as having left. SSP is the
recommended default for ion pairs.

!!! note "Relation to tame's definitions"
    This mirrors the [tame](https://github.com/Roy-Kid/tame) `persist` recipe.
    tame names two definitions: **IMM** (Impey–Madden–McDonald, 1983), a single
    cutoff with a *tolerance time* that ignores escapes shorter than $t^*$, and
    **SSP** (Laage–Hynes). MolPy's `continuous`/`intermittent`/`ssp` cover the
    same physics with explicit, time-tolerance-free criteria; the IMM
    tolerance-time variant is intentionally not reproduced (its published
    implementation threaded an out-of-scope timestep).

---

## 4. Using `Persist`

`Persist` takes tags of the form `"t1,t2:method:r0[,r1]"`:

- `t1,t2` — reference and partner atom types (e.g. cation and anion).
- `method` — `continuous`, `intermittent`, or `ssp`.
- `r0[,r1]` — inner (and optional outer) cutoff in Å. A single value sets
  $r_1=r_0$.

```python
from molpy.compute import Persist

p = Persist(
    tags=[
        "3,4:ssp:3.0,4.0",        # cation-anion, stable-states picture
        "1,1:continuous:3.5",      # like-species, single-cutoff continuous
    ],
    max_dt=30.0,    # longest lag, ps
    dt=0.01,        # timestep, ps
)
result = p(trajectory)

C = result.correlations["3,4:ssp:3.0,4.0"]
C[0]            # mean coordination number (partners per reference ion)
result.time     # lag times tau, ps
```

When `t1 == t2` the like-species self-pair ($i=j$) is dropped automatically.

---

## 5. From persistence to pairing diffusion

Persistence is not only descriptive — it lets you *decompose* a transport
coefficient by bonding state. Combining the distinct-diffusion correlation of
[§2 of the transport guide](transport.md#2-self-vs-distinct-diffusion-the-mdc)
with a survival function $f(r_{ij};s)$ yields the **pairing contribution** to the
diffusion coefficient:[^gudla]

$$
D^\text{d,pairing}_{\alpha\beta} = \lim_{\tau\to\infty}\frac{1}{6\tau N}
   \sum_i\sum_{j\ne i}\big\langle\Delta\mathbf{r}_{i,\alpha}(\tau)\cdot\Delta\mathbf{r}_{j,\beta}(\tau)\,f(r_{ij};s)\big\rangle,
$$

i.e. only the displacement of partners that remain associated (the "in" channel)
is counted. This separates how *paired* ions diffuse from how *free* ions
diffuse — directly relevant to whether ion pairs carry net charge. Interpret the
result only where **both** the persistence count has converged **and** the MDC is
linear in time.

---

## 6. Pitfalls checklist

1. **Single cutoff with rattling** → the `continuous` lifetime collapses to the
   frame spacing. Use `ssp` with a buffer ($r_1 > r_0$), or `intermittent`.
2. **Cutoff chosen off the RDF** → pick $r_0$ at the first coordination-shell
   minimum of $g(r)$; an arbitrary cutoff makes the lifetime meaningless.
3. **`max_dt` shorter than the lifetime** → $C(\tau)$ never decays within the
   window and no residence time can be read off.
4. **Sparse sampling** → both survival products miss fast re-crossings; sample
   densely enough to resolve the contact dynamics.
5. **Comparing across definitions** → continuous, intermittent, and SSP give
   *different numbers by construction*; always report which one.

---

## 7. References

- D. C. Rapaport, *Mol. Phys.* **50**, 1151 (1983) — continuous vs intermittent
  hydrogen-bond correlation functions.
- A. Luzar, D. Chandler, *Nature* **379**, 55 (1996); *Phys. Rev. Lett.* **76**,
  928 (1996) — intermittent correlation and hydrogen-bond kinetics.
- A. Luzar, *J. Chem. Phys.* **113**, 10663 (2000) — persistence/residence-time
  definitions.
- R. W. Impey, P. A. Madden, I. R. McDonald, *J. Phys. Chem.* **87**, 5071 (1983)
  — residence time with a tolerance time (IMM).
- D. Laage, J. T. Hynes, *J. Phys. Chem. B* **112**, 14230 (2008) — stable-states
  picture (SSP).

[^gudla]: H. Gudla, Y. Shao et al., *J. Phys. Chem. Lett.* **12**, 8460 (2021) —
    pairing contribution to diffusion via a persistence-weighted distinct
    correlation.

## See also

- [Diffusion & Ionic Transport](transport.md) — MSD, distinct diffusion, Onsager
  coefficients, and conductivity.
- [Compute overview](index.md) — the Compute → Result pattern.
- [API reference: Compute](../api/compute.md).
