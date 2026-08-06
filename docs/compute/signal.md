# Signal primitives

Every time-correlation function on these pages — [VACF](vacf.md),
[JACF](jacf.md), [Dielectric](dielectric.md), [Spectra](spectra.md) — is built
from the same three operations: correlate a series with its own past, taper it,
and transform it. `molpy.compute.signal` exposes those operations directly, for
when the packaged computes do not fit the series you have.

They are module-level functions rather than classes, because they are pure array
maths with no natural owner:

```text
signal.acf_fft(series, max_lag)      # autocorrelation
signal.apply_window(series, window)  # taper before transforming
signal.frequency_grid(n, dt_fs)      # the matching frequency axis
```

## Correlating without the O(T²)

The autocorrelation of a series of length $T$ looks like $T$ separate sums, one
per lag, which is $\mathcal{O}(T^2)$ and quickly impossible. The
Wiener–Khinchin theorem gives a shortcut: the autocorrelation is the inverse
Fourier transform of the power spectrum, so an FFT there and back costs
$\mathcal{O}(T\log T)$.

That is what `acf_fft` does. For a 10⁵-frame trajectory the difference is
between seconds and hours, which is why every correlation function in MolPy goes
through it.

Test it on a signal whose correlation you can write down. A cosine of period
$P$ has an autocorrelation that is also a cosine of period $P$:

```python
import numpy as np
from molpy.compute import signal

t = np.arange(2048, dtype=float)
series = np.ascontiguousarray(np.cos(2 * np.pi * t / 64))   # period 64 frames

acf = np.asarray(signal.acf_fft(series, max_lag=200))
print(acf.shape)                                     # -> (201,)
print(round(float(acf[64] / acf[0]), 2))             # -> 0.97
print(round(float(acf[32] / acf[0]), 2))             # -> -0.98
```

At a lag of one full period the correlation is back near +1; at half a period it
is near −1. The small shortfall from exactly ±1 is the finite series length, not
an error.

Note the length: `max_lag=200` returns **201** values, because lag 0 is
included. And note that `acf_fft` does **not** normalize — `acf[0]` here is
1024, the sum of squares, not 1. Divide by `acf[0]` yourself when you want a
normalized correlation function.

## Why you must taper

An FFT assumes the signal repeats forever. If your correlation function has not
decayed to zero by the last lag, the transform sees a discontinuity where the
end wraps around to the beginning, and that artificial step scatters energy
across the whole spectrum as ringing — sinc side-lobes around every real peak.

The fix is to multiply by a window that goes smoothly to zero at both ends:

```python
tapered = np.asarray(signal.apply_window(np.ascontiguousarray(acf), "hann"))
print(round(float(tapered[0]), 3), round(float(tapered[-1]), 3))   # -> 0.0 0.0
```

Two window shapes are accepted, `"hann"` and `"blackman"`; anything else raises
`ValueError: unknown window type`. Blackman suppresses the side-lobes harder and
broadens peaks more.

Both ends are now exactly zero, so there is no discontinuity to transform.

The cost is resolution: tapering throws away information at the ends, which
broadens every peak. That is the trade. Window when your correlation function is
truncated before it decays; do not window when it has genuinely decayed, because
then you are only broadening peaks for nothing.

## The frequency axis

`frequency_grid` builds the axis matching a real FFT of `n` samples spaced
`dt_fs` apart:

```python
grid = np.asarray(signal.frequency_grid(len(acf), 10.0))
print(grid.shape)                                    # -> (101,)
print(round(float(grid.max()), 3))                   # -> 0.313
```

**The grid is in angular frequency, rad fs⁻¹** — not Hz and not cm⁻¹. That 0.313
is $\pi/\Delta t$ for even-length input, and to reach the wavenumbers a
spectroscopist quotes you divide by $2\pi c$:

```python
wavenumbers = grid * 1e15 / (2 * np.pi * 2.998e10)   # rad/fs -> cm^-1
print(round(float(wavenumbers.max())))               # -> 1659
```

1659 cm⁻¹, which is the same Nyquist limit as $16678/(\Delta t/\mathrm{fs}) =
1668$ cm⁻¹ up to the odd/even endpoint. If a computed spectrum comes out a
factor of $2\pi$ or $c$ from where you expect, this conversion is the first
place to look.

Two further properties are worth internalizing. The grid is **half as long** as the
input, because a real signal's spectrum is symmetric and only positive
frequencies are independent. And the maximum is the **Nyquist frequency**, set
entirely by the sampling interval: sample every 10 fs and no information above
that frequency exists, no matter how long you run. In wavenumbers the limit is
$\tilde\nu_{\max} \approx 16678/(\Delta t/\text{fs})$ cm⁻¹, which is 1668 cm⁻¹
at 10 fs — fine for a monatomic liquid, hopeless for O–H stretches.

Resolution is the other half of the story and it is set by the *total length*:
$\Delta\tilde\nu \approx 1/(cT)$. Sampling faster raises the ceiling;
correlating to longer lags sharpens the detail. They are independent knobs and
people routinely reach for the wrong one.

## When to use these instead of a compute

Prefer the packaged computes when they fit: [`Acf`](vacf.md) handles
multi-particle `(T, N, 3)` series and averages correctly over particles and time
origins, which is fiddly to reproduce. Reach for `signal` when you have

- a single scalar series — one degree of freedom, or an already-collective flux;
- an observable with no compute of its own, such as a bespoke order parameter;
- a correlation function assembled elsewhere that you only need to window and
  transform.

The one trap: averaging particles *before* correlating measures the memory of
the collective quantity, not of a typical particle. Those are different physical
quantities, and the [VACF](vacf.md) page shows the difference.

## When it goes wrong

**The spectrum is covered in regular ripples.**
Spectral leakage from an un-tapered, un-decayed correlation function. Apply a
window.

**Peaks are broader than expected after windowing.**
Expected — that is what a window costs. If it matters, correlate to longer lags
rather than removing the window.

**`acf[0]` is not 1.**
By design; `acf_fft` returns unnormalized correlations. Divide by `acf[0]`.

**Frequencies come out an order of magnitude wrong.**
`dt_fs` is in femtoseconds. Passing picoseconds shifts every peak by 1000.

**The array has `max_lag` entries and your loop is off by one.**
It has `max_lag + 1`, because lag 0 is a lag.

**`TypeError` about array layout.**
These functions cross into Rust, which requires C-contiguous float64 input and
hands back a view rather than a NumPy array. That is why every call here is
wrapped in `np.ascontiguousarray` going in and `np.asarray` coming out. Slicing
or transposing an array usually makes it non-contiguous, so wrap it again after
you do.

## Check yourself

- Correlate a pure cosine and confirm the ACF is a cosine of the same period, as
  above.
- Correlate white noise: you should get a spike at lag 0 and nothing else.
- Build a two-frequency signal, transform it, and confirm both peaks land where
  `frequency_grid` says they should.

## References

- N. Wiener, *Acta Math.* **55**, 117 (1930); A. Khinchin, *Math. Ann.* **109**,
  604 (1934) — the theorem behind `acf_fft`.
- F. J. Harris, *Proc. IEEE* **66**, 51 (1978) — windows and spectral leakage,
  still the standard reference.

## See also

- [VACF](vacf.md) — the multi-particle correlation you usually want instead
- [Spectra](spectra.md) — where windowing and the frequency grid are used
- [Dielectric](dielectric.md) · [JACF](jacf.md)
- [API reference](../api/compute.md)
