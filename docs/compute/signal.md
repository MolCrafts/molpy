# Signal

Textbook guide to **signal helpers** shared by transport and spectroscopy:
unbiased FFT autocorrelations, windowing, and frequency grids.

These are array primitives — not frame analyses. Spectra and Green–Kubo routes
compose them with Fits (`CumulativeTrapezoid`, `PowerSpectrum`, …).

---

## 1. Autocorrelation via FFT (Wiener–Khinchin)

For a real series $x_t$,

$$
C(\tau) = \frac{1}{N-\tau}\sum_{t=0}^{N-\tau-1} x_t\, x_{t+\tau}
$$

(unbiased origin average). `signal.acf_fft` evaluates this in $\mathcal{O}(N\log N)$.
For multi-entity vector series $(T,N,3)$ prefer `Acf().compute(...)` which
contracts components and averages entities consistently.

## 2. Windowing and spectral resolution

Truncating $C(\tau)$ before it decays produces sinc ringing in frequency space.
Apply Hann/Blackman (`signal.apply_window`) before the FFT. Resolution
$\Delta\tilde\nu \sim 1/(c T_\mathrm{ACF})$; Nyquist
$\tilde\nu_\max \approx 16678/(\Delta t/\mathrm{fs})$ cm⁻¹.

---

## 3. Usage

```python
import numpy as np
from molpy.compute import signal, Acf, PowerSpectrum

rng = np.random.default_rng(0)
x = np.ascontiguousarray(rng.standard_normal(256))
C = signal.acf_fft(x, max_lag=64)
C_w = signal.apply_window(C, "hann")
vdos = PowerSpectrum()(C_w, dt_fs=1.0)

vel = np.ascontiguousarray(rng.standard_normal((128, 20, 3)))
vacf = Acf().compute(vel, max_lag=32)
```

---

## 4. Pitfalls

1. Biased ($1/N$) vs unbiased ($1/(N-\tau)$) normalization mismatch.
2. No window on a truncated ACF → spectral ringing.
3. Averaging atoms *before* correlating when you wanted per-particle memory.

## See also

- [VACF](vacf.md) · [Spectra](spectra.md) · [JACF](jacf.md) · [Dielectric](dielectric.md)
- [API reference](../api/compute.md)
