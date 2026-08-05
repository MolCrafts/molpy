# Signal

Overview

| Class / entry | Description |
|---------------|-------------|
| [`acf_fft`](#acf_fft) | Autocorrelation via FFT. |
| [`apply_window`](#apply_window) | Apply a spectral window. |
| [`frequency_grid`](#frequency_grid) | Frequency axis for spectra. |

Details

The `molpy.compute.signal` module: shared signal-processing helpers.

## `acf_fft`

Autocorrelation via FFT.

```python
import numpy as np
from molpy.compute.signal import acf_fft, apply_window, frequency_grid

x = np.random.default_rng(0).standard_normal(64)
acf = acf_fft(x, 10)
```

## `apply_window`

Apply a spectral window.

```python
windowed = apply_window(x, "hann")
```

## `frequency_grid`

Frequency axis for spectra.

```python
freqs = frequency_grid(n_fft=64, dt=1.0)
```

## See also

- [Spectra](spectra.md)
- [Dielectric](dielectric.md)
