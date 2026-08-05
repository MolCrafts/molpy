# Spectra

Overview

| Class / entry | Description |
|---------------|-------------|
| [`PowerSpectrum`](#powerspectrum) | VDOS from a velocity ACF. |
| [`IRSpectrum`](#irspectrum) | IR from a dipole-flux ACF. |
| [`RamanSpectrum`](#ramanspectrum) | Raman from isotropic + anisotropic polarizability ACFs. |
| [`VcdSpectrum`](#vcdspectrum) | VCD from the relevant cross-correlation ACF. |
| [`RoaSpectrum`](#roaspectrum) | Raman optical activity. |
| [`ResonanceRamanSpectrum`](#resonanceramanspectrum) | Resonance Raman. |

Details

The `molpy.compute.spectra` module: vibrational spectra from **precomputed** ACFs (`fit(acf, dt_fs)`).

## `PowerSpectrum`

VDOS from a velocity ACF.

```python
import numpy as np
from molpy.compute import PowerSpectrum
from molpy.compute.signal import acf_fft

rng = np.random.default_rng(0)
vel = rng.normal(size=64)  # one scalar series; real VDOS averages DOFs
acf = acf_fft(vel, 16)
vdos = PowerSpectrum()(acf, dt_fs=1.0)
```

## `IRSpectrum`

IR from a dipole-flux ACF.

```python
from molpy.compute import IRSpectrum

ir = IRSpectrum()(acf, dt_fs=1.0)
```

## `RamanSpectrum`

Raman from isotropic + anisotropic polarizability ACFs.

```python
from molpy.compute import RamanSpectrum

acf_iso = acf_fft(rng.normal(size=64), 16)
acf_aniso = acf_fft(rng.normal(size=64), 16)
raman = RamanSpectrum()(acf_iso, acf_aniso, dt_fs=1.0)
```

## `VcdSpectrum`

VCD from the relevant cross-correlation ACF.

```python
from molpy.compute import VcdSpectrum

vcd = VcdSpectrum()(acf, dt_fs=1.0)
```

## `RoaSpectrum`

Raman optical activity.

```python
from molpy.compute import RoaSpectrum

roa = RoaSpectrum()(acf_iso, acf_aniso, dt_fs=1.0)
```

## `ResonanceRamanSpectrum`

Resonance Raman.

```python
from molpy.compute import ResonanceRamanSpectrum

rr = ResonanceRamanSpectrum()(acf_iso, acf_aniso, dt_fs=1.0)
```

## See also

- [Signal](signal.md)
- [Dielectric](dielectric.md)
