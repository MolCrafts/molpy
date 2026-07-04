"""Vibrational spectra from autocorrelation functions — molrs-backed.

These operators are spectral *transforms*: each takes a raw, precomputed
autocorrelation function (ACF) sampled at ``dt_fs`` femtoseconds and returns the
corresponding spectrum (Fourier transform with the appropriate quantum/temperature
prefactor). They are the time-correlation route to vibrational spectroscopy from
molecular-dynamics trajectories:

- :class:`PowerSpectrum` — vibrational density of states from a velocity ACF.
- :class:`IRSpectrum` — infrared absorption from a dipole-derivative (flux) ACF.
- :class:`RamanSpectrum` — Raman from isotropic + anisotropic polarizability ACFs.
- :class:`VcdSpectrum` — vibrational circular dichroism.
- :class:`RoaSpectrum` — Raman optical activity.
- :class:`ResonanceRamanSpectrum` — resonance Raman.

Thin shells over the molrs analysis-parity kernels; called as ``compute(acf, dt_fs)``
(or ``compute(acf_iso, acf_aniso, dt_fs)`` for the polarizability-based spectra).

References
----------
- D. A. McQuarrie, *Statistical Mechanics*, Harper & Row (1976) — time-correlation
  functions and spectral densities.
- M. Thomas, M. Brehm, R. Fligg, P. Vöhringer, B. Kirchner, *Phys. Chem. Chem.
  Phys.* **15**, 6608 (2013) — IR and Raman spectra from AIMD via TCFs.
- M. Brehm, M. Thomas, *J. Phys. Chem. Lett.* **8**, 3409 (2017) — VCD, ROA and
  resonance Raman from MD (reference implementation).
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — reference implementation.
"""

from __future__ import annotations

import molrs

from .base import Compute


class PowerSpectrum(Compute):
    """Vibrational density of states (VDOS) from a velocity ACF.

    Called as ``compute(acf, dt_fs)``.
    """

    def __init__(self):
        super().__init__()
        self._inner = molrs.PowerSpectrum()

    def __call__(self, acf, dt_fs):
        return self._inner.fit(acf, dt_fs)


class IRSpectrum(Compute):
    """Infrared absorption spectrum from a dipole-flux ACF.

    Called as ``compute(acf, dt_fs)``.
    """

    def __init__(self):
        super().__init__()
        self._inner = molrs.IRSpectrum()

    def __call__(self, acf, dt_fs):
        return self._inner.fit(acf, dt_fs)


class RamanSpectrum(Compute):
    """Raman spectrum from isotropic + anisotropic polarizability ACFs.

    Parameters
    ----------
    incident_frequency_cm1 : float, default 0.0
        Laser frequency for the Raman prefactor (0 disables it).
    temperature_k : float, default 0.0
        Temperature for the Bose-Einstein prefactor (0 disables it).
    averaged : bool, default False
        Apply orientational averaging.

    Called as ``compute(acf_iso, acf_aniso, dt_fs)``.
    """

    def __init__(
        self,
        incident_frequency_cm1: float = 0.0,
        temperature_k: float = 0.0,
        averaged: bool = False,
    ):
        super().__init__(
            incident_frequency_cm1=incident_frequency_cm1,
            temperature_k=temperature_k,
            averaged=averaged,
        )
        self._inner = molrs.RamanSpectrum(
            incident_frequency_cm1, temperature_k, averaged
        )

    def __call__(self, acf_iso, acf_aniso, dt_fs):
        return self._inner.fit(acf_iso, acf_aniso, dt_fs)


class VcdSpectrum(Compute):
    """Vibrational circular dichroism from an electric/magnetic cross-ACF.

    Called as ``compute(acf, dt_fs)``.
    """

    def __init__(self):
        super().__init__()
        self._inner = molrs.VcdSpectrum()

    def __call__(self, acf, dt_fs):
        return self._inner.fit(acf, dt_fs)


class RoaSpectrum(Compute):
    """Raman optical activity from isotropic + anisotropic ROA ACFs.

    Parameters mirror :class:`RamanSpectrum`. Called as
    ``compute(acf_iso, acf_aniso, dt_fs)``.
    """

    def __init__(
        self,
        incident_frequency_cm1: float = 0.0,
        temperature_k: float = 0.0,
        averaged: bool = False,
    ):
        super().__init__(
            incident_frequency_cm1=incident_frequency_cm1,
            temperature_k=temperature_k,
            averaged=averaged,
        )
        self._inner = molrs.RoaSpectrum(incident_frequency_cm1, temperature_k, averaged)

    def __call__(self, acf_iso, acf_aniso, dt_fs):
        return self._inner.fit(acf_iso, acf_aniso, dt_fs)


class ResonanceRamanSpectrum(Compute):
    """Resonance-Raman spectrum from resonant isotropic + anisotropic ACFs.

    Parameters mirror :class:`RamanSpectrum`. Called as
    ``compute(acf_iso, acf_aniso, dt_fs)``.
    """

    def __init__(
        self,
        incident_frequency_cm1: float = 0.0,
        temperature_k: float = 0.0,
        averaged: bool = False,
    ):
        super().__init__(
            incident_frequency_cm1=incident_frequency_cm1,
            temperature_k=temperature_k,
            averaged=averaged,
        )
        self._inner = molrs.ResonanceRamanSpectrum(
            incident_frequency_cm1, temperature_k, averaged
        )

    def __call__(self, acf_iso, acf_aniso, dt_fs):
        return self._inner.fit(acf_iso, acf_aniso, dt_fs)
