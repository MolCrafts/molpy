"""Dielectric response and conductivity building blocks.

Compose as follows (analysis time in fs):

* ε(0): ``Dielectric.static_dielectric_constant(...)``
* ε(ω) EH: ``DebyeRelaxation`` → ``EinsteinHelfandSpectrum``
* ε(ω) GK: ``GreenKuboConductivity`` → ``GreenKuboSpectrum``
* σ EH: ``M = Σ q r`` (unwrapped) → ``EinsteinConductivity`` → ``LinearFit``
* σ GK: ``J = Σ q v`` → ``GreenKuboConductivity`` → ``CumulativeTrapezoid``
"""

from __future__ import annotations

from molrs.compute.dielectric import Dielectric
from molrs.compute.fitting import CumulativeTrapezoid, LinearFit
from molrs.compute.spectroscopy import EinsteinHelfandSpectrum, GreenKuboSpectrum
from molrs.compute.transport import (
    DebyeFit,
    DebyeRelaxation,
    EinsteinConductivity,
    GreenKuboConductivity,
)
from molrs.signal import acf_fft, apply_window, frequency_grid

__all__ = [
    "Dielectric",
    "DebyeRelaxation",
    "DebyeFit",
    "EinsteinConductivity",
    "GreenKuboConductivity",
    "EinsteinHelfandSpectrum",
    "GreenKuboSpectrum",
    "LinearFit",
    "CumulativeTrapezoid",
    "acf_fft",
    "apply_window",
    "frequency_grid",
]
