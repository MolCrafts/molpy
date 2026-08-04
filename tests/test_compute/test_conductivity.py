"""EinsteinConductivity.compute + DielectricResult.fit_debye."""

from __future__ import annotations

import numpy as np
import pytest

from molpy.compute import EinsteinConductivity
from molpy.compute.result import DielectricResult


def test_einstein_conductivity_raw_msd():
    n = 50
    m = np.zeros((n, 3))
    m[:, 0] = 0.01 * np.arange(n)
    raw = EinsteinConductivity().compute(m, 2.0, 20)
    assert raw["msd"][0] == pytest.approx(0.0, abs=1e-12)
    assert raw["msd"][-1] > raw["msd"][1]


def test_dielectric_result_fit_debye():
    tau0, delta, eps_inf = 5.0, 40.0, 1.0
    n, dt = 513, 0.5
    n_pad = 2 * (n - 1)
    freq = 2.0 * np.pi * np.fft.rfftfreq(n_pad, d=dt)
    x = freq * tau0
    denom = 1.0 + x * x
    er = eps_inf + delta / denom
    ei = delta * x / denom
    er[0] = eps_inf + delta
    ei[0] = 0.0
    res = DielectricResult(
        frequency=freq,
        epsilon_real=er,
        epsilon_imag=ei,
        epsilon_static=eps_inf + delta,
        epsilon_inf=eps_inf,
        route="einstein-helfand",
        component="full",
    )
    fit = res.fit_debye()
    assert fit.tau == pytest.approx(tau0, rel=0.2)
