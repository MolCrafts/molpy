"""Transport Computes compose with Fits (raw curve → integral / slope)."""

from __future__ import annotations

import numpy as np

from molpy.compute import EinsteinConductivity, GreenKuboConductivity
from molrs.compute.fitting import CumulativeTrapezoid, LinearFit


def test_green_kubo_compose_raw_plus_integral():
    j = np.ones((32, 3))
    j[:, 1:] = 0.0
    raw = GreenKuboConductivity().compute(j, 1.0, 10)
    integ = CumulativeTrapezoid().fit(raw["jacf"], 1.0)
    assert "integral" in integ
    assert integ["integral"][-1] > 0.0


def test_einstein_compose_raw_plus_linear_fit():
    n = 40
    m = np.zeros((n, 3))
    m[:, 0] = np.linspace(0.0, 1.0, n)
    raw = EinsteinConductivity().compute(m, 1.0, 15)
    fit = LinearFit(0.1, 0.9).fit(raw["lag_times"], raw["msd"])
    assert "slope" in fit
    assert fit["slope"] > 0.0
