"""Dielectric / conductivity benchmarks against molrs Computes (array API).

Recipe wrappers (ACFAnalyzer, DielectricSusceptibility, IonicConductivity) are
gone — compose raw curves with Fits, same surface as molrs.
"""

from __future__ import annotations

import numpy as np
import pytest

from molpy.compute import (
    Acf,
    DebyeFit,
    DebyeRelaxation,
    Dielectric,
    EinsteinConductivity,
    EinsteinHelfandSpectrum,
    GreenKuboConductivity,
    GreenKuboSpectrum,
)

pytestmark = pytest.mark.benchmark


def test_acf(benchmark) -> None:
    # Acf expects (n_frames, n_entities, n_components)
    series = np.zeros((32, 1, 3), dtype=np.float64)
    series[:, 0, 0] = np.linspace(1.0, 0.2, 32)

    def run():
        return Acf().compute(series, max_lag=8)

    out = benchmark(run)
    assert len(out.acf) == 9


def test_debye_relaxation(benchmark) -> None:
    rng = np.random.default_rng(0)
    M = np.cumsum(rng.normal(size=(40, 3)) * 0.1, axis=0)

    def run():
        return DebyeRelaxation(volume=1000.0, temperature=300.0).compute(M, 1.0, 10)

    raw = benchmark(run)
    assert raw["acf"].shape[0] == 11


def test_einstein_helfand_spectrum(benchmark) -> None:
    rng = np.random.default_rng(1)
    M = np.cumsum(rng.normal(size=(40, 3)) * 0.1, axis=0)
    raw = DebyeRelaxation(volume=1000.0, temperature=300.0).compute(M, 1.0, 10)
    fit = EinsteinHelfandSpectrum(
        dt=1.0,
        volume=raw["volume"],
        temperature=raw["temperature"],
        epsilon_inf=1.0,
        zero_lag_variance=raw["zero_lag_variance"],
    )

    def run():
        return fit.fit(raw["acf"])

    spec = benchmark(run)
    assert "frequencies" in spec and "eps_real" in spec


def test_green_kubo_spectrum(benchmark) -> None:
    j = np.ones((32, 3), dtype=np.float64)
    j[:, 1:] = 0.0
    raw = GreenKuboConductivity().compute(j, 1.0, 10)
    fit = GreenKuboSpectrum(
        dt=1.0, volume=1000.0, temperature=300.0, epsilon_inf=1.0
    )

    def run():
        return fit.fit(raw["jacf"])

    spec = benchmark(run)
    assert "frequencies" in spec and "eps_real" in spec


def test_debye_fit(benchmark) -> None:
    phi = np.exp(-np.arange(20, dtype=np.float64) / 5.0)

    def run():
        return DebyeFit().fit(phi, 1.0)

    fit = benchmark(run)
    assert fit is not None


def test_static_dielectric(benchmark) -> None:
    rng = np.random.default_rng(2)
    M = rng.normal(size=(30, 3))

    def run():
        return Dielectric.static_dielectric_constant(M, 1000.0, 300.0, 1.0)

    eps = benchmark(run)
    assert np.isfinite(eps)


def test_einstein_conductivity_from_dipole(benchmark) -> None:
    """Ionic-conductivity EH path: M = Σ q r → EinsteinConductivity."""
    m = np.zeros((40, 3), dtype=np.float64)
    m[:, 0] = np.linspace(0.0, 2.0, 40)

    def run():
        return EinsteinConductivity().compute(m, 1.0, 10)

    raw = benchmark(run)
    assert raw["msd"].shape[0] == 11
