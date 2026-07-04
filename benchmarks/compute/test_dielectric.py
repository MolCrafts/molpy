"""molpy.compute dielectric-spectroscopy benchmarks.

ACFAnalyzer (collective ACF) and SpectralAnalyzer (its FFT) are the building
blocks; DielectricSusceptibility runs both Einstein-Helfand and Green-Kubo
routes; DebyeFit fits a single-relaxation model to the resulting spectrum;
IonicConductivity is the charge-current MSD route.
"""

from __future__ import annotations

import pytest

from molpy.compute import (
    ACFAnalyzer,
    DielectricSusceptibility,
    IonicConductivity,
    SpectralAnalyzer,
)

pytestmark = pytest.mark.benchmark


def test_acf_analyzer(benchmark, charge_traj) -> None:
    op = ACFAnalyzer(columns=["x", "y", "z"], max_lag=5, unwrap=False)
    result = benchmark(op, charge_traj)
    assert result.acf.shape == (6,)


def test_spectral_analyzer(benchmark, charge_traj) -> None:
    acf_result = ACFAnalyzer(columns=["x", "y", "z"], max_lag=5, unwrap=False)(
        charge_traj
    )
    op = SpectralAnalyzer(dt=0.001, window_type="hann")
    result = benchmark(op, acf_result)
    assert len(result.frequency) == len(result.spectrum)


def test_dielectric_susceptibility(benchmark, charge_traj) -> None:
    op = DielectricSusceptibility(
        dt=0.001,
        temperature=300.0,
        max_correlation_time=5,
        routes=["einstein-helfand", "green-kubo"],
    )
    result = benchmark(op, charge_traj)
    assert "EH-full" in result.results and "GK-full" in result.results


def test_debye_fit(benchmark, charge_traj) -> None:
    eh = DielectricSusceptibility(dt=0.001, temperature=300.0, max_correlation_time=5)(
        charge_traj
    ).results["EH-full"]
    fit = benchmark(eh.fit_debye)
    assert fit.tau is not None


def test_ionic_conductivity(benchmark, ion_traj) -> None:
    op = IonicConductivity(dt=2.0, temperature=298.15, max_correlation_time=8)
    result = benchmark(op, ion_traj)
    assert result.time.shape == result.msd.shape
