"""molpy.compute vibrational-spectra benchmarks.

The spectral transforms take a precomputed autocorrelation curve and the
sampling interval (fs) and return a ``{frequency, intensity}`` spectrum. Power /
IR / VCD take one ACF; Raman / ROA / resonance-Raman take iso + aniso ACFs.
"""

from __future__ import annotations

import pytest

from molpy.compute import (
    IRSpectrum,
    PowerSpectrum,
    RamanSpectrum,
    ResonanceRamanSpectrum,
    RoaSpectrum,
    VcdSpectrum,
)

pytestmark = pytest.mark.benchmark

DT_FS = 0.5


def test_power_spectrum(benchmark, raw_acf) -> None:
    out = benchmark(PowerSpectrum(), raw_acf, DT_FS)
    assert "frequencies_cm1" in out


def test_ir_spectrum(benchmark, raw_acf) -> None:
    out = benchmark(IRSpectrum(), raw_acf, DT_FS)
    assert "frequencies_cm1" in out


def test_vcd_spectrum(benchmark, raw_acf) -> None:
    out = benchmark(VcdSpectrum(), raw_acf, DT_FS)
    assert "frequencies_cm1" in out


def test_raman_spectrum(benchmark, raw_acf) -> None:
    op = RamanSpectrum(incident_frequency_cm1=20000.0, temperature_k=300.0)
    out = benchmark(op, raw_acf, raw_acf, DT_FS)
    assert "frequencies_cm1" in out


def test_roa_spectrum(benchmark, raw_acf) -> None:
    out = benchmark(RoaSpectrum(averaged=True), raw_acf, raw_acf, DT_FS)
    assert "frequencies_cm1" in out


def test_resonance_raman_spectrum(benchmark, raw_acf) -> None:
    op = ResonanceRamanSpectrum(incident_frequency_cm1=20000.0)
    out = benchmark(op, raw_acf, raw_acf, DT_FS)
    assert "frequencies_cm1" in out
