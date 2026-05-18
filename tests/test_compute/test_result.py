"""Tests for dielectric result dataclasses in molpy.compute.result."""

import dataclasses

import numpy as np

from molpy.compute.result import (
    ACFResult,
    DielectricResult,
    DielectricSusceptibilityResult,
    Result,
    SpectralResult,
    TimeSeriesResult,
)


class TestACFResult:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(ACFResult)

    def test_extends_timeseries_result(self):
        assert issubclass(ACFResult, TimeSeriesResult)
        assert issubclass(ACFResult, Result)

    def test_default_fields(self):
        r = ACFResult()
        assert r.acf.size == 0
        assert r.n_lags == 0

    def test_construct_with_values(self):
        acf = np.array([1.0, 0.5, 0.0])
        r = ACFResult(acf=acf, n_lags=3, time=np.array([0.0, 1.0, 2.0]))
        assert np.array_equal(r.acf, acf)
        assert r.n_lags == 3

    def test_to_dict(self):
        r = ACFResult(acf=np.array([1.0, 0.5]), n_lags=2, time=np.array([0.0, 1.0]))
        d = r.to_dict()
        assert "acf" in d
        assert d["n_lags"] == 2

    def test_acf_dtype(self):
        r = ACFResult(acf=np.array([1.0, 2.0], dtype=np.float64), n_lags=2)
        assert r.acf.dtype == np.float64


class TestSpectralResult:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(SpectralResult)

    def test_extends_result(self):
        # Frequency-domain results no longer inherit from TimeSeriesResult
        # (the inherited `time` field was always populated with frequency
        # values, which made the API misleading).
        assert issubclass(SpectralResult, Result)
        assert not issubclass(SpectralResult, TimeSeriesResult)

    def test_construct_with_values(self):
        freq = np.array([0.0, 1.0, 2.0])
        spec = np.array([1.0, 0.5, 0.1])
        r = SpectralResult(frequency=freq, spectrum=spec)
        assert np.array_equal(r.frequency, freq)
        assert np.array_equal(r.spectrum, spec)


class TestDielectricResult:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(DielectricResult)

    def test_extends_result(self):
        # See TestSpectralResult.test_extends_result.
        assert issubclass(DielectricResult, Result)
        assert not issubclass(DielectricResult, TimeSeriesResult)

    def test_construct_minimal(self):
        r = DielectricResult(
            frequency=np.array([0.0, 1.0]),
            epsilon_real=np.array([80.0, 70.0]),
            epsilon_imag=np.array([0.0, 5.0]),
            epsilon_static=80.0,
            epsilon_inf=1.8,
            route="green-kubo",
            component="full",
        )
        assert r.epsilon_static == 80.0
        assert r.route == "green-kubo"

    def test_optional_conductivity_defaults_to_none(self):
        r = DielectricResult(
            frequency=np.array([0.0]),
            epsilon_real=np.array([80.0]),
            epsilon_imag=np.array([0.0]),
            epsilon_static=80.0,
            epsilon_inf=1.0,
            route="einstein-helfand",
            component="water",
        )
        assert r.conductivity is None

    def test_conductivity_settable(self):
        sigma = np.array([0.1, 0.2])
        r = DielectricResult(
            frequency=np.array([0.0, 1.0]),
            epsilon_real=np.array([80.0, 70.0]),
            epsilon_imag=np.array([0.0, 5.0]),
            epsilon_static=80.0,
            epsilon_inf=1.0,
            route="green-kubo",
            component="full",
            conductivity=sigma,
        )
        assert np.array_equal(r.conductivity, sigma)


class TestDielectricSusceptibilityResult:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(DielectricSusceptibilityResult)

    def test_extends_result(self):
        assert issubclass(DielectricSusceptibilityResult, Result)

    def test_construct_empty(self):
        r = DielectricSusceptibilityResult()
        assert r.results == {}
        assert r.metadata == {}

    def test_construct_with_results(self):
        dr = DielectricResult(
            frequency=np.array([0.0, 1.0]),
            epsilon_real=np.array([80.0, 70.0]),
            epsilon_imag=np.array([0.0, 5.0]),
            epsilon_static=80.0,
            epsilon_inf=1.0,
            route="green-kubo",
            component="full",
        )
        r = DielectricSusceptibilityResult(
            results={"GK-full": dr},
            metadata={"dt": 0.001, "temperature": 300.0},
        )
        assert "GK-full" in r.results
        assert r.results["GK-full"].epsilon_static == 80.0

    def test_to_dict_recursive(self):
        dr = DielectricResult(
            frequency=np.array([0.0]),
            epsilon_real=np.array([80.0]),
            epsilon_imag=np.array([0.0]),
            epsilon_static=80.0,
            epsilon_inf=1.0,
            route="einstein-helfand",
            component="full",
        )
        r = DielectricSusceptibilityResult(results={"EH-full": dr})
        d = r.to_dict()
        assert d["results"]["EH-full"]["epsilon_static"] == 80.0
