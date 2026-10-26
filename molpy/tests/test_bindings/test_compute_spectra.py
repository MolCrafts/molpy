"""FFI seam of the fits, kinetic, dielectric and spectroscopy computes.

Each class is built from its required arguments only and called once on tiny
synthetic data; results are passed on to the next compute the way a workflow
chains them. The numbers themselves are pinned by the Rust unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

import molpy as mp
from molpy.compute import (
    ConductivitySumRule,
    Correlator,
    CumulativeTrapezoid,
    DebyeFit,
    DebyeFitResult,
    DielectricSpectrumResult,
    DipoleAcf,
    DipoleAutocorrelationSpectrum,
    DipoleMoment,
    DipoleMomentResult,
    DipoleRateCrossSpectrum,
    EinsteinHelfandSpectrum,
    GreenKuboSpectrum,
    KineticEnergy,
    KineticEnergyResult,
    KramersKronig,
    KramersKronigCheck,
    LinearFit,
    LinearFitResult,
    Msd,
    Plateau,
    PlateauResult,
    PolarizabilityParts,
    RamanSpectrum,
    RamanSpectrumResult,
    RouteAgreement,
    RouteAgreementCheck,
    StaticDielectric,
    StaticDielectricResult,
    SumRuleCheck,
    TimeCorrelation,
    Vacf,
)

RNG = np.random.default_rng(7)


def _debye_curve(
    n_lags: int = 64, dt: float = 0.1, tau: float = 1.0
) -> TimeCorrelation:
    t = np.arange(n_lags) * dt
    return TimeCorrelation(t, np.exp(-t / tau)[:, None])


def _dipoles(n_frames: int = 128) -> np.ndarray:
    return RNG.normal(size=(n_frames, 3))


class TestFits:
    def test_debye_fit_reads_a_built_curve(self) -> None:
        out = DebyeFit().compute(correlation=_debye_curve(), channel=0)
        assert isinstance(out, DebyeFitResult)
        assert out.tau == pytest.approx(1.0, rel=1e-6)
        assert out.n_samples > 0

    def test_msd_feeds_linear_fit(self) -> None:
        positions = np.cumsum(RNG.normal(size=(60, 4, 3)), axis=0)
        acc = Msd(20).accumulator()
        for frame in positions:
            acc.push(frame, dt=1.0)
        msd = acc.finish()
        fit = LinearFit(0.2, 0.8).compute(correlation=msd, channel=0)
        assert isinstance(fit, LinearFitResult)
        assert fit.slope > 0.0
        assert 0 <= fit.fit_start < fit.fit_end

    def test_vacf_feeds_cumulative_trapezoid_then_plateau(self) -> None:
        velocities = RNG.normal(size=(80, 6))
        acc = Vacf(20).accumulator()
        for frame in velocities:
            acc.push(frame.reshape(-1, 1), dt=0.5)
        vacf = acc.finish()
        integral = CumulativeTrapezoid().compute(correlation=vacf, channel=0)
        assert isinstance(integral, TimeCorrelation)
        assert integral.values.shape == vacf.values.shape
        plateau = Plateau(0.5, 1.0).compute(correlation=integral, channel=0)
        assert isinstance(plateau, PlateauResult)
        assert np.isfinite(plateau.value)

    def test_cumulative_trapezoid_takes_n_lags(self) -> None:
        trapezoid = CumulativeTrapezoid(n_lags=10)
        assert dict(trapezoid.tunables())["n_lags"] == 10
        out = trapezoid.compute(correlation=_debye_curve(), channel=0)
        assert out.n_lags == 10


class TestKinetic:
    def test_kinetic_energy_and_temperature(self) -> None:
        masses = np.array([1.0, 2.0, 3.0])
        velocities = RNG.normal(size=(3, 3))
        out = KineticEnergy("real").compute(masses=masses, velocities=velocities)
        assert isinstance(out, KineticEnergyResult)
        assert out.energy > 0.0
        assert out.temperature(6) > 0.0

    def test_units_may_be_a_preset_object(self) -> None:
        preset = mp.core.UnitPreset("metal")
        out = KineticEnergy(preset).compute(
            masses=np.ones(2), velocities=np.ones((2, 3))
        )
        assert out.boltzmann == pytest.approx(preset.boltzmann())

    def test_constants_override_the_preset(self) -> None:
        masses, velocities = np.array([2.0, 1.0]), np.array([[1.0, 0, 0], [0, 2, 2]])
        out = KineticEnergy("real", mvv2e=1.0, boltzmann=0.5).compute(
            masses=masses, velocities=velocities
        )
        assert (out.energy, out.boltzmann) == (5.0, 0.5)
        assert out.temperature(6) == pytest.approx(2.0 * 5.0 / (6 * 0.5))
        real = KineticEnergy("real").compute(masses=masses, velocities=velocities)
        only_k = KineticEnergy("real", boltzmann=0.5).compute(
            masses=masses, velocities=velocities
        )
        assert (only_k.energy, only_k.boltzmann) == (real.energy, 0.5)
        with pytest.raises(TypeError):
            KineticEnergy(mvv2e=1.0, boltzmann=0.5)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="boltzmann out of range"):
            KineticEnergy("real", boltzmann=0.0)
        with pytest.raises(ValueError, match="mvv2e out of range"):
            KineticEnergy("real", mvv2e=-1.0)


class TestDielectric:
    def test_dipole_moment_whole_system_and_partitioned(self) -> None:
        positions = RNG.normal(size=(5, 4, 3))
        charges = np.array([1.0, -1.0, 0.5, -0.5])
        whole = DipoleMoment().compute(positions=positions, charges=charges)
        assert isinstance(whole, DipoleMomentResult)
        assert whole.series(0).shape == (5, 3)

        labels = np.array([0, 0, 1, 1], dtype=np.uint32)
        split = DipoleMoment().compute(
            positions=positions, charges=charges, partition=labels
        )
        assert list(split.groups) == [0, 1]
        assert split.moments.shape == (5, 2, 3)

    def test_static_dielectric(self) -> None:
        out = StaticDielectric("real").compute(
            dipoles=_dipoles(), volume=1000.0, temperature=300.0, boundary="tin_foil"
        )
        assert isinstance(out, StaticDielectricResult)
        assert out.n_frames == 128
        assert out.epsilon_axes.shape == (3,)


class TestSpectroscopy:
    def _acf(self) -> TimeCorrelation:
        return DipoleAcf(32).compute(dipoles=_dipoles(), dt=1.0)

    def test_dipole_acf_feeds_einstein_helfand_then_kramers_kronig(self) -> None:
        spectrum = EinsteinHelfandSpectrum("real").compute(
            correlation=self._acf(),
            volume=1000.0,
            temperature=300.0,
            boundary="tin_foil",
        )
        assert isinstance(spectrum, DielectricSpectrumResult)
        assert spectrum.eps_real.shape == spectrum.frequencies.shape
        check = KramersKronig().compute(spectrum=spectrum)
        assert isinstance(check, KramersKronigCheck)
        assert isinstance(check.passed, bool)

    def test_routes_feed_route_agreement(self) -> None:
        acf = self._acf()
        spectra = [
            route.compute(
                correlation=acf, volume=1000.0, temperature=300.0, boundary="tin_foil"
            )
            for route in (
                EinsteinHelfandSpectrum("real"),
                DipoleAutocorrelationSpectrum("real", plateau=(0.8, 1.0)),
                DipoleAutocorrelationSpectrum(
                    "real", plateau=(0.8, 1.0), plateau_value="end"
                ),
            )
        ]
        check = RouteAgreement().compute(spectra=spectra)
        assert isinstance(check, RouteAgreementCheck)
        assert len(check.pairwise) == 3

    @pytest.mark.parametrize("route", [GreenKuboSpectrum, DipoleRateCrossSpectrum])
    def test_each_route_computes(self, route) -> None:
        out = route("real").compute(
            correlation=self._acf(),
            volume=1000.0,
            temperature=300.0,
            boundary="tin_foil",
        )
        assert isinstance(out, DielectricSpectrumResult)

    def test_conductivity_sum_rule(self) -> None:
        frequencies = np.linspace(0.0, 1.0, 50)
        out = ConductivitySumRule("real").compute(
            frequencies=frequencies,
            conductivity=np.exp(-frequencies),
            current_sq_mean=1.0,
            volume=1000.0,
            temperature=300.0,
        )
        assert isinstance(out, SumRuleCheck)
        assert np.isfinite(out.relative_error)

    def test_polarizability_parts_feed_raman(self) -> None:
        alpha = RNG.normal(size=(64, 6))
        parts = PolarizabilityParts().compute(alpha=alpha)
        assert parts.isotropic.shape == (64, 1)
        assert parts.anisotropic.shape == (64, 6)

        correlator = Correlator("product", 16)
        iso = correlator.compute(a=parts.isotropic[:, None, :], dt=1.0)
        aniso = correlator.compute(a=parts.anisotropic[:, None, :], dt=1.0)
        out = RamanSpectrum().compute(
            isotropic=iso, anisotropic=aniso, temperature=300.0
        )
        assert isinstance(out, RamanSpectrumResult)
        assert out.isotropic.shape == out.frequencies_cm1.shape


class TestConfiguration:
    def test_kramers_kronig_keeps_the_tolerance_not_given(self) -> None:
        default = dict(KramersKronig().tunables())
        tuned = dict(KramersKronig(relative_tolerance=0.5).tunables())
        assert tuned["relative_tolerance"] == 0.5
        assert tuned["absolute_tolerance"] == default["absolute_tolerance"]

    def test_route_window_default_is_the_core_one(self) -> None:
        tunables = dict(GreenKuboSpectrum("real").tunables())
        assert tunables["window"] == "cosine_sq"
        assert (
            dict(GreenKuboSpectrum("real", window="hann").tunables())["window"]
            == "hann"
        )

    def test_bad_choice_raises_the_core_message(self) -> None:
        with pytest.raises(ValueError, match='unknown EwaldBoundary "metal"'):
            StaticDielectric("real").compute(
                dipoles=_dipoles(), volume=1.0, temperature=1.0, boundary="metal"
            )
        with pytest.raises(ValueError, match='unknown Window "square"'):
            RamanSpectrum(window="square")

    def test_unknown_unit_preset_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown unit preset"):
            KineticEnergy(units="furlongs")
