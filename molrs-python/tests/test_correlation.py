"""``TimeCorrelation`` — the one result of every correlator — and the
producers that build it.

Each derived-quantity method is pinned to the standalone ``Fit`` it
delegates to, and each producer to the definition written out in numpy.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import molrs
from molrs.compute import (
    Acf,
    CumulativeTrapezoid,
    DebyeFit,
    DebyeRelaxation,
    DipoleRateCross,
    EinsteinConductivity,
    EinsteinDiffusion,
    GreenKuboConductivity,
    GreenKuboDiffusion,
    LinearFit,
    Msd,
    Plateau,
    PowerSpectrum,
    Spectrum,
    TimeCorrelation,
    Vacf,
    pair_survival_tcf,
)


def _series(n_frames=48, n_entities=4, n_components=3, seed=3):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n_frames, n_entities, n_components))
    # Smooth it so the ACF decays rather than dropping to noise at lag 1.
    for t in range(1, n_frames):
        s[t] = 0.8 * s[t - 1] + 0.2 * s[t]
    return s


def _acf_direct(series, max_lag, stride=1, mean_subtract=False):
    n_frames, n_entities, _ = series.shape
    mean = series.mean(axis=0, keepdims=True) if mean_subtract else 0.0
    d = series - mean
    out = []
    for lag in range(max_lag + 1):
        origins = range(0, n_frames - lag, stride)
        acc = sum(float((d[tau] * d[tau + lag]).sum()) for tau in origins)
        out.append(acc / (n_entities * len(origins)))
    return np.array(out)


class TestTimeCorrelationObject:
    def test_construction_defaults_and_fields(self):
        tc = TimeCorrelation(np.array([2.0, 1.0, 0.5]))
        assert tc.lags.tolist() == [0, 1, 2]
        assert tc.lag_times is None
        assert tc.counts is None
        assert tc.per_entity is None
        assert tc.kind == "custom"
        assert tc.normalization == "unbiased"
        assert len(tc) == 3
        np.testing.assert_array_equal(np.asarray(tc), [2.0, 1.0, 0.5])
        with pytest.raises(ValueError):
            TimeCorrelation(np.array([]))
        with pytest.raises(ValueError):
            TimeCorrelation(np.ones(3), normalization="weird")

    def test_with_dt_sets_lag_times(self):
        tc = TimeCorrelation(np.array([2.0, 1.0, 0.5])).with_dt(0.5)
        np.testing.assert_allclose(tc.lag_times, [0.0, 0.5, 1.0])
        with pytest.raises(ValueError):
            tc.with_dt(0.0)

    def test_pickle_round_trip_keeps_everything(self):
        s = _series()
        tc = Acf(mean_subtract=True, stride=2).compute(s, max_lag=9, dt=0.25, per_entity=True)
        back = pickle.loads(pickle.dumps(tc))
        np.testing.assert_array_equal(back.values, tc.values)
        np.testing.assert_array_equal(back.per_entity, tc.per_entity)
        np.testing.assert_array_equal(back.counts, tc.counts)
        assert back.to_dict().keys() == tc.to_dict().keys()
        assert (back.kind, back.dt, back.stride, back.mean_subtract, back.n_frames) == (
            "acf",
            0.25,
            2,
            True,
            48,
        )

    def test_normalized_starts_at_one(self):
        tc = Acf().compute(_series(), max_lag=5, per_entity=True)
        n = tc.normalized()
        assert n.values[0] == 1.0
        np.testing.assert_allclose(n.values, tc.values / tc.values[0])
        np.testing.assert_allclose(n.per_entity[0], 1.0)
        msd = Msd(method="window").compute(np.cumsum(_series(), axis=0), max_lag=4)
        with pytest.raises(ValueError, match="normalize"):
            msd.normalized()

    def test_integrate_is_the_cumulative_trapezoid(self):
        tc = Acf().compute(_series(), max_lag=10, dt=0.5)
        running = tc.integrate(running=True)
        want = CumulativeTrapezoid().fit(tc.values, 0.5)["integral"]
        np.testing.assert_array_equal(running, want)
        assert tc.integrate() == want[-1]
        assert tc.integrate(n_lags=4) == CumulativeTrapezoid().fit(tc.values, 0.5, n_lags=4)["integral"][-1]
        assert tc.integrate() == pytest.approx(np.trapezoid(tc.values, dx=0.5))

    def test_linear_fit_is_linear_fit(self):
        tc = TimeCorrelation(np.arange(10.0) * 1.5 + 2.0, dt=0.5, kind="msd")
        got = tc.linear_fit((0.0, 1.0))
        want = LinearFit(0.0, 1.0).fit(tc.lag_times, tc.values)
        assert got == want
        assert got["slope"] == pytest.approx(3.0)

    def test_fit_exponential_is_the_debye_fit_of_the_normalized_curve(self):
        dt, tau = 0.5, 7.0
        values = 3.0 * np.exp(-np.arange(40) * dt / tau)
        tc = TimeCorrelation(values, dt=dt)
        got = tc.fit_exponential()
        want = DebyeFit().fit(values / values[0], dt)
        assert got == want
        assert got["tau"] == pytest.approx(tau)

    def test_diffusion_einstein_and_green_kubo(self):
        msd = TimeCorrelation(6.0 * 0.25 * np.arange(20) * 0.5, dt=0.5, kind="msd")
        assert msd.diffusion(3, (0.0, 1.0)) == pytest.approx(0.25)
        vacf = TimeCorrelation(np.exp(-np.arange(200) * 0.1), dt=0.1, kind="vacf")
        running = CumulativeTrapezoid().fit(vacf.values, 0.1)["integral"]
        want = Plateau(0.8, 1.0).fit(running)["value"] / 3.0
        assert vacf.diffusion(3, (0.8, 1.0)) == want
        with pytest.raises(ValueError, match="kind"):
            TimeCorrelation(np.ones(5), kind="jacf").diffusion(3, (0.0, 1.0))
        with pytest.raises(ValueError):
            msd.diffusion(0, (0.0, 1.0))

    def test_spectrum_is_the_power_spectrum_fit(self):
        tc = Vacf().compute(_series().reshape(48, -1), 0.5, 30)
        spec = tc.spectrum()
        want = PowerSpectrum().fit(tc.values, 0.5)
        assert isinstance(spec, Spectrum)
        np.testing.assert_array_equal(spec.intensities, want.intensities)
        np.testing.assert_array_equal(spec.frequencies_cm1, want.frequencies_cm1)
        assert (spec.window, spec.dt_fs, spec.resolution, spec.n_frames) == ("cosine_sq", 0.5, 30, 31)
        hann = tc.spectrum(window="hann")
        assert not np.allclose(hann.intensities, spec.intensities)
        with pytest.raises(ValueError, match="dt"):
            Acf().compute(_series(), max_lag=5).spectrum()
        back = pickle.loads(pickle.dumps(spec))
        np.testing.assert_array_equal(back.intensities, spec.intensities)
        assert back.to_dict()["window"] == "cosine_sq"
        assert len(back) == len(spec)


class TestProducers:
    def test_acf_normalizations_and_stride(self):
        s = _series()
        unb = Acf().compute(s, max_lag=8)
        bia = Acf(normalization="biased").compute(s, max_lag=8)
        raw = Acf(normalization="raw").compute(s, max_lag=8)
        n_frames = s.shape[0]
        np.testing.assert_allclose(bia.values, unb.values * (n_frames - np.arange(9)) / n_frames)
        np.testing.assert_allclose(raw.values / (s.shape[1] * unb.counts), unb.values)
        strided = Acf(stride=3, mean_subtract=True).compute(s, max_lag=8)
        np.testing.assert_allclose(strided.values, _acf_direct(s, 8, stride=3, mean_subtract=True), atol=1e-10)
        assert strided.counts.tolist() == [(n_frames - 1 - lag) // 3 + 1 for lag in range(9)]

    def test_per_entity_rows_average_to_the_curve(self):
        s = _series()
        tc = Acf().compute(s, max_lag=6, per_entity=True)
        assert tc.per_entity.shape == (7, s.shape[1])
        np.testing.assert_allclose(tc.per_entity.mean(axis=1), tc.values)
        for e in range(s.shape[1]):
            alone = Acf().compute(s[:, e : e + 1, :], max_lag=6)
            np.testing.assert_allclose(tc.per_entity[:, e], alone.values)

    def test_vacf_is_the_mean_subtracted_acf_over_dof(self):
        s = _series()
        flat = s.reshape(s.shape[0], -1)
        vacf = Vacf().compute(flat, 0.5, 10)
        acf = Acf(mean_subtract=True).compute(flat[:, :, None], max_lag=10, dt=0.5)
        np.testing.assert_array_equal(vacf.values, acf.values)
        assert vacf.kind == "vacf"
        three_d = Vacf().compute(s, 0.5, 10)
        np.testing.assert_array_equal(three_d.values, vacf.values)
        gk = GreenKuboDiffusion().compute(flat, 0.5, 10)
        np.testing.assert_array_equal(gk.values, vacf.values)

    def test_einstein_diffusion_is_the_window_msd(self):
        pos = np.cumsum(_series(n_entities=5), axis=0)
        ed = EinsteinDiffusion().compute(pos, 2.0)
        msd = Msd(method="window").compute(pos, dt=2.0, per_entity=False)
        np.testing.assert_array_equal(ed.values, msd.values)
        assert ed.per_entity is None and ed.n_entities == 5

    def test_einstein_conductivity_is_the_collective_msd(self):
        m = np.cumsum(_series(n_entities=1)[:, 0, :], axis=0)
        ec = EinsteinConductivity().compute(m, 1.0, 10)
        for lag in range(11):
            d = m[lag:] - m[: len(m) - lag]
            assert ec.values[lag] == pytest.approx((d**2).sum(axis=1).mean(), abs=1e-9)
        with pytest.raises(ValueError, match="shape"):
            EinsteinConductivity().compute(m[:, :2], 1.0, 10)

    def test_green_kubo_conductivity_does_not_mean_subtract(self):
        j = _series(n_entities=1)[:, 0, :] + 0.7
        gk = GreenKuboConductivity().compute(j, 1.0, 8)
        np.testing.assert_allclose(gk.values, _acf_direct(j[:, None, :], 8), atol=1e-10)
        assert gk.kind == "jacf"
        centred = GreenKuboConductivity(mean_subtract=True).compute(j, 1.0, 8)
        assert not np.allclose(centred.values, gk.values)

    def test_debye_relaxation_carries_its_metadata(self):
        m = _series(n_entities=1)[:, 0, :]
        d = DebyeRelaxation(1000.0, 300.0).compute(m, 1.0, 8)
        assert d.kind == "dipole_acf"
        np.testing.assert_allclose(d.values, _acf_direct(m[:, None, :], 8, mean_subtract=True), atol=1e-10)
        assert d.extras == {
            "volume": 1000.0,
            "temperature": 300.0,
            "boundary": "tinfoil",
            "zero_lag_variance": d.values[0],
        }
        assert pickle.loads(pickle.dumps(d)).extras == d.extras
        with pytest.raises(ValueError):
            DebyeRelaxation(0.0, 300.0)

    def test_dipole_rate_cross_is_the_gradient_cross_correlation(self):
        m = _series(n_entities=1)[:, 0, :]
        dt = 0.5
        c = DipoleRateCross().compute(m, dt, 8)
        mdot = np.gradient(m, dt, axis=0, edge_order=2)
        a = mdot - mdot.mean(axis=0)
        b = m - m.mean(axis=0)
        want = [float((a[: len(m) - lag] * b[lag:]).sum()) / (len(m) - lag) for lag in range(9)]
        np.testing.assert_allclose(c.values, want, atol=1e-10)
        assert c.kind == "cross"

    def test_pair_survival_returns_a_time_correlation(self):
        rng = np.random.default_rng(1)
        coords = rng.uniform(0.0, 10.0, size=(12, 6, 3))
        box = np.full((12, 3), 10.0)
        tc = pair_survival_tcf(coords, coords, box, 1.0, 4.0, "intermittent", 1.0, 5, exclude_self=True)
        assert tc.kind == "survival"
        assert len(tc) == 6
        assert tc.counts.tolist() == [12 - lag for lag in range(6)]
        assert tc.lag_times[5] == 5.0


class TestRemovedNames:
    def test_result_classes_are_gone(self):
        for name in ("AcfResult", "MsdResult", "MsdTimeSeries"):
            assert not hasattr(molrs.compute, name)
