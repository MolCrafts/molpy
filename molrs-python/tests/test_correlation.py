"""Time correlations through the Python seam: ``Correlator`` and its surfaces
against the definition written out in numpy, and the fits that turn a
``TimeCorrelation`` into a number.

Ported from upstream #137's ``test_correlation.py`` / ``test_oacf.py`` onto
the redesigned API: one ``Correlator`` with named surfaces, the derived
quantities as the fits (``CumulativeTrapezoid``, ``LinearFit``, ``Plateau``,
``DebyeFit``, ``Spectrum``) rather than methods of the curve, and
``ReorientationAcf`` as the orientational autocorrelation (OACF).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from molrs.compute import (
    Correlator,
    CumulativeTrapezoid,
    CurrentAcf,
    DebyeFit,
    Derivative,
    DipoleAcf,
    LinearFit,
    Msd,
    OnsagerCorrelation,
    Plateau,
    ReorientationAcf,
    Spectrum,
    TimeCorrelation,
    Vacf,
)


def _series(n_frames=48, n_items=4, n_components=3, seed=3):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n_frames, n_items, n_components))
    # Smooth it so the correlation decays rather than dropping to noise.
    for t in range(1, n_frames):
        s[t] = 0.8 * s[t - 1] + 0.2 * s[t]
    return s


def _unit_vectors(n_frames=40, n_vectors=6, seed=5):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n_frames, n_vectors, 3))
    for t in range(1, n_frames):
        u[t] = 0.9 * u[t - 1] + 0.1 * u[t]
    return u / np.linalg.norm(u, axis=2, keepdims=True)


def _origins(n_frames, lag, stride):
    return range(0, n_frames - lag, stride)


def _acf_direct(series, max_lag, stride=1, mean_subtract=False):
    """``Σ_τ∈O Σ_items,components a(τ)·a(τ+t) / (N · |O_t|)``."""
    n_frames, n_items, _ = series.shape
    d = series - series.mean(axis=0, keepdims=True) if mean_subtract else series
    out = []
    for lag in range(max_lag + 1):
        origins = _origins(n_frames, lag, stride)
        acc = sum(float((d[tau] * d[tau + lag]).sum()) for tau in origins)
        out.append(acc / (n_items * len(origins)))
    return np.array(out)


def _legendre(order, x):
    return {1: x, 2: 0.5 * (3 * x * x - 1), 3: 0.5 * (5 * x**3 - 3 * x)}[order]


def _oacf_direct(u, order, max_lag, stride=1):
    n_frames = u.shape[0]
    out = []
    for lag in range(max_lag + 1):
        origins = _origins(n_frames, lag, stride)
        acc = sum(
            float(_legendre(order, (u[tau] * u[tau + lag]).sum(axis=1)).sum())
            for tau in origins
        )
        out.append(acc / (u.shape[1] * len(origins)))
    return np.array(out)


class TestTimeCorrelation:
    def test_pickle_round_trip(self):
        tc = Correlator("product", 9, reduction="per_item", stride=2).compute(
            a=_series(), dt=0.25
        )
        back = pickle.loads(pickle.dumps(tc))
        assert isinstance(back, TimeCorrelation)
        np.testing.assert_array_equal(back.values, tc.values)
        np.testing.assert_array_equal(back.lag_times, tc.lag_times)
        assert (back.n_lags, back.n_channels) == (10, 4)

    def test_built_from_a_curve(self):
        tc = TimeCorrelation(np.arange(3) * 0.5, np.array([[2.0], [1.0], [0.5]]))
        np.testing.assert_allclose(tc.lag_times, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(tc.channel(0), [2.0, 1.0, 0.5])
        with pytest.raises(ValueError):
            tc.channel(1)


class TestCorrelator:
    def test_normalizations_and_stride(self):
        s = _series()
        n_frames = s.shape[0]
        unb = Correlator("product", 8).compute(a=s, dt=1.0).channel(0)
        bia = Correlator("product", 8, normalization="biased").compute(a=s, dt=1.0).channel(0)
        raw = Correlator("product", 8, normalization="none").compute(a=s, dt=1.0).channel(0)
        counts = n_frames - np.arange(9)
        np.testing.assert_allclose(bia, unb * counts / n_frames)
        # "none" leaves the sum over origins (still averaged over items).
        np.testing.assert_allclose(raw / counts, unb)
        strided = Correlator("product", 8, stride=3, mean="subtract").compute(a=s, dt=1.0)
        np.testing.assert_allclose(
            strided.channel(0), _acf_direct(s, 8, stride=3, mean_subtract=True), atol=1e-10
        )

    def test_per_item_rows_average_to_the_curve(self):
        s = _series()
        mean = Correlator("product", 6).compute(a=s, dt=1.0)
        rows = Correlator("product", 6, reduction="per_item").compute(a=s, dt=1.0)
        assert rows.values.shape == (7, s.shape[1])
        np.testing.assert_allclose(rows.values.mean(axis=1), mean.channel(0))
        for item in range(s.shape[1]):
            alone = Correlator("product", 6).compute(a=s[:, item : item + 1, :], dt=1.0)
            np.testing.assert_allclose(rows.channel(item), alone.channel(0))

    def test_strided_displacement_and_its_stream_form(self):
        pos = np.cumsum(_series(n_items=5), axis=0)
        got = Correlator("squared_displacement", 7, stride=4).compute(a=pos, dt=1.0)
        n_frames = pos.shape[0]
        for lag in range(8):
            origins = list(_origins(n_frames, lag, 4))
            d = pos[[tau + lag for tau in origins]] - pos[origins]
            assert got.channel(0)[lag] == pytest.approx((d**2).sum(axis=2).mean(), abs=1e-9)

    def test_a_bad_stride_is_refused(self):
        with pytest.raises(ValueError, match="stride"):
            Correlator("product", 3, stride=0)
        with pytest.raises(ValueError, match="stride"):
            Correlator("product", 3, origins="first", stride=2).compute(a=_series(), dt=1.0)


class TestSurfaces:
    def test_vacf_is_the_mean_subtracted_correlator_over_dof(self):
        s = _series()
        flat = s.reshape(s.shape[0], -1)
        vacf = Vacf(10).compute(velocities=flat, dt=0.5)
        acf = Correlator("product", 10, mean="subtract").compute(a=flat[:, :, None], dt=0.5)
        np.testing.assert_array_equal(vacf.values, acf.values)
        strided = Vacf(10, stride=3, normalization="biased").compute(velocities=flat, dt=0.5)
        explicit = Correlator(
            "product", 10, mean="subtract", normalization="biased", stride=3
        ).compute(a=flat[:, :, None], dt=0.5)
        np.testing.assert_array_equal(strided.values, explicit.values)

    def test_an_array_of_the_wrong_rank_or_dtype_says_what_it_expected(self):
        with pytest.raises(
            TypeError,
            match=r"velocities.*expected a 2-D float64 array, "
            r"got a 3-D float64 array of shape \(10, 4, 3\)",
        ):
            Vacf(5).compute(velocities=np.zeros((10, 4, 3)), dt=1.0)
        with pytest.raises(
            TypeError,
            match=r"velocities.*expected a 2-D float64 array, "
            r"got a 2-D int64 array of shape \(100, 32\)",
        ):
            Vacf(5).compute(velocities=np.zeros((100, 32), dtype=np.int64), dt=1.0)
        with pytest.raises(TypeError, match=r"'a': expected a 3-D float64 array, got list"):
            Correlator("product", 2).compute(a=[[[1.0]]], dt=1.0)

    def test_msd_all_origins_is_the_nested_loop(self):
        pos = np.cumsum(_series(n_frames=20, n_items=3), axis=0)
        got = Msd(6).compute(positions=pos, dt=1.0)
        for lag in range(7):
            d = pos[lag:] - pos[: len(pos) - lag]
            assert got.channel(0)[lag] == pytest.approx((d**2).sum(axis=2).mean(), abs=1e-9)

    def test_msd_first_origin_is_the_single_origin_curve(self):
        pos = np.cumsum(_series(n_frames=12, n_items=4), axis=0)
        got = Msd(5, origins="first", reduction="per_item").compute(positions=pos, dt=1.0)
        want = ((pos - pos[0]) ** 2).sum(axis=2)
        np.testing.assert_allclose(got.values, want[:6])
        with pytest.raises(ValueError, match="stride"):
            Msd(5, origins="first", stride=2).compute(positions=pos, dt=1.0)

    def test_onsager_of_straight_lines(self):
        t = np.arange(10.0)[:, None]
        p = np.hstack([t, t, t])
        out = OnsagerCorrelation(4).compute(a=p, b=p, dt=1.0)
        # Straight-line motion: |ΔP|² = 3 t².
        np.testing.assert_allclose(out.channel(0), 3.0 * np.arange(5.0) ** 2, atol=1e-9)

    def test_current_acf_keeps_the_mean_and_dipole_acf_removes_it(self):
        j = _series(n_items=1)[:, 0, :] + 0.7
        gk = CurrentAcf(8).compute(current=j, dt=1.0)
        np.testing.assert_allclose(gk.channel(0), _acf_direct(j[:, None, :], 8), atol=1e-10)
        d = DipoleAcf(8).compute(dipoles=j, dt=1.0)
        np.testing.assert_allclose(
            d.channel(0), _acf_direct(j[:, None, :], 8, mean_subtract=True), atol=1e-10
        )
        strided = CurrentAcf(8, stride=2).compute(current=j, dt=1.0)
        np.testing.assert_allclose(
            strided.channel(0), _acf_direct(j[:, None, :], 8, stride=2), atol=1e-10
        )

    def test_dipole_rate_cross_is_the_gradient_cross_correlation(self):
        m = _series(n_items=1)[:, 0, :]
        dt = 0.5
        mdot = Derivative("gradient").compute(series=m, dt=dt)
        np.testing.assert_allclose(mdot, np.gradient(m, dt, axis=0, edge_order=2))
        c = Correlator("product", 8, mean="subtract").compute(
            a=mdot[:, None, :], b=m[:, None, :], dt=dt
        )
        a = mdot - mdot.mean(axis=0)
        b = m - m.mean(axis=0)
        want = [float((a[: len(m) - lag] * b[lag:]).sum()) / (len(m) - lag) for lag in range(9)]
        np.testing.assert_allclose(c.channel(0), want, atol=1e-10)


class TestReorientationAcf:
    """The orientational autocorrelation (upstream ``Oacf``)."""

    @pytest.mark.parametrize("order", [1, 2, 3])
    @pytest.mark.parametrize("stride", [1, 3])
    def test_matches_the_direct_definition(self, order, stride):
        u = _unit_vectors()
        got = ReorientationAcf(order, 12, stride=stride).compute(vectors=u, dt=0.5)
        np.testing.assert_allclose(got.channel(0), _oacf_direct(u, order, 12, stride), atol=1e-12)
        assert got.channel(0)[0] == pytest.approx(1.0)
        assert got.lag_times[12] == 6.0

    def test_p2_is_the_quadratic_component_identity(self):
        u = _unit_vectors()
        q = np.einsum("tni,tnj->tnij", u, u).reshape(u.shape[0], u.shape[1], 9)
        c_q = Correlator("product", 10).compute(a=q, dt=1.0).channel(0)
        c2 = ReorientationAcf(2, 10).compute(vectors=u, dt=1.0).channel(0)
        np.testing.assert_allclose(1.5 * c_q - 0.5, c2, atol=1e-12)

    def test_c2_is_even_under_reversal(self):
        u = _unit_vectors()
        a = ReorientationAcf(2, 8).compute(vectors=u, dt=1.0).channel(0)
        b = ReorientationAcf(2, 8).compute(vectors=-u, dt=1.0).channel(0)
        np.testing.assert_allclose(a, b, atol=1e-14)

    def test_quarter_turns(self):
        angles = np.arange(16) * np.pi / 2
        u = np.zeros((16, 2, 3))
        u[:, :, 0] = np.cos(angles)[:, None]
        u[:, :, 1] = np.sin(angles)[:, None]
        c1 = ReorientationAcf(1, 4).compute(vectors=u, dt=1.0).channel(0)
        c2 = ReorientationAcf(2, 4).compute(vectors=u, dt=1.0).channel(0)
        np.testing.assert_allclose(c1, [1, 0, -1, 0, 1], atol=1e-12)
        np.testing.assert_allclose(c2, [1, -0.5, 1, -0.5, 1], atol=1e-12)

    def test_per_vector_rows_and_integrated_time(self):
        u = _unit_vectors()
        mean = ReorientationAcf(2, 10).compute(vectors=u, dt=0.5)
        rows = ReorientationAcf(2, 10, reduction="per_item").compute(vectors=u, dt=0.5)
        assert rows.values.shape == (11, 6)
        np.testing.assert_allclose(rows.values.mean(axis=1), mean.channel(0), atol=1e-12)
        running = CumulativeTrapezoid().compute(correlation=mean, channel=0)
        assert running.channel(0)[-1] == pytest.approx(np.trapezoid(mean.channel(0), dx=0.5))

    def test_vectors_are_normalised_not_refused(self):
        """Unlike upstream ``Oacf`` (which refused ``|u| != 1``), the
        Legendre kernel takes vectors of any non-zero length: the cosine is
        the quantity."""
        u = _unit_vectors(n_frames=8, n_vectors=2)
        unit = ReorientationAcf(1, 3).compute(vectors=u, dt=1.0)
        long = ReorientationAcf(1, 3).compute(vectors=2.5 * u, dt=1.0)
        np.testing.assert_allclose(long.values, unit.values, atol=1e-14)

    def test_rejects_bad_input(self):
        u = _unit_vectors(n_frames=8, n_vectors=2)
        zero = u.copy()
        zero[3, 1] = 0.0
        with pytest.raises(ValueError, match="zero length"):
            ReorientationAcf(1, 3).compute(vectors=zero, dt=1.0)
        with pytest.raises(ValueError, match="stride"):
            ReorientationAcf(1, 3, stride=0)
        with pytest.raises(ValueError):
            ReorientationAcf(1, 3).compute(vectors=u[:1], dt=1.0)
        with pytest.raises(ValueError):
            ReorientationAcf(1, 3, reduction="per_component").compute(vectors=u, dt=1.0)


class TestFitsOfACurve:
    """Upstream's ``TimeCorrelation`` methods are the fits applied to it."""

    def test_integral_is_the_cumulative_trapezoid(self):
        tc = Correlator("product", 10).compute(a=_series(), dt=0.5)
        running = CumulativeTrapezoid().compute(correlation=tc, channel=0).channel(0)
        assert running[-1] == pytest.approx(np.trapezoid(tc.channel(0), dx=0.5))
        partial = CumulativeTrapezoid(n_lags=4).compute(correlation=tc, channel=0).channel(0)
        np.testing.assert_allclose(partial, running[:4])

    def test_linear_fit_and_the_einstein_coefficient(self):
        values = 6.0 * 0.25 * np.arange(20) * 0.5
        tc = TimeCorrelation(np.arange(20) * 0.5, values[:, None])
        fit = LinearFit(0.0, 1.0).compute(correlation=tc, channel=0)
        assert fit.slope / (2 * 3) == pytest.approx(0.25)

    def test_green_kubo_is_trapezoid_then_plateau(self):
        dt = 0.1
        vacf = TimeCorrelation(np.arange(200) * dt, np.exp(-np.arange(200) * dt)[:, None])
        running = CumulativeTrapezoid().compute(correlation=vacf, channel=0)
        d = Plateau(0.8, 1.0).compute(correlation=running, channel=0).value / 3.0
        assert d == pytest.approx(1.0 / 3.0, rel=1e-3)

    def test_debye_fit_recovers_the_relaxation_time(self):
        dt, tau = 0.5, 7.0
        values = 3.0 * np.exp(-np.arange(40) * dt / tau)
        tc = TimeCorrelation(np.arange(40) * dt, values[:, None])
        fit = DebyeFit().compute(correlation=tc, channel=0)
        assert fit.tau == pytest.approx(tau)
        assert fit.amplitude == pytest.approx(3.0)

    def test_spectrum_window_is_a_choice(self):
        tc = Vacf(30).compute(velocities=_series().reshape(48, -1), dt=0.5)
        cos2 = Spectrum("cosine_sq", 4).compute(signal=tc.channel(0), dt=0.5)
        hann = Spectrum("hann", 4).compute(signal=tc.channel(0), dt=0.5)
        assert cos2.n_fft == hann.n_fft
        assert not np.allclose(cos2.real, hann.real)
