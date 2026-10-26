"""The ``mp.compute`` FFI seam: the bindings are derived from the core.

Numerical correctness lives in the Rust unit tests of ``molrs::compute``; these
tests pin what the binding itself owns:

* **name parity** — the exported compute classes are exactly
  ``{spec.name for spec in ComputeSpec.all()}``;
* **signature parity** — ``__init__`` takes exactly the spec's parameters and
  ``compute`` exactly the spec's input fields, every optional one ``None``; a
  stream only (``Msd``, ``Vacf``, ``Rdf``) has no ``compute`` and its
  accumulator's ``push`` takes every spec input;
* **defaults parity** — a class built from its required arguments alone holds
  the spec's defaults (``tunables()``), so no default is typed in a binding;
* the time-series classes construct, compute and hand their
  ``TimeCorrelation`` on.

The other domains' seam tests are ``test_compute_{spectra,structure,order,groups}.py``.
"""

from __future__ import annotations

import inspect

import molpy as mp
import numpy as np
import pytest
from molpy.compute import Axis, ComputeSpec, RegularGrid

# The required arguments of every compute type, by spec name. A parameter
# required by one of several constructor forms (``required_in``) is given for
# one form only.
REQUIRED: dict[str, dict[str, object]] = {
    # time series
    "Correlator": {"kernel": "product", "max_lag": 4},
    "Msd": {"max_lag": 4},
    "Vacf": {"max_lag": 4},
    "OnsagerCorrelation": {"max_lag": 4},
    "DipoleAcf": {"max_lag": 4},
    "CurrentAcf": {"max_lag": 4},
    "ReorientationAcf": {"l": 2, "max_lag": 4},
    "RotationalAutocorrelation": {"l": 2, "max_lag": 4},
    "Survival": {"kind": "continuous", "max_lag": 4},
    "Derivative": {"stencil": "central"},
    "Unwrap": {},
    "Spectrum": {"window": "hann", "pad": 2},
    # fits
    "CumulativeTrapezoid": {},
    "DebyeFit": {},
    "LinearFit": {"start": 0.1, "end": 0.9},
    "Plateau": {"start": 0.5, "end": 1.0},
    # dielectric, spectroscopy, checks
    "KineticEnergy": {"units": "real"},
    "DipoleMoment": {},
    "StaticDielectric": {"units": "real"},
    "EinsteinHelfandSpectrum": {"units": "real"},
    "GreenKuboSpectrum": {"units": "real"},
    "DipoleRateCrossSpectrum": {"units": "real"},
    "DipoleAutocorrelationSpectrum": {"units": "real"},
    "KramersKronig": {},
    "ConductivitySumRule": {"units": "real"},
    "RouteAgreement": {},
    "PolarizabilityParts": {},
    "RamanSpectrum": {},
    # binned and gridded structure
    "Rdf": {"n_bins": 10, "r_max": 2.0, "r_min": 0.0},
    "CorrelationFunction": {"n_bins": 10, "r_max": 2.0, "r_min": 0.0},
    "LocalDensity": {"r_max": 1.5},
    "GaussianDensity": {"dims": [4, 4, 4], "sigma": 0.5},
    "SphereVoxelization": {"dims": [4, 4, 4], "r_max": 0.5},
    "SpatialDistribution": {"grid": RegularGrid.centered([4.0, 4.0, 4.0], [4, 4, 4])},
    "DistributionFunction": {"observable": "distance", "axis": Axis(0.0, 3.0, 10)},
    "CombinedDistribution": {
        "observables": ["distance", "angle"],
        "axes": [Axis(0.0, 3.0, 6), Axis(0.0, 180.0, 6)],
    },
    "StaticStructureFactorDebye": {"k_values": [1.0, 2.0]},
    "StaticStructureFactorDirect": {"k_max": 5.0, "n_bins": 10},
    "DiffractionPattern": {"n_grid": 16, "sigma": 0.5},
    # dynamics
    "PairPresence": {"r0": 1.0, "exclude_self": True},
    "VanHove": {"axis": Axis(0.0, 2.0, 10), "lags": [1, 2]},
    "HBonds": {},
    # order and environment
    "Steinhardt": {"l": [6]},
    "Hexatic": {"k": 6},
    "SolidLiquid": {},
    "ContinuousCoordination": {"power": 2.0},
    "Nematic": {},
    "Cubatic": {},
    "BondOrientationalOrder": {"n_theta": 4, "n_phi": 8},
    "LocalDescriptors": {"l_max": 4},
    "LocalBondProjection": {},
    "EnvironmentMatch": {"rmsd_threshold": 0.1},
    "AngularSeparationGlobal": {},
    "AngularSeparationNeighbor": {},
    "PmftR12": {"r_max": 2.0, "n_r": 4, "n_t1": 4, "n_t2": 4},
    "PmftXy": {"x_max": 2.0, "y_max": 2.0, "n_x": 4, "n_y": 4},
    "PmftXyt": {"x_max": 2.0, "y_max": 2.0, "n_x": 4, "n_y": 4, "n_t": 4},
    "PmftXyz": {
        "x_max": 2.0,
        "y_max": 2.0,
        "z_max": 2.0,
        "n_x": 4,
        "n_y": 4,
        "n_z": 4,
    },
    # groups and features
    "Cluster": {},
    "CenterOfMass": {},
    "GyrationTensor": {},
    "InertiaTensor": {},
    "Pca": {"n_components": 2},
    "Kmeans": {"k": 2},
    # voronoi
    "RadicalVoronoi": {},
    "VoronoiIntegration": {"unit": "angstrom"},
    "VoronoiDomainAnalysis": {},
    "VoronoiVoidAnalysis": {},
    "FiniteFieldPolarizability": {"field": 0.01},
}

SPECS = {spec.name: spec for spec in ComputeSpec.all()}


def _compute_classes() -> set[str]:
    """Every exported class that computes: a batch ``compute`` or a stream's
    ``accumulator``."""
    return {
        name
        for name in mp.compute.__all__
        if inspect.isclass(obj := getattr(mp.compute, name))
        and obj not in (mp.compute.Compute, mp.compute.Stream)
        and (
            callable(getattr(obj, "compute", None))
            or callable(getattr(obj, "accumulator", None))
        )
    }


STREAMS = {"Msd", "Vacf", "Rdf"}


def _is_stream(name: str) -> bool:
    """A stream only: ``accumulator`` and no batch ``compute``."""
    return name in STREAMS


def _parameters(func) -> list[tuple[str, bool, bool]]:
    """``(name, keyword_only, defaults_to_none)`` per parameter of a native
    callable."""
    return [
        (p.name, p.kind is p.KEYWORD_ONLY, p.default is None)
        for p in inspect.signature(func).parameters.values()
        if p.name != "self"
    ]


# ---------------------------------------------------------------------------
# Parity with the core descriptors
# ---------------------------------------------------------------------------


def test_the_exported_computes_are_the_specs_by_name():
    assert _compute_classes() == set(SPECS)


def test_the_required_table_covers_every_spec():
    assert set(REQUIRED) == set(SPECS)


@pytest.mark.parametrize("name", sorted(SPECS))
def test_required_arguments_are_the_spec_requirements(name):
    spec = SPECS[name]
    given = set(REQUIRED[name])
    required = {p.name for p in spec.params if p.required and p.required_in is None}
    alternatives = {p.name for p in spec.params if p.required_in is not None}
    assert required <= given
    assert given - required <= alternatives


@pytest.mark.parametrize("name", sorted(SPECS))
def test_init_takes_the_spec_parameters(name):
    spec = SPECS[name]
    params = _parameters(getattr(mp.compute, name))
    assert [p[0] for p in params] == [p.name for p in spec.params]
    for (key, keyword_only, none), described in zip(params, spec.params, strict=True):
        if described.required and described.required_in is None:
            assert not none, key
        else:
            assert none and keyword_only, key


@pytest.mark.parametrize("name", sorted(SPECS))
def test_compute_takes_the_spec_inputs_by_keyword(name):
    spec = SPECS[name]
    cls = getattr(mp.compute, name)
    if _is_stream(name):
        assert not hasattr(cls, "compute"), name
        acc = cls(**REQUIRED[name]).accumulator()
        params = dict((p[0], p) for p in _parameters(acc.push))
        for described in spec.inputs:
            assert described.name in params, described.name
            assert params[described.name][2] == described.optional, described.name
        extra = set(params) - {i.name for i in spec.inputs}
        assert all(params[key][2] for key in extra), extra
        return
    params = _parameters(cls.compute)
    assert [p[0] for p in params] == [i.name for i in spec.inputs]
    for (key, keyword_only, none), described in zip(params, spec.inputs, strict=True):
        assert keyword_only, key
        assert none == described.optional, key


def _expected_default(spec, param, given):
    if param.same_as is not None:
        return given[param.same_as]
    if param.scaled is not None:
        of, factor = param.scaled
        return factor * given[of]
    return param.default


@pytest.mark.parametrize("name", sorted(SPECS))
def test_required_arguments_alone_give_the_spec_defaults(name):
    spec = SPECS[name]
    given = REQUIRED[name]
    instance = getattr(mp.compute, name)(**given)
    defaulted = [p for p in spec.params if not p.required]
    tunables = instance.tunables()
    assert [key for key, _ in tunables] == [p.name for p in defaulted]
    for (key, value), param in zip(tunables, defaulted, strict=True):
        expected = _expected_default(spec, param, given)
        if isinstance(expected, float):
            assert value == pytest.approx(expected), key
        else:
            assert value == expected, key


def test_spec_is_reachable_from_the_class():
    assert mp.compute.Msd.spec().name == "Msd"
    assert ComputeSpec.find("Msd").name == "Msd"
    assert ComputeSpec.find("NotACompute") is None


def test_spec_describes_parameters_and_inputs():
    spec = ComputeSpec.find("Correlator")
    assert spec.form == "single"
    assert spec.param("kernel").kind == "choice"
    assert spec.param("kernel").choices == [
        "product",
        "squared_displacement",
        "legendre",
        "wigner_d",
        "persistence",
    ]
    assert spec.param("kernel").required
    assert spec.param("l").kind == "int"
    assert not spec.param("l").required
    origins = spec.param("origins")
    assert origins.kind == "choice"
    assert origins.default in origins.choices
    assert spec.input("b").optional
    assert spec.input("a").kind == "trajectory"
    assert ComputeSpec.find("Rdf").form == "frames"
    assert (
        ComputeSpec.find("LinearFit").input("correlation").upstream == "TimeCorrelation"
    )
    assert ComputeSpec.find("PairPresence").param("r1").same_as == "r0"


def test_window_defaults_to_the_core_window():
    """The four routes default to the core ``Window`` default (``cosine_sq``),
    not a literal a binding once carried."""
    for name in (
        "EinsteinHelfandSpectrum",
        "GreenKuboSpectrum",
        "DipoleRateCrossSpectrum",
        "DipoleAutocorrelationSpectrum",
    ):
        tunables = dict(getattr(mp.compute, name)(units="real").tunables())
        assert tunables["window"] == "cosine_sq"


# ---------------------------------------------------------------------------
# Time series: construct, compute, hand the TimeCorrelation on
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(7)
T, N = 24, 5


def _walk() -> np.ndarray:
    return np.cumsum(RNG.normal(scale=0.1, size=(T, N, 3)), axis=0)


def _unit_vectors() -> np.ndarray:
    v = RNG.normal(size=(T, N, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _unit_quaternions() -> np.ndarray:
    q = RNG.normal(size=(T, N, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _stream(stream, frames: np.ndarray, *, dt: float) -> mp.compute.TimeCorrelation:
    """Push every frame of ``frames`` into a fresh accumulator; finish."""
    acc = stream.accumulator()
    for frame in frames:
        acc.push(frame, dt=dt)
    assert acc.n_frames == len(frames)
    return acc.finish()


class TestTimeSeries:
    def test_correlator_every_kernel(self):
        walk = _walk()
        for kernel, l, data in (
            ("product", None, walk),
            ("squared_displacement", None, walk),
            ("legendre", 2, _unit_vectors()),
            ("wigner_d", 1, _unit_quaternions()),
        ):
            tc = mp.compute.Correlator(kernel, 5, l=l).compute(a=data, dt=0.5)
            assert isinstance(tc, mp.compute.TimeCorrelation)
            assert tc.n_lags == 6
            assert tc.lag_times[1] == pytest.approx(0.5)

    def test_correlator_cross_and_options(self):
        a, b = _walk(), _walk()
        tc = mp.compute.Correlator(
            "product", 3, origins="first", mean="subtract", reduction="per_item"
        ).compute(a=a, b=b, dt=1.0)
        assert tc.values.shape == (4, N)

    def test_unknown_kernel_and_choice_raise_the_core_message(self):
        with pytest.raises(ValueError, match='unknown LagKernelKind "spline"'):
            mp.compute.Correlator("spline", 3)
        with pytest.raises(ValueError, match="required by legendre"):
            mp.compute.Correlator("legendre", 3)
        with pytest.raises(ValueError, match="only legendre / wigner_d"):
            mp.compute.Correlator("product", 3, l=2)
        with pytest.raises(ValueError, match="unknown Origins"):
            mp.compute.Msd(3, origins="last")

    def test_msd_equals_its_correlator(self):
        walk = _walk()
        msd = _stream(mp.compute.Msd(6), walk, dt=1.0)
        same = mp.compute.Correlator("squared_displacement", 6).compute(a=walk, dt=1.0)
        # The stream equals the batch correlator to FFT round-off (core
        # `CorrelatorAccumulator` contract).
        np.testing.assert_allclose(msd.values, same.values, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(msd.lag_times, same.lag_times)

    def test_msd_with_boxes_refuses_a_wrapped_trajectory(self):
        box = mp.core.Box(mp.core.Cell.lengths([1.0] * 3))
        jump = np.zeros((3, 1, 3))
        jump[1, 0, 0] = 0.9
        acc = mp.compute.Msd(2).accumulator()
        acc.push(jump[0], dt=1.0, simbox=box)
        with pytest.raises(ValueError, match="wrapped"):
            acc.push(jump[1], dt=1.0, simbox=box)
        assert acc.n_frames == 1

    def test_series_surfaces(self):
        series = RNG.normal(size=(T, 3))
        for cls, key in (
            (mp.compute.DipoleAcf, "dipoles"),
            (mp.compute.CurrentAcf, "current"),
        ):
            tc = cls(4).compute(**{key: series, "dt": 1.0})
            assert tc.n_lags == 5
        vacf = _stream(mp.compute.Vacf(4), series[:, :, None], dt=1.0)
        assert vacf.n_lags == 5
        onsager = mp.compute.OnsagerCorrelation(4).compute(a=series, b=series, dt=1.0)
        assert onsager.n_lags == 5

    def test_orientation_surfaces(self):
        tc = mp.compute.ReorientationAcf(1, 4).compute(vectors=_unit_vectors(), dt=1.0)
        assert tc.values[0, 0] == pytest.approx(1.0)
        rot = mp.compute.RotationalAutocorrelation(1, 4).compute(
            orientations=_unit_quaternions(), dt=1.0
        )
        assert rot.values[0, 0] == pytest.approx(1.0)

    def test_survival_of_a_presence_series(self):
        presence = np.ones((T, 2))
        tc = mp.compute.Survival("continuous", 3).compute(presence=presence, dt=1.0)
        np.testing.assert_allclose(tc.values[:, 0], 1.0)

    def test_convention_options_reach_the_core(self):
        """The previous conventions are ``__init__`` keywords parsed by the
        core; their data rides on ``compute``."""
        presence = np.ones((4, 2))
        per_reference = mp.compute.Survival(
            "intermittent", 1, normalization="per_reference"
        ).compute(presence=presence, dt=1.0, n_reference=1)
        np.testing.assert_allclose(per_reference.values[:, 0], 2.0)
        inner = np.zeros((4, 2))
        inner[0] = 1.0
        gated = mp.compute.Survival("continuous", 1, origin="inner").compute(
            presence=presence, dt=1.0, inner=inner
        )
        assert gated.values[0, 0] == pytest.approx(2.0 / 8.0)
        with pytest.raises(ValueError, match="n_reference"):
            mp.compute.Survival("continuous", 1).compute(
                presence=presence, dt=1.0, n_reference=1
            )
        raw = mp.compute.CurrentAcf(2, normalization="none")
        assert dict(raw.tunables())["normalization"] == "none"
        with pytest.raises(ValueError, match="unknown SurvivalOrigin"):
            mp.compute.Survival("continuous", 1, origin="born")

    def test_derivative_unwrap_spectrum(self):
        series = np.arange(10.0)[:, None] * np.ones((1, 2))
        d = mp.compute.Derivative("central").compute(series=series, dt=2.0)
        np.testing.assert_allclose(d[1:-1], 0.5)

        box = mp.core.Box(mp.core.Cell.lengths([1.0] * 3))
        wrapped = np.zeros((3, 1, 3))
        wrapped[:, 0, 0] = [0.8, 0.95, 0.1]
        unwrapped = mp.compute.Unwrap().compute(positions=wrapped, boxes=[box] * 3)
        assert unwrapped[2, 0, 0] == pytest.approx(1.1)

        spectrum = mp.compute.Spectrum("none", 1).compute(signal=np.ones(8), dt=1.0)
        assert isinstance(spectrum, mp.compute.OneSidedSpectrum)
        assert spectrum.frequencies.shape == spectrum.real.shape

    def test_time_correlation_is_a_value(self):
        tc = mp.compute.TimeCorrelation(np.arange(4.0), np.ones((4, 2)))
        assert (tc.n_lags, tc.n_channels) == (4, 2)
        np.testing.assert_array_equal(tc.channel(1), np.ones(4))
        with pytest.raises(ValueError, match="channel out of range"):
            tc.channel(2)
        fit = mp.compute.LinearFit(0.0, 1.0).compute(correlation=tc, channel=0)
        assert fit.slope == pytest.approx(0.0)

    def test_msd_feeds_linear_fit(self):
        t = np.arange(T, dtype=float)
        ballistic = np.zeros((T, 1, 3))
        ballistic[:, 0, 0] = t
        msd = _stream(mp.compute.Msd(8), ballistic, dt=1.0)
        fit = mp.compute.LinearFit(0.5, 1.0).compute(correlation=msd, channel=0)
        assert fit.r2 > 0.9
