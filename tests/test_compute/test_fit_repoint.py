"""Regression lock for the compute/fit repoint of molpy wrappers.

Phase 02 repoints :class:`IonicConductivity` (Einstein-Helfand) and
:class:`JACF` (Green-Kubo) from the deprecated molrs bundled free functions onto
the explicit raw-compute + fit classes (``EinsteinConductivity`` + ``LinearFit``,
``GreenKuboConductivity`` + ``RunningIntegral``). These tests pin each wrapper's
public output to its pre-migration value and assert the wrappers no longer route
through a deprecated binding (no ``DeprecationWarning`` raised).

The pre-migration reference values were captured against the molrs wheel before
the repoint and are exact (the underlying OLS / trapezoid arithmetic is
identical, lifted into the shared Rust helpers).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import molpy as mp
from molpy.compute import IonicConductivity, JACF
from molpy.compute.result import ConductivityResult, JACFResult
from molpy.core.trajectory import Trajectory


# ── fixtures ─────────────────────────────────────────────────────────────────


def _ion_trajectory(n=400, drift=0.01):
    """Cation drifts linearly along x; anion fixed -> growing collective MSD."""
    frames = []
    for i in range(n):
        f = mp.Frame()
        f["atoms"] = {
            "x": np.array([1.0 + drift * i, 5.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "charge": np.array([1.0, -1.0]),
        }
        f.box = mp.Box.cubic(30.0)
        frames.append(f)
    return Trajectory(frames)


def _current_trajectory(n=11, box_len=10.0, vcat=1.0):
    """Constant current J=(vcat,0,0): flat ACF, plateau Green-Kubo integral."""
    frames = []
    for _ in range(n):
        f = mp.Frame()
        f["atoms"] = {
            "x": np.array([0.0, 5.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "vx": np.array([vcat, 0.0]),
            "vy": np.array([0.0, 0.0]),
            "vz": np.array([0.0, 0.0]),
            "type": np.array([1, 2]),
        }
        f.box = mp.Box.cubic(box_len)
        frames.append(f)
    return Trajectory(frames)


# ── ac-006: public output preserved through delegation ───────────────────────

# Pre-migration reference (captured against the molrs wheel before the repoint).
_EH_SIGMA = 0.0011548057330809818
_EH_SLOPE = 0.003
_EH_FIT_START = 10
_EH_FIT_END = 50
_EH_MSD_50 = 0.25

_GK_SIGMA = 82.633279036342714


def test_ionic_conductivity_output_pinned():
    res = IonicConductivity(dt=2.0, temperature=298.15, max_correlation_time=100)(
        _ion_trajectory()
    )
    assert isinstance(res, ConductivityResult)
    assert res.sigma == pytest.approx(_EH_SIGMA, rel=1e-12)
    assert res.slope == pytest.approx(_EH_SLOPE, rel=1e-12)
    assert res.fit_start == _EH_FIT_START
    assert res.fit_end == _EH_FIT_END
    assert res.msd[50] == pytest.approx(_EH_MSD_50, rel=1e-12)
    assert res.msd[0] == 0.0
    assert res.time.shape == res.msd.shape


def test_jacf_output_pinned():
    res = JACF(cation_type=1, anion_type=2, max_dt=5.0, dt=1.0, temperature=300.0)(
        _current_trajectory()
    )
    assert isinstance(res, JACFResult)
    np.testing.assert_allclose(res.jacf, np.ones(5), atol=1e-12)
    assert res.sigma == pytest.approx(_GK_SIGMA, rel=1e-12)
    # Running integral grows linearly toward the plateau sigma at the last lag.
    assert res.sigma_running[-1] == pytest.approx(_GK_SIGMA, rel=1e-12)
    assert res.sigma_running[0] == 0.0


# ── repoint proof: wrappers no longer hit a deprecated binding ────────────────


def test_ionic_conductivity_emits_no_deprecation_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        IonicConductivity(dt=2.0, temperature=298.15, max_correlation_time=50)(
            _ion_trajectory(n=120)
        )


def test_jacf_emits_no_deprecation_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        JACF(cation_type=1, anion_type=2, max_dt=5.0, dt=1.0, temperature=300.0)(
            _current_trajectory()
        )


# ── delegation parity: molpy path == direct molrs raw+fit ────────────────────


def test_ionic_conductivity_matches_direct_molrs_raw_plus_fit():
    import molrs

    traj = _ion_trajectory(n=200)
    dt, temperature, mct = 2.0, 298.15, 80
    start_frac, end_frac = 0.1, 0.5
    volume = 30.0**3

    res = IonicConductivity(
        dt=dt,
        temperature=temperature,
        max_correlation_time=mct,
        fit_start_frac=start_frac,
        fit_end_frac=end_frac,
    )(traj)

    # Rebuild the same translational dipole the wrapper builds (no unwrap needed:
    # the cation stays inside the box over this short window).
    frames = list(traj)
    n_frames = len(frames)
    pos = np.empty((n_frames, 2, 3))
    for i, fr in enumerate(frames):
        pos[i, :, 0] = fr["atoms"]["x"]
        pos[i, :, 1] = fr["atoms"]["y"]
        pos[i, :, 2] = fr["atoms"]["z"]
    charges = np.asarray(frames[0]["atoms"]["charge"], dtype=np.float64)
    dipole = np.ascontiguousarray(np.einsum("a,fad->fd", charges, pos))

    raw = molrs.EinsteinConductivity().compute(dipole, dt, mct)
    fit = molrs.LinearFit(start_frac, end_frac).fit(raw["lag_times"], raw["msd"])
    assert res.slope == pytest.approx(fit["slope"], rel=1e-12)
    assert res.fit_start == fit["fit_start"]
    assert res.fit_end == fit["fit_end"]
    np.testing.assert_allclose(res.msd, raw["msd"], rtol=1e-12)


def test_jacf_matches_direct_molrs_raw_plus_fit():
    import molrs

    traj = _current_trajectory()
    dt, mct = 1.0, 4
    current = np.tile([1.0, 0.0, 0.0], (11, 1))
    raw = molrs.GreenKuboConductivity().compute(np.ascontiguousarray(current), dt, mct)

    res = JACF(cation_type=1, anion_type=2, max_dt=5.0, dt=dt, temperature=300.0)(traj)
    np.testing.assert_allclose(res.jacf, raw["jacf"], rtol=1e-12)
