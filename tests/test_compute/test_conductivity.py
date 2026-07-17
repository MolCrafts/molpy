"""Tests for IonicConductivity compute and the NumPy Debye fit."""

import numpy as np
import pytest

from molpy.compute import ConductivityResult, DebyeFit, IonicConductivity
from molpy.compute.base import Compute
from molpy.compute.result import DielectricResult
from molpy.core.box import Box
from molrs import Block, Frame


class ListTrajectory:
    def __init__(self, frames):
        self._frames = frames

    def __getitem__(self, idx):
        return self._frames[idx]

    def __len__(self):
        return len(self._frames)

    def __iter__(self):
        return iter(self._frames)


def make_ion_trajectory(n_frames=400, drift=0.01):
    """Two ions (+1, -1); the cation drifts linearly so M_J grows in time."""
    frames = []
    for i in range(n_frames):
        block = Block()
        # cation drifts along x; anion fixed -> M_J(t) = q_+ * x_+(t) grows
        block["x"] = np.array([1.0 + drift * i, 5.0])
        block["y"] = np.array([0.0, 0.0])
        block["z"] = np.array([0.0, 0.0])
        block["charge"] = np.array([1.0, -1.0])
        frame = Frame()
        frame["atoms"] = block
        frame.simbox = Box.cubic(30.0)
        frames.append(frame)
    return ListTrajectory(frames)


class TestIonicConductivity:
    def test_is_compute(self):
        assert issubclass(IonicConductivity, Compute)

    def test_returns_conductivity_result(self):
        traj = make_ion_trajectory()
        comp = IonicConductivity(dt=2.0, temperature=298.15, max_correlation_time=100)
        res = comp(traj)
        assert isinstance(res, ConductivityResult)
        assert res.time.shape == res.msd.shape
        assert res.msd[0] == 0.0
        # Linear drift -> positive MSD slope -> positive conductivity.
        assert res.slope > 0.0
        assert res.sigma > 0.0

    def test_raises_missing_charge(self):
        traj = make_ion_trajectory()
        for fr in traj:
            del fr["atoms"]["charge"]
        comp = IonicConductivity(dt=2.0, temperature=298.15, max_correlation_time=50)
        with pytest.raises(ValueError, match="column"):
            comp(traj)

    def test_raises_insufficient_frames(self):
        traj = make_ion_trajectory(n_frames=1)
        comp = IonicConductivity(dt=2.0, temperature=298.15, max_correlation_time=50)
        with pytest.raises(ValueError, match="at least 2"):
            comp(traj)

    def test_immutability(self):
        traj = make_ion_trajectory(n_frames=50)
        x0 = traj[0]["atoms"]["x"].copy()
        comp = IonicConductivity(dt=2.0, temperature=298.15, max_correlation_time=20)
        comp(traj)
        assert np.array_equal(traj[0]["atoms"]["x"], x0)


class TestDebyeFit:
    def _debye_result(self, tau0=5.0, delta=40.0, eps_inf=1.0, n=513, dt=0.5):
        n_pad = 2 * (n - 1)
        # rfft-style angular-frequency grid, rad/ps
        freq = 2.0 * np.pi * np.fft.rfftfreq(n_pad, d=dt)
        x = freq * tau0
        denom = 1.0 + x * x
        er = eps_inf + delta / denom
        ei = delta * x / denom
        er[0] = eps_inf + delta  # exact static DC bin
        ei[0] = 0.0
        return DielectricResult(
            frequency=freq,
            epsilon_real=er,
            epsilon_imag=ei,
            epsilon_static=eps_inf + delta,
            epsilon_inf=eps_inf,
            route="einstein-helfand",
            component="full",
        )

    def test_recovers_tau_and_delta(self):
        res = self._debye_result(tau0=5.0, delta=40.0, eps_inf=1.0)
        fit = res.fit_debye()
        assert isinstance(fit, DebyeFit)
        assert fit.tau == pytest.approx(5.0, rel=0.05)
        assert fit.delta_eps == pytest.approx(40.0, rel=1e-6)
        assert fit.eps_inf == 1.0
        # loss peak of a Debye sits at omega = 1/tau
        assert fit.omega_peak == pytest.approx(1.0 / 5.0, rel=0.25)

    def test_epsilon_roundtrip(self):
        res = self._debye_result(tau0=3.0, delta=20.0, eps_inf=1.5)
        fit = res.fit_debye()
        er, ei = fit.epsilon(res.frequency)
        # reconstructed model matches the input spectrum it was fit to
        assert np.allclose(er, res.epsilon_real, atol=0.05)
        assert np.allclose(ei, res.epsilon_imag, atol=0.05)
