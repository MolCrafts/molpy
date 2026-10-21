"""Seam tests for molrs.md — the phase driver, not a potential."""

from __future__ import annotations

import molrs
import numpy as np
import pytest
from molrs.ff.potential import PairLjCut
from molrs.md import MD, MaxwellBoltzmann, NeighborList, VelocityVerlet


def _pair(edge: float = 10.0) -> molrs.core.Frame:
    atoms = molrs.core.Block()
    pos = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    for d, key in enumerate("xyz"):
        atoms.insert(key, np.ascontiguousarray(pos[:, d]))
    for key in ("vx", "vy", "vz"):
        atoms.insert(key, np.zeros(2))
    atoms.insert("mass", np.ones(2))
    frame = molrs.core.Frame()
    frame["atoms"] = atoms
    frame.box = molrs.core.Box(molrs.core.Cell.lengths([edge, edge, edge]))
    return frame


class TestPotentialProtocol:
    def test_md_defines_no_potential(self) -> None:
        from molrs import md

        for name in ("Potential", "Potentials", "PairLjCut"):
            assert not hasattr(md, name)
            assert not hasattr(molrs._native.md, name)


class TestDriver:
    def test_pair_lj_one_step_writes_the_frame(self) -> None:
        frame = _pair()
        sim = MD()
        sim.set_integrator(VelocityVerlet(0.001))
        sim.set_neighbor(NeighborList(2.5, 0.5))
        sim.set_evaluator(PairLjCut(1.0, 1.0, 2.5))
        sim.run(frame, 1)
        energy = frame.meta["energy"]
        assert energy != 0.0
        assert np.isfinite(energy)
        fx = np.asarray(frame["atoms"]["fx"])
        assert fx.shape == (2,)
        assert np.isfinite(fx).all()

    def test_a_missing_neighbor_list_is_named(self) -> None:
        sim = MD()
        sim.set_integrator(VelocityVerlet(0.001))
        sim.set_evaluator(PairLjCut(1.0, 1.0, 2.5))
        with pytest.raises(ValueError, match="set_neighbor"):
            sim.run(_pair(), 1)


class TestMaxwellBoltzmann:
    def test_kbt_constructor(self) -> None:
        mb = MaxwellBoltzmann(molrs.core.UnitPreset("real").boltzmann() * 300.0, seed=1)
        vel = mb.velocities(np.zeros((4, 3)), np.ones(4))
        assert vel.shape == (4, 3)
        assert np.all(np.isfinite(vel))

    def test_a_scalar_mass_is_refused(self) -> None:
        with pytest.raises(TypeError, match="scalar"):
            MaxwellBoltzmann(1.0).velocities(np.zeros((2, 3)), 1.0)


class TestMass:
    @pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
    def test_a_non_positive_or_non_finite_mass_is_refused(self, invalid) -> None:
        with pytest.raises(ValueError, match="mass"):
            MaxwellBoltzmann(1.0).velocities(np.zeros((1, 3)), np.array([invalid]))
