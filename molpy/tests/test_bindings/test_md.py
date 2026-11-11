"""Seam tests for mp.md — the phase driver, not a potential."""

from __future__ import annotations

import molpy as mp
import numpy as np
import pytest
from molpy.ff.potential import PairLjCut
from molpy.core import NeighborList
from molpy.md import MD, MaxwellBoltzmann, VelocityVerlet


def _pair(edge: float = 10.0) -> mp.core.Frame:
    atoms = mp.core.Block()
    pos = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    for d, key in enumerate("xyz"):
        atoms.insert(key, np.ascontiguousarray(pos[:, d]))
    for key in ("vx", "vy", "vz"):
        atoms.insert(key, np.zeros(2))
    atoms.insert("mass", np.ones(2))
    frame = mp.core.Frame()
    frame["atoms"] = atoms
    frame.box = mp.core.Box(mp.core.Cell.lengths([edge, edge, edge]))
    return frame


class TestDriver:
    def test_pair_lj_one_step_writes_the_frame(self) -> None:
        frame = _pair()
        sim = MD(units="real")
        sim.set_timestep(0.001)
        sim.set_integrator(VelocityVerlet())
        sim.set_neighbor(NeighborList(3.0), skin=0.5)
        sim.set_evaluator(PairLjCut(1.0, 1.0, 2.5))
        sim.run(frame, 1)
        energy = frame.meta["energy"]
        assert energy != 0.0
        assert np.isfinite(energy)
        fx = np.asarray(frame["atoms"]["fx"])
        assert fx.shape == (2,)
        assert np.isfinite(fx).all()

    def test_a_missing_neighbor_list_is_named(self) -> None:
        sim = MD(units="real")
        sim.set_timestep(0.001)
        sim.set_integrator(VelocityVerlet())
        sim.set_evaluator(PairLjCut(1.0, 1.0, 2.5))
        with pytest.raises(ValueError, match="set_neighbor"):
            sim.run(_pair(), 1)


class TestUnits:
    def test_the_default_style_is_lj_at_its_lammps_timestep(self) -> None:
        sim = MD()
        assert sim.units == "lj"
        assert sim.timestep == 0.005

    def test_a_style_sets_its_lammps_timestep(self) -> None:
        assert MD(units="real").timestep == 1.0
        assert MD(units="metal").timestep == 0.001

    def test_units_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            MD("real")  # type: ignore[misc]

    def test_a_name_that_is_no_lammps_style_is_refused(self) -> None:
        with pytest.raises(mp.core.UnitsError, match="openmm"):
            MD(units="openmm")  # type: ignore[arg-type]

    def test_a_force_field_in_another_style_is_refused(self) -> None:
        def bonded(units: str) -> mp.ff.potential.WeightedTerms:
            ff = mp.ff.forcefield.ForceField("pair", units=units)
            a = ff.def_style("atom", "full").def_type("A", mass=1.0, charge=0.0)
            ff.def_style("bond", "harmonic").def_type("A-A", a, a, k=100.0, r0=1.0)
            frame = _pair()
            frame["atoms"].insert("type", ["A", "A"])
            bonds = mp.core.Block()
            bonds.insert("atomi", np.array([0], dtype=np.uint32))
            bonds.insert("atomj", np.array([1], dtype=np.uint32))
            bonds.insert("type", ["A-A"])
            frame["bonds"] = bonds
            return mp.ff.PotentialCompiler(ff).compile_typed(frame)

        sim = MD(units="real")
        sim.set_evaluator(bonded("metal"))
        sim.set_integrator(VelocityVerlet())
        with pytest.raises(ValueError, match="'metal'.*'real'"):
            sim.run(_pair(), 0)
        sim.set_evaluator(bonded("real"))
        sim.run(_pair(), 0)


class TestMaxwellBoltzmann:
    def test_temperature_constructor(self) -> None:
        mb = MaxwellBoltzmann(300.0, units="real", seed=1)
        assert mb.temperature == 300.0
        vel = mb.velocities(np.zeros((4, 3)), np.ones(4))
        assert vel.shape == (4, 3)
        assert np.all(np.isfinite(vel))

    def test_a_scalar_mass_is_refused(self) -> None:
        with pytest.raises(TypeError, match="scalar"):
            MaxwellBoltzmann(1.0, units="real").velocities(np.zeros((2, 3)), 1.0)


class TestMass:
    @pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
    def test_a_non_positive_or_non_finite_mass_is_refused(self, invalid) -> None:
        with pytest.raises(ValueError, match="mass"):
            MaxwellBoltzmann(1.0, units="real").velocities(
                np.zeros((1, 3)), np.array([invalid])
            )
