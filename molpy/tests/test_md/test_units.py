"""UnitPreset is the MD-facing constants view."""

import molpy as mp


def test_real_boltzmann_is_positive():
    kb = mp.core.UnitPreset("real").boltzmann()
    assert isinstance(kb, float)
    assert kb > 0.0


def test_real_energy_and_time_names():
    p = mp.core.UnitPreset("real")
    assert p.energy() == "kilocalorie_per_mole"
    assert p.time() == "femtosecond"
