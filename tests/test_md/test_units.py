"""UnitPreset is the MD-facing constants view."""

from molpy import UnitPreset


def test_real_boltzmann_is_positive():
    kb = UnitPreset("real").boltzmann()
    assert isinstance(kb, float)
    assert kb > 0.0


def test_real_energy_and_time_names():
    p = UnitPreset("real")
    assert p.energy() == "kilocalorie_per_mole"
    assert p.time() == "femtosecond"
