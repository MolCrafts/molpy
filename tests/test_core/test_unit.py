"""Units are molrs's: ``mp.UnitRegistry`` / ``mp.UnitPreset`` by identity.

molpy keeps no unit registry of its own; what ``UnitSystem`` used to add — the
``openmm`` preset, preset registration, ``k_B`` and LJ reduced units — is
native. These tests pin the molpy-facing contract of those names.
"""

import molrs
import pytest

import molpy as mp

REQUIRED_DIMS = ("mass", "length", "time", "energy", "temperature", "charge")


def test_unit_names_are_molrs_objects():
    for name in ("Unit", "Quantity", "UnitRegistry", "UnitPreset", "UnitsError"):
        assert getattr(mp, name) is getattr(molrs.units, name)


def test_quantity_and_conversion():
    units = mp.UnitRegistry()
    quantity = 1.5 * units.angstrom
    assert isinstance(quantity, mp.Quantity)
    assert quantity.to("nanometer").magnitude == pytest.approx(0.15)
    assert (1.0 * units.kilocalorie_per_mole).to("eV").magnitude == pytest.approx(
        0.0433641, rel=1e-5
    )


def test_every_lammps_preset_and_openmm_resolve():
    names = {"real", "metal", "si", "cgs", "electron", "micro", "nano", "openmm"}
    assert names <= set(mp.UnitPreset.names())
    units = mp.UnitRegistry()
    for name in names:
        preset = mp.UnitPreset(name)
        for dim in REQUIRED_DIMS:
            units.parse(getattr(preset, dim)())


def test_openmm_preset_is_kj_nm():
    units = mp.UnitRegistry()
    preset = mp.UnitPreset("openmm")
    assert units.parse(preset.energy()) == units.kilojoule_per_mole
    assert units.parse(preset.length()) == units.nanometer


def test_a_registered_preset_is_named_and_refuses_a_taken_name():
    base = mp.UnitPreset("real")
    dims = (
        "mass",
        "length",
        "time",
        "energy",
        "temperature",
        "charge",
        "pressure",
        "velocity",
        "force",
        "density",
    )
    units = {dim: getattr(base, dim)() for dim in dims}
    preset = mp.UnitPreset.register(
        "test_molpy_md",
        {**units, "length": "nanometer"},
        boltzmann=base.boltzmann(),
        coulomb=base.coulomb(),
    )
    assert preset.length() == "nanometer"
    assert "test_molpy_md" in mp.UnitPreset.names()
    with pytest.raises(ValueError):
        mp.UnitPreset.register(
            "real", units, boltzmann=base.boltzmann(), coulomb=base.coulomb()
        )


def test_boltzmann_constant_is_in_the_default_registry():
    units = mp.UnitRegistry()
    assert (1.0 * units.k_B).to("electron_volt / kelvin").magnitude == pytest.approx(
        8.617333262e-5, rel=1e-12
    )


def test_lj_reduced_units():
    argon = mp.UnitRegistry()
    argon.define_lj_units(
        39.948 * argon.amu, 3.405 * argon.angstrom, 0.2381 * argon.kilocalorie_per_mole
    )
    assert (3.405 * argon.angstrom).to(argon.lj_sigma).magnitude == pytest.approx(1.0)
    assert (1.0 * argon.lj_tau).to(argon.ps).magnitude == pytest.approx(2.16, abs=0.01)
    assert (1.0 * argon.lj_epsilon_over_kB).to(argon.K).magnitude == pytest.approx(
        119.8, abs=0.5
    )


def test_lj_rejects_wrong_dimensions():
    units = mp.UnitRegistry()
    with pytest.raises(mp.UnitsError):
        units.define_lj_units(1.0 * units.second, 1.0 * units.angstrom, 1.0 * units.eV)
