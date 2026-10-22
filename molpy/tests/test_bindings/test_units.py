import molpy as mp
import pytest


def test_registry_parses_and_converts_molecular_units():
    units = mp.core.UnitRegistry()
    energy = units.quantity(1.0, "kilocalorie_per_mole")
    assert energy.to("kilojoule_per_mole").value == pytest.approx(4.184)
    assert energy.to("eV").value == pytest.approx(0.0433641, rel=1e-5)


def test_spelled_out_si_prefixes_are_native():
    units = mp.core.UnitRegistry()
    assert units.parse("nanometer").factor_to(units.angstrom) == pytest.approx(10.0)
    assert units.parse("femtosecond").factor_to(units.second) == pytest.approx(1e-15)


def test_quantity_arithmetic_and_dimension_errors():
    units = mp.core.UnitRegistry()
    total = 1.0 * units.nanometer + 5.0 * units.angstrom
    assert total.value == pytest.approx(1.5)
    assert total.unit == units.nanometer

    speed = (2.0 * units.nanometer) / (4.0 * units.picosecond)
    assert speed.to("meter / second").value == pytest.approx(500.0)

    with pytest.raises(mp.core.UnitsError, match="dimension mismatch"):
        (1.0 * units.meter).to("second")


def test_custom_definition_is_registry_local():
    units = mp.core.UnitRegistry()
    units.define("smoot", 1.7018, units.meter.dimension)
    assert (1.0 * units.smoot).to(units.meter).value == pytest.approx(1.7018)
    with pytest.raises(AttributeError):
        _ = mp.core.UnitRegistry().smoot


def test_affine_temperature_conversion():
    units = mp.core.UnitRegistry()
    assert units.quantity(25.0, "degC").to("K").value == pytest.approx(298.15)


def test_define_lj_sigma_defines_the_reduced_length_unit_alone():
    units = mp.core.UnitRegistry()
    units.define_lj_sigma(units.quantity(4.2, "angstrom"))
    assert units.parse("lj_sigma").factor_to(units.angstrom) == pytest.approx(
        4.2, rel=1e-12
    )
    # Only sigma is known, so no reduced mass exists.
    with pytest.raises(mp.core.UnitsError):
        units.parse("lj_mass")


def test_define_lj_sigma_refuses_a_sigma_that_is_not_a_length():
    units = mp.core.UnitRegistry()
    with pytest.raises(mp.core.UnitsError):
        units.define_lj_sigma(units.quantity(1.0, "second"))


def test_openmm_preset_is_native_with_molrs_boltzmann():
    preset = mp.core.UnitPreset("openmm")
    assert preset.length() == "nanometer"
    assert preset.energy() == "kilojoule_per_mole"
    real = mp.core.UnitPreset("real")
    units = mp.core.UnitRegistry()
    kj = units.quantity(real.boltzmann(), "kilocalorie_per_mole").to(
        "kilojoule_per_mole"
    )
    assert preset.boltzmann() == pytest.approx(kj.value, rel=1e-14)
    assert "openmm" in mp.core.UnitPreset.names()


def test_boltzmann_constant_is_a_registry_unit():
    units = mp.core.UnitRegistry()
    rt = (300.0 * units.parse("k_B * kelvin")).to("kilojoule_per_mole")
    assert rt.value == pytest.approx(2.494338785, rel=1e-9)


def test_register_a_custom_preset():
    table = {
        dim: getattr(mp.core.UnitPreset("real"), dim)()
        for dim in (
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
    }
    table["length"] = "nanometer"
    preset = mp.core.UnitPreset.register(
        "real_nm_test", table, boltzmann=1.0, coulomb=2.0
    )
    assert mp.core.UnitPreset("real_nm_test").length() == "nanometer"
    assert preset.coulomb() == 2.0
    with pytest.raises(ValueError, match="already"):
        mp.core.UnitPreset.register("real_nm_test", table, boltzmann=1.0, coulomb=2.0)
    mp.core.UnitPreset.register(
        "real_nm_test", table, boltzmann=3.0, coulomb=2.0, overwrite=True
    )
    assert mp.core.UnitPreset("real_nm_test").boltzmann() == 3.0
    del table["mass"]
    with pytest.raises(ValueError, match="mass"):
        mp.core.UnitPreset.register("broken", table, boltzmann=1.0, coulomb=1.0)
