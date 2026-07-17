import pytest

import molpy
import molrs
from molpy import UnitSystem


REQUIRED_DIMS = {
    "mass",
    "length",
    "time",
    "energy",
    "temperature",
    "charge",
    "pressure",
}


def test_unit_system_is_native_registry_sugar():
    assert issubclass(UnitSystem, molrs.UnitRegistry)
    assert molpy.UnitSystem is UnitSystem
    assert not hasattr(UnitSystem(), "_inner")


def test_native_quantity_and_conversion_contract():
    units = UnitSystem()
    quantity = 1.5 * units.angstrom
    assert isinstance(quantity, molrs.Quantity)
    assert quantity.magnitude == pytest.approx(1.5)
    assert quantity.to("nanometer").magnitude == pytest.approx(0.15)
    assert (1.0 * units.kilocalorie_per_mole).to("eV").magnitude == pytest.approx(
        0.0433641, rel=1e-5
    )


def test_constructor_records_native_base_units():
    units = UnitSystem(base_units={"length": "nm", "time": "ps"})
    assert isinstance(units.base_units["length"], molrs.Unit)
    assert units.base_units["length"] == units.nanometer
    assert units.base_units["time"] == units.picosecond


def test_all_lammps_presets_resolve_natively():
    assert set(UnitSystem.preset_names()) == {
        "real",
        "metal",
        "si",
        "cgs",
        "electron",
        "micro",
        "nano",
    }
    for name in UnitSystem.preset_names():
        system = UnitSystem.preset(name)
        assert not (REQUIRED_DIMS - set(system.base_units))
        assert all(isinstance(unit, molrs.Unit) for unit in system.base_units.values())


def test_preset_override_and_registration():
    assert (
        UnitSystem.preset("real", pressure="bar").base_units["pressure"]
        == UnitSystem().bar
    )
    UnitSystem.register_preset(
        "test_native_md", {"length": "nm", "time": "ps", "energy": "kJ/mol"}
    )
    assert UnitSystem.preset("test_native_md").base_units["length"] == UnitSystem().nm
    with pytest.raises(ValueError, match="already exists"):
        UnitSystem.register_preset("real", {"length": "angstrom"})


def test_unknown_preset_fails_fast():
    with pytest.raises(ValueError, match="unknown preset"):
        UnitSystem.preset("nonsense")


@pytest.fixture
def argon():
    source = UnitSystem()
    return UnitSystem.lj(
        mass=39.948 * source.amu,
        sigma=3.405 * source.angstrom,
        epsilon=0.2381 * source.kilocalorie_per_mole,
    )


def test_lj_scales_are_native_units(argon):
    assert argon.base_units["length"] == argon.lj_sigma
    assert argon.convert(3.405 * argon.angstrom, "lj_sigma").magnitude == pytest.approx(
        1.0
    )
    assert argon.convert(1.0 * argon.lj_tau, "ps").magnitude == pytest.approx(
        2.16, abs=0.01
    )
    assert argon.convert(
        1.0 * argon.lj_epsilon_over_kB, "K"
    ).magnitude == pytest.approx(119.8, abs=0.5)
    assert argon.convert(4.0 * argon.angstrom, "lj_sigma").magnitude == pytest.approx(
        1.1747, abs=1e-3
    )


def test_lj_rejects_wrong_dimensions_and_nonpositive_scales():
    units = UnitSystem()
    with pytest.raises(molrs.UnitsError, match="dimension mismatch"):
        UnitSystem.lj(
            mass=1.0 * units.second,
            sigma=1.0 * units.angstrom,
            epsilon=1.0 * units.eV,
        )
    with pytest.raises(molrs.UnitsError, match="finite and positive"):
        UnitSystem.lj(
            mass=1.0 * units.amu,
            sigma=0.0 * units.angstrom,
            epsilon=1.0 * units.eV,
        )
