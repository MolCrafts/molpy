import pint
import pytest

import molpy
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


class TestSubclassesUnitRegistry:
    def test_is_a_pint_registry(self):
        u = UnitSystem()
        assert isinstance(u, pint.UnitRegistry)

    def test_plain_ctor_works_like_pint(self):
        u = UnitSystem()
        q = 1.5 * u.angstrom
        assert q.magnitude == pytest.approx(1.5)
        assert str(q.units) == "angstrom"

    def test_pint_kwargs_forwarded(self):
        u = UnitSystem(case_sensitive=False)
        # UnitRegistry keeps the kwarg; no explosion is the test
        assert (1.0 * u.Angstrom).magnitude == 1.0


class TestMolPyCustomUnits:
    def test_kilocalorie_per_mole_registered(self):
        u = UnitSystem()
        q = 1.0 * u.kilocalorie_per_mole
        assert q.to(u.electron_volt).magnitude == pytest.approx(0.04336, rel=1e-3)

    def test_kilojoule_per_mole_registered(self):
        u = UnitSystem()
        q = 1.0 * u.kilocalorie_per_mole
        assert q.to(u.kilojoule_per_mole).magnitude == pytest.approx(4.184, rel=1e-4)

    def test_lj_dimensions_registered(self):
        u = UnitSystem()
        assert (1.0 * u.lj_sigma).dimensionality == pint.util.UnitsContainer(
            {"[length_lj]": 1}
        )
        assert (1.0 * u.lj_epsilon).dimensionality == pint.util.UnitsContainer(
            {"[energy_lj]": 1}
        )

    def test_pint_sigma_alias_not_shadowed(self):
        """Our LJ symbols are prefixed so pint's own aliases keep working.

        Stock pint registers ``sigma`` as an alias for
        ``stefan_boltzmann_constant``. With an ``lj_*`` prefix we no
        longer shadow it, so string-based parsing in downstream code
        (FF files, LAMMPS input) keeps its textbook meaning.
        """
        u = UnitSystem()
        assert u.parse_expression("sigma").to_base_units().magnitude == pytest.approx(
            5.670374419e-8, rel=1e-6
        )


class TestBaseUnits:
    def test_empty_by_default(self):
        u = UnitSystem()
        assert u.base_units == {}

    def test_populated_by_constructor(self):
        u = UnitSystem(base_units={"length": "nm", "time": "ps"})
        assert u.base_units["length"] == u.nanometer
        assert u.base_units["time"] == u.picosecond

    def test_is_a_dict_of_pint_units(self):
        u = UnitSystem(base_units={"length": "nm"})
        assert isinstance(u.base_units["length"], pint.Unit)


class TestPresetFactory:
    def test_preset_names_includes_all_lammps_styles(self):
        assert set(UnitSystem.preset_names()) == {
            "real",
            "metal",
            "si",
            "cgs",
            "electron",
            "micro",
            "nano",
        }

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="unknown preset"):
            UnitSystem.preset("nonsense")

    def test_preset_populates_base_units(self):
        real = UnitSystem.preset("real")
        assert real.base_units["length"] == real.angstrom
        assert real.base_units["energy"] == real.kilocalorie_per_mole
        assert real.base_units["time"] == real.femtosecond

    def test_metal_preset_values(self):
        metal = UnitSystem.preset("metal")
        assert metal.base_units["length"] == metal.angstrom
        assert metal.base_units["energy"] == metal.electron_volt
        assert metal.base_units["time"] == metal.picosecond
        assert metal.base_units["pressure"] == metal.bar

    def test_si_preset_values(self):
        si = UnitSystem.preset("si")
        assert si.base_units["length"] == si.meter
        assert si.base_units["energy"] == si.joule
        assert si.base_units["time"] == si.second

    @pytest.mark.parametrize("name", UnitSystem.preset_names())
    def test_all_presets_have_required_dimensions(self, name):
        system = UnitSystem.preset(name)
        missing = REQUIRED_DIMS - set(system.base_units)
        assert not missing, f"{name!r} missing dimensions: {missing}"

    @pytest.mark.parametrize("name", UnitSystem.preset_names())
    def test_every_dimension_resolves_to_pint_unit(self, name):
        system = UnitSystem.preset(name)
        for dim, unit in system.base_units.items():
            assert isinstance(unit, pint.Unit), f"{name}.base_units[{dim!r}] → {unit!r}"

    def test_override_replaces_single_dimension(self):
        real = UnitSystem.preset("real", pressure="bar")
        assert real.base_units["pressure"] == real.bar
        # Unchanged dimensions persist
        assert real.base_units["length"] == real.angstrom


class TestLJFactory:
    """Argon: ε/kB ≈ 119.8 K, τ ≈ 2.16 ps (Allen & Tildesley, §2.6)."""

    @pytest.fixture
    def argon(self):
        u = UnitSystem()
        return UnitSystem.lj(
            mass=39.948 * u.amu,
            sigma=3.405 * u.angstrom,
            epsilon=0.2381 * u.kilocalorie_per_mole,
        )

    def test_base_units_cover_lj_dimensions(self, argon):
        assert argon.base_units["length"] == argon.lj_sigma
        assert argon.base_units["energy"] == argon.lj_epsilon
        assert argon.base_units["time"] == argon.lj_tau
        assert argon.base_units["temperature"] == argon.lj_epsilon_over_kB

    def test_context_named_lj_is_registered(self, argon):
        with argon.context("lj"):
            (1.0 * argon.angstrom).to("lj_sigma")  # must not raise

    def test_tau_matches_textbook(self, argon):
        with argon.context("lj"):
            tau_ps = (1.0 * argon.lj_tau).to("ps").magnitude
        assert tau_ps == pytest.approx(2.16, abs=0.01)

    def test_temperature_base_matches_textbook(self, argon):
        with argon.context("lj"):
            T_K = (1.0 * argon.lj_epsilon_over_kB).to("kelvin").magnitude
        assert T_K == pytest.approx(119.8, abs=0.5)

    def test_physical_length_reduces(self, argon):
        with argon.context("lj"):
            r_star = (4.0 * argon.angstrom).to("lj_sigma").magnitude
        assert r_star == pytest.approx(1.1747, abs=1e-3)

    def test_length_round_trip(self, argon):
        with argon.context("lj"):
            r_star = (3.405 * argon.angstrom).to("lj_sigma").magnitude
        assert r_star == pytest.approx(1.0, abs=1e-10)

    def test_temperature_reduction(self, argon):
        with argon.context("lj"):
            T_star = (300 * argon.kelvin).to("lj_epsilon_over_kB").magnitude
        assert T_star == pytest.approx(300 / 119.8, rel=1e-3)

    def test_energy_round_trip(self, argon):
        with argon.context("lj"):
            eps = 0.2381 * argon.kilocalorie_per_mole
            e_star = eps.to("lj_epsilon").magnitude
            back = (e_star * argon.lj_epsilon).to("kilocalorie_per_mole").magnitude
        assert e_star == pytest.approx(1.0, abs=1e-10)
        assert back == pytest.approx(0.2381, abs=1e-10)

    def test_accepts_quantities_from_other_registries(self):
        """Scales may come from any pint registry; SI conversion decouples them."""
        source = UnitSystem()
        ar = UnitSystem.lj(
            mass=39.948 * source.amu,
            sigma=3.405 * source.angstrom,
            epsilon=0.2381 * source.kilocalorie_per_mole,
        )
        with ar.context("lj"):
            T_K = (1.0 * ar.lj_epsilon_over_kB).to("kelvin").magnitude
        assert T_K == pytest.approx(119.8, abs=0.5)

    def test_two_independent_lj_systems(self):
        u = UnitSystem()
        ar = UnitSystem.lj(
            mass=39.948 * u.amu,
            sigma=3.405 * u.angstrom,
            epsilon=0.2381 * u.kilocalorie_per_mole,
        )
        ne = UnitSystem.lj(
            mass=20.18 * u.amu,
            sigma=2.75 * u.angstrom,
            epsilon=0.0801 * u.kilocalorie_per_mole,
        )
        with ar.context("lj"):
            ar_T = (1.0 * ar.lj_epsilon_over_kB).to("kelvin").magnitude
        with ne.context("lj"):
            ne_T = (1.0 * ne.lj_epsilon_over_kB).to("kelvin").magnitude
        assert ar_T == pytest.approx(119.8, abs=0.5)
        assert ne_T == pytest.approx(40.3, abs=0.5)


class TestPresetAgnosticConsumers:
    """Downstream code must read base_units — never branch on preset name."""

    def test_custom_and_preset_expose_same_api(self):
        real = UnitSystem.preset("real")
        custom = UnitSystem(base_units={"length": "angstrom", "energy": "eV"})

        # A generic consumer that only looks at base_units:
        def length_unit(system):
            return system.base_units["length"]

        assert length_unit(real) == real.angstrom
        assert length_unit(custom) == custom.angstrom

    def test_lj_system_uses_same_consumer_path(self):
        u = UnitSystem()
        ar = UnitSystem.lj(
            mass=39.948 * u.amu,
            sigma=3.405 * u.angstrom,
            epsilon=0.2381 * u.kilocalorie_per_mole,
        )

        def length_unit(system):
            return system.base_units["length"]

        assert length_unit(ar) == ar.lj_sigma


class TestPublicExports:
    def test_only_unitsystem_is_toplevel(self):
        assert molpy.UnitSystem is UnitSystem

    def test_no_pint_internals_leaked(self):
        for leaked in ("ureg", "lj_context", "LAMMPS_SYSTEMS"):
            assert not hasattr(molpy, leaked), f"molpy.{leaked} leaked"

    def test_no_preset_names_leaked(self):
        for leaked in ("real", "metal", "si", "cgs", "electron", "micro", "nano"):
            assert not hasattr(molpy, leaked), f"molpy.{leaked} leaked"
