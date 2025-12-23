from typing import Literal

from pint import DimensionalityError, Quantity
from pint import Unit as PintUnit
from pint import UnitRegistry

UnitSystemName = Literal["real", "metal", "si", "cgs", "electron", "micro", "nano"]


class Unit:
    """
    A comprehensive wrapper around pint for LAMMPS-like unit systems.

    Initialize with a LAMMPS preset or custom base units and optionally override any derived units.
    Provides base units (mass, length, time, energy, temperature, charge, pressure,
    viscosity, electric_field, dipole, density) and derived units (velocity, force, torque),
    plus conversion and reduced-unit methods.

    Examples:
        # Using a preset system:
        real = Unit('real')
        # Override pressure unit:
        custom = Unit('metal', pressure=2.0 * Unit._registry.atmosphere)

        # Custom explicit system (without preset):
        ureg = Unit._registry  # Get the shared registry
        custom2 = Unit(
            mass=1.0 * ureg.kilogram,
            length=1.0 * ureg.meter,
            energy=1.0 * ureg.joule
        )

        # LJ units (special case):
        lj_ar = Unit.lj(
            mass=39.948 * ureg.atomic_mass_unit,
            length=3.405 * ureg.angstrom,
            energy=0.2381 * ureg.kilocalorie / ureg.mole
        )

        # Convert units:
        factor = real.conversion_factor('angstrom', 'nanometer')
        # Convert to reduced:
        eps_red = real.to_reduced(0.5 * ureg.kilocalorie / ureg.mole)
        # Convert from reduced:
        v = real.from_reduced(1.2, 'meter/second')
    """

    _registry = UnitRegistry()

    # Base and derived keys
    _base_keys = [
        "mass",
        "length",
        "time",
        "energy",
        "temperature",
        "charge",
        "pressure",
        "viscosity",
        "electric_field",
        "dipole",
        "density",
    ]

    # LAMMPS presets from https://docs.lammps.org/units.html
    # Note: 'lj' is handled separately via lj() classmethod
    _lammps_presets: dict[UnitSystemName, dict[str, Quantity]] = {
        "real": {
            "mass": 1.0 * _registry.gram,
            "length": 1.0 * _registry.angstrom,
            "time": 1.0 * _registry.femtosecond,
            "energy": 1.0 * _registry.kilocalorie / _registry.mole,
            "temperature": 1.0 * _registry.kelvin,
            "charge": 1.0 * _registry.elementary_charge,
            "pressure": 1.0 * _registry.atmosphere,
            "viscosity": 1.0 * _registry.poise,
            "electric_field": 1.0 * _registry.volt / _registry.angstrom,
            "dipole": (1.0 * _registry.elementary_charge) * _registry.angstrom,
            "density": 1.0 * _registry.gram / (_registry.centimeter**3),
        },
        "metal": {
            "mass": 1.0 * _registry.gram / _registry.mole,
            "length": 1.0 * _registry.angstrom,
            "time": 1.0 * _registry.picosecond,
            "energy": 1.0 * _registry.electron_volt,
            "temperature": 1.0 * _registry.kelvin,
            "charge": 1.0 * _registry.elementary_charge,
            "pressure": 1.0 * _registry.bar,
            "viscosity": 1.0 * _registry.poise,
            "electric_field": 1.0 * _registry.volt / _registry.angstrom,
            "dipole": (1.0 * _registry.elementary_charge) * _registry.angstrom,
            "density": 1.0 * _registry.gram / _registry.centimeter**3,
        },
        "si": {
            "mass": 1.0 * _registry.kilogram,
            "length": 1.0 * _registry.meter,
            "time": 1.0 * _registry.second,
            "energy": 1.0 * _registry.joule,
            "temperature": 1.0 * _registry.kelvin,
            "charge": 1.0 * _registry.coulomb,
            "pressure": 1.0 * _registry.pascal,
            "viscosity": 1.0 * _registry.pascal * _registry.second,
            "electric_field": 1.0 * _registry.volt / _registry.meter,
            "dipole": 1.0 * _registry.coulomb * _registry.meter,
            "density": 1.0 * _registry.kilogram / _registry.meter**3,
        },
        "cgs": {
            "mass": 1.0 * _registry.gram,
            "length": 1.0 * _registry.centimeter,
            "time": 1.0 * _registry.second,
            "energy": 1.0 * _registry.erg,
            "temperature": 1.0 * _registry.kelvin,
            "charge": 4.8032044e-10 * _registry.coulomb,  # statcoulomb
            "pressure": 1.0 * _registry.dyne / _registry.centimeter**2,
            "viscosity": 1.0 * _registry.poise,
            "electric_field": (1.0 * _registry.dyne / (_registry.coulomb)),
            "dipole": (4.8032044e-10 * _registry.coulomb) * _registry.centimeter,
            "density": 1.0 * _registry.gram / _registry.centimeter**3,
        },
        "electron": {
            "mass": _registry.electron_mass,
            "length": 1.0 * _registry.bohr,
            "time": 1.0 * _registry.femtosecond,
            "energy": 1.0 * _registry.hartree,
            "temperature": 1.0 * _registry.kelvin,
            "charge": 1.0 * _registry.elementary_charge,
            "pressure": 1.0 * _registry.pascal,
            "viscosity": 1.0 * _registry.pascal * _registry.second,
            "electric_field": 1.0 * _registry.volt / _registry.centimeter,
            "dipole": 1.0 * _registry.debye,
            "density": 1.0 * _registry.kilogram / _registry.meter**3,
        },
        "micro": {
            "mass": 1.0 * _registry.picogram,
            "length": 1.0 * _registry.micrometer,
            "time": 1.0 * _registry.microsecond,
            "energy": 1.0
            * _registry.picogram
            * _registry.micrometer**2
            / _registry.microsecond**2,
            "temperature": 1.0 * _registry.kelvin,
            "charge": 1.6021765e-7 * _registry.coulomb,
            "pressure": 1.0
            * _registry.picogram
            / (_registry.micrometer * _registry.microsecond**2),
            "viscosity": 1.0
            * _registry.picogram
            / (_registry.micrometer * _registry.microsecond),
            "electric_field": 1.0 * _registry.volt / _registry.micrometer,
            "dipole": 1.6021765e-7 * _registry.coulomb * _registry.micrometer,
            "density": 1.0 * _registry.picogram / _registry.micrometer**3,
        },
        "nano": {
            "mass": 1.0 * _registry.attogram,
            "length": 1.0 * _registry.nanometer,
            "time": 1.0 * _registry.nanosecond,
            "energy": 1.0
            * _registry.attogram
            * _registry.nanometer**2
            / _registry.nanosecond**2,
            "temperature": 1.0 * _registry.kelvin,
            "charge": 1.0 * _registry.elementary_charge,
            "pressure": 1.0
            * _registry.attogram
            / (_registry.nanometer * _registry.nanosecond**2),
            "viscosity": 1.0
            * _registry.attogram
            / (_registry.nanometer * _registry.nanosecond),
            "electric_field": 1.0 * _registry.volt / _registry.nanometer,
            "dipole": 1.0 * _registry.elementary_charge * _registry.nanometer,
            "density": 1.0 * _registry.attogram / _registry.nanometer**3,
        },
    }

    def __init__(
        self,
        system: UnitSystemName | None = None,
        **overrides: Quantity,
    ) -> None:
        """
        Initialize a Unit system.

        Args:
            system: Optional LAMMPS preset name. Use Unit.lj() for LJ units.
            overrides: quantities to override or define custom units.

        Note:
            For LJ units, use the lj() classmethod instead of passing 'lj' as system.
        """
        if system == "lj":
            raise ValueError(
                "LJ units require explicit mass, length, and energy scales. "
                "Use Unit.lj(mass=..., length=..., energy=...) instead."
            )

        self.ureg = Unit._registry
        self._preset_name: UnitSystemName | None = system  # Track original preset name
        units: dict[str, Quantity] = {}
        if system:
            units |= Unit._lammps_presets[system]
        units |= overrides
        # Validate base requirements
        if not all(k in units for k in ("mass", "length", "energy")):
            raise ValueError("Must specify at least mass, length, energy units")
        self._define_units(units)

    def _define_units(self, units: dict[str, Quantity]) -> None:
        """Define base and derived units in the registry."""
        self.base: dict[str, PintUnit] = {}
        self._original_preset: dict[str, Quantity] | None = None

        # Store original units for system matching (before registry pollution)
        if hasattr(self, "_preset_name"):
            self._original_preset = units.copy()

        # First define base quantities with unique names to avoid pollution
        import uuid

        self._unit_suffix = str(uuid.uuid4())[:8]

        for key in Unit._base_keys:
            qty = units.get(key)
            if qty is None:
                continue
            name = f"{key}_unit_{self._unit_suffix}"
            q0 = qty.to_base_units()
            self.ureg.define(f"{name} = {q0.magnitude} * {q0.units}")
            self.base[key] = getattr(self.ureg, name)

        # Define derived units only if base units are available
        if "length" in self.base and "time" in self.base:
            self.velocity_unit = self.base["length"] / self.base["time"]
        else:
            self.velocity_unit = None

        if "energy" in self.base and "length" in self.base:
            self.force_unit = self.base["energy"] / self.base["length"]
        else:
            self.force_unit = None

        if "energy" in self.base:
            self.torque_unit = self.base["energy"]
        else:
            self.torque_unit = None

    def conversion_factor(self, from_unit: str, to_unit: str) -> float:
        """Return factor to multiply a value in `from_unit` to get `to_unit`.

        Args:
            from_unit: Source unit string (e.g., 'angstrom')
            to_unit: Target unit string (e.g., 'nanometer')

        Returns:
            float: Conversion factor

        Raises:
            ValueError: If units have incompatible dimensions
        """
        try:
            return (1 * self.ureg.parse_units(from_unit)).to(to_unit).magnitude
        except DimensionalityError as e:
            raise e  # Re-raise the original DimensionalityError
        except Exception as e:
            raise ValueError(f"Cannot convert {from_unit} to {to_unit}") from e

    def to_reduced(self, qty: Quantity) -> float:
        """Convert a quantity to its dimensionless reduced value."""
        # Convert input to base units
        q_base = qty.to_base_units()

        # Find the appropriate base unit by matching dimensionality
        for _key, base_unit in self.base.items():
            base_qty = 1.0 * base_unit
            base_qty_base = base_qty.to_base_units()

            # Check if dimensionalities match
            if q_base.dimensionality == base_qty_base.dimensionality:
                # Return the ratio of magnitudes
                return q_base.magnitude / base_qty_base.magnitude

        # If no direct match found, try to construct from fundamental units
        # This handles complex derived quantities
        return self._to_reduced_complex(q_base)

    def _to_reduced_complex(self, q_base: Quantity) -> float:
        """Handle complex derived quantities by decomposing dimensionality."""
        # Build denominator from fundamental dimensions
        denom = self.ureg.dimensionless

        # Get fundamental dimensions (mass, length, time) and their exponents
        fundamental_dims = [
            "[mass]",
            "[length]",
            "[time]",
            "[temperature]",
            "[substance]",
            "[current]",
            "[luminosity]",
        ]

        for dim_str in fundamental_dims:
            exp = q_base.dimensionality.get(dim_str, 0)
            if exp == 0:
                continue

            # Find corresponding base unit
            base_unit = None
            if dim_str == "[mass]" and "mass" in self.base:
                base_unit = self.base["mass"]
            elif dim_str == "[length]" and "length" in self.base:
                base_unit = self.base["length"]
            elif dim_str == "[time]" and "time" in self.base:
                base_unit = self.base["time"]
            elif dim_str == "[temperature]" and "temperature" in self.base:
                base_unit = self.base["temperature"]
            # Add more mappings as needed

            if base_unit is not None:
                denom *= base_unit**exp

        try:
            result = (q_base / denom).to_base_units().magnitude
            return result
        except:
            # If all else fails, return 1.0 as a fallback
            return 1.0

    def from_reduced(self, val: float, to_unit: str) -> Quantity:
        """Convert a reduced value back to a physical quantity in `to_unit`."""
        # Parse the target unit to understand what we need to construct
        target_qty = self.ureg.Quantity(1, to_unit)
        target_base = target_qty.to_base_units()

        # Find the appropriate base unit with matching dimensionality
        for _key, base_unit in self.base.items():
            base_qty = 1.0 * base_unit
            base_qty_base = base_qty.to_base_units()

            # Check if dimensionalities match
            if target_base.dimensionality == base_qty_base.dimensionality:
                # Convert the reduced value to the base unit quantity
                physical_base = val * base_qty_base
                # Convert to the requested unit
                return physical_base.to(to_unit)

        # If no direct match, try complex construction
        raise ValueError(f"Cannot convert reduced units to {to_unit}")

    def get_system_name(self) -> str | None:
        """Return the LAMMPS system name if this matches a preset, None otherwise."""
        # If we started with a preset, check if it still matches exactly
        if self._preset_name and self._original_preset:
            original_preset = Unit._lammps_presets[self._preset_name]
            if self._compare_unit_dicts(self._original_preset, original_preset):
                return self._preset_name

        # Otherwise, check each preset for a match
        for name, preset in self._lammps_presets.items():
            if self._matches_preset(preset):
                return name
        return None

    def _matches_preset(self, preset: dict[str, Quantity]) -> bool:
        """Check if current unit system matches a preset."""
        # Quick check: if we stored original preset, compare with that
        if self._original_preset:
            return self._compare_unit_dicts(self._original_preset, preset)

        # Fallback: compare actual units (less reliable due to registry pollution)
        for key, expected_qty in preset.items():
            if key not in self.base:
                return False
            # This is less reliable but still works for simple comparisons
            expected_base = expected_qty.to_base_units()
            actual_qty = 1.0 * self.base[key]
            actual_base = actual_qty.to_base_units()

            # Check if dimensionalities match
            if expected_base.dimensionality != actual_base.dimensionality:
                return False

            # Check if magnitudes are close (within 1e-10 relative tolerance)
            if abs(expected_base.magnitude - actual_base.magnitude) > 1e-10 * abs(
                expected_base.magnitude
            ):
                return False
        return True

    def _compare_unit_dicts(
        self, dict1: dict[str, Quantity], dict2: dict[str, Quantity]
    ) -> bool:
        """Compare two unit dictionaries for equality."""
        if set(dict1.keys()) != set(dict2.keys()):
            return False

        for key in dict1:
            qty1 = dict1[key].to_base_units()
            qty2 = dict2[key].to_base_units()

            if qty1.dimensionality != qty2.dimensionality:
                return False

            if abs(qty1.magnitude - qty2.magnitude) > 1e-10 * abs(qty2.magnitude):
                return False
        return True

    @classmethod
    def list_available_systems(cls) -> list[str]:
        """Return list of available LAMMPS preset system names."""
        return list(cls._lammps_presets.keys())

    def get_base_units(self) -> dict[str, str]:
        """Return dictionary mapping unit types to their string representations."""
        return {key: str(unit) for key, unit in self.base.items()}

    def __repr__(self) -> str:
        """String representation of the Unit system."""
        system_name = self.get_system_name()
        if system_name:
            return f"Unit(system='{system_name}')"
        else:
            base_str = ", ".join(f"{k}={v}" for k, v in self.get_base_units().items())
            return f"Unit({base_str})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        system_name = self.get_system_name()
        if system_name:
            return f"LAMMPS {system_name} unit system"
        else:
            return f"Custom unit system with {len(self.base)} defined units"

    @classmethod
    def lj(
        cls,
        mass: Quantity,
        length: Quantity,
        energy: Quantity,
        **additional_units: Quantity,
    ) -> "Unit":
        """
        Create a Lennard-Jones (LJ) reduced unit system.

        In LJ units, the fundamental scales are set by the LJ parameters:
        - mass: typically the mass of one particle (e.g., argon atom mass)
        - length: LJ sigma parameter (particle size)
        - energy: LJ epsilon parameter (well depth)

        Time, temperature, and other derived units are computed from these.

        Args:
            mass: Mass scale (e.g., mass of one Ar atom)
            length: Length scale (e.g., LJ sigma parameter)
            energy: Energy scale (e.g., LJ epsilon parameter)
            additional_units: Optional additional unit overrides

        Returns:
            Unit: LJ unit system instance

        Examples:
            # Argon LJ parameters
            ureg = Unit._registry
            lj_ar = Unit.lj(
                mass=39.948 * ureg.atomic_mass_unit,
                length=3.405 * ureg.angstrom,
                energy=0.2381 * ureg.kilocalorie / ureg.mole
            )

            # Simple LJ system
            lj_simple = Unit.lj(
                mass=1.0 * ureg.atomic_mass_unit,
                length=1.0 * ureg.angstrom,
                energy=1.0 * ureg.kilocalorie / ureg.mole
            )
        """
        # Start with the three fundamental LJ scales
        units = {
            "mass": mass,
            "length": length,
            "energy": energy,
        }

        # Calculate derived units from LJ fundamentals
        # Time scale: sqrt(mass * sigma^2 / epsilon)
        ureg = cls._registry
        time_scale = ((mass * length**2 / energy) ** 0.5).to_base_units()
        units["time"] = time_scale

        # Temperature scale: epsilon / k_B
        k_b = ureg.boltzmann_constant
        temp_scale = (energy / k_b).to_base_units()
        units["temperature"] = temp_scale

        # Other derived scales (using standard LJ relationships)
        units["pressure"] = energy / length**3  # epsilon / sigma^3
        units["velocity"] = length / time_scale  # sigma / tau
        units["force"] = energy / length  # epsilon / sigma

        # Add any additional unit overrides
        units.update(additional_units)

        # Create instance bypassing normal __init__ validation
        instance = cls.__new__(cls)
        instance.ureg = ureg
        instance._define_units(units)
        return instance
