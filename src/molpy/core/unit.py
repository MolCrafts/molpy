"""Python unit-system sugar over molrs' native unit engine."""

from __future__ import annotations

from typing import Mapping, Self

import molrs

__all__ = ["UnitSystem"]


_LAMMPS_PRESETS: dict[str, dict[str, str]] = {
    "real": {
        "mass": "gram_per_mole",
        "length": "angstrom",
        "time": "femtosecond",
        "energy": "kilocalorie_per_mole",
        "temperature": "kelvin",
        "charge": "elementary_charge",
        "pressure": "atmosphere",
        "velocity": "angstrom / femtosecond",
        "force": "kilocalorie_per_mole / angstrom",
        "density": "gram / centimeter ** 3",
    },
    "metal": {
        "mass": "gram_per_mole",
        "length": "angstrom",
        "time": "picosecond",
        "energy": "electron_volt",
        "temperature": "kelvin",
        "charge": "elementary_charge",
        "pressure": "bar",
        "velocity": "angstrom / picosecond",
        "force": "electron_volt / angstrom",
        "density": "gram / centimeter ** 3",
    },
    "si": {
        "mass": "kilogram",
        "length": "meter",
        "time": "second",
        "energy": "joule",
        "temperature": "kelvin",
        "charge": "coulomb",
        "pressure": "pascal",
        "velocity": "meter / second",
        "force": "newton",
        "density": "kilogram / meter ** 3",
    },
    "cgs": {
        "mass": "gram",
        "length": "centimeter",
        "time": "second",
        "energy": "erg",
        "temperature": "kelvin",
        "charge": "statcoulomb",
        "pressure": "dyne / centimeter ** 2",
        "velocity": "centimeter / second",
        "force": "dyne",
        "density": "gram / centimeter ** 3",
    },
    "electron": {
        "mass": "amu",
        "length": "bohr",
        "time": "femtosecond",
        "energy": "hartree",
        "temperature": "kelvin",
        "charge": "elementary_charge",
        "pressure": "pascal",
        "velocity": "bohr / femtosecond",
        "force": "hartree / bohr",
    },
    "micro": {
        "mass": "picogram",
        "length": "micrometer",
        "time": "microsecond",
        "energy": "picogram * micrometer ** 2 / microsecond ** 2",
        "temperature": "kelvin",
        "charge": "picocoulomb",
        "pressure": "picogram / (micrometer * microsecond ** 2)",
        "velocity": "micrometer / microsecond",
        "force": "picogram * micrometer / microsecond ** 2",
        "density": "picogram / micrometer ** 3",
    },
    "nano": {
        "mass": "attogram",
        "length": "nanometer",
        "time": "nanosecond",
        "energy": "attogram * nanometer ** 2 / nanosecond ** 2",
        "temperature": "kelvin",
        "charge": "elementary_charge",
        "pressure": "attogram / (nanometer * nanosecond ** 2)",
        "velocity": "nanometer / nanosecond",
        "force": "attogram * nanometer / nanosecond ** 2",
        "density": "attogram / nanometer ** 3",
    },
}


class UnitSystem(molrs.UnitRegistry):
    """Native unit registry with LAMMPS presets and LJ construction sugar.

    Parsing, definitions, dimensional arithmetic, and conversion all execute in
    molrs. ``base_units`` only records the user's chosen working units.
    """

    def __new__(cls, *, base_units: Mapping[str, str] | None = None) -> "UnitSystem":
        del base_units
        return super().__new__(cls)

    def __init__(self, *, base_units: Mapping[str, str] | None = None) -> None:
        super().__init__()
        self.base_units = {
            dimension: self.parse(expression)
            for dimension, expression in (base_units or {}).items()
        }

    @classmethod
    def preset(cls, name: str, **overrides: str) -> Self:
        """Create a unit system from a LAMMPS unit-style preset."""
        try:
            preset = _LAMMPS_PRESETS[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown preset {name!r}; available: {sorted(_LAMMPS_PRESETS)}"
            ) from exc
        return cls(base_units={**preset, **overrides})

    @classmethod
    def preset_names(cls) -> tuple[str, ...]:
        """Return registered preset names."""
        return tuple(_LAMMPS_PRESETS)

    @classmethod
    def register_preset(
        cls,
        name: str,
        base_units: dict[str, str],
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a custom base-unit mapping."""
        if not isinstance(base_units, dict) or not base_units:
            raise TypeError("base_units must be a non-empty dict[str, str]")
        if name in _LAMMPS_PRESETS and not overwrite:
            raise ValueError(
                f"preset {name!r} already exists; pass overwrite=True to replace it"
            )
        _LAMMPS_PRESETS[name] = dict(base_units)

    @classmethod
    def lj(
        cls,
        *,
        mass: molrs.Quantity,
        sigma: molrs.Quantity,
        epsilon: molrs.Quantity,
    ) -> Self:
        """Create a native Lennard-Jones reduced unit system."""
        system = cls()
        system.define_lj_units(mass, sigma, epsilon)
        system.base_units = {
            "length": system.lj_sigma,
            "energy": system.lj_epsilon,
            "time": system.lj_tau,
            "temperature": system.lj_epsilon_over_kB,
        }
        return system

    def convert(self, quantity: molrs.Quantity, target: str | molrs.Unit):
        """Convert with this registry, including registry-local LJ units."""
        return quantity.to(self.parse(target) if isinstance(target, str) else target)
