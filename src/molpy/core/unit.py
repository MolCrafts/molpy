"""Python unit-system sugar over the native unit engine."""

from __future__ import annotations

from typing import Mapping, Self

import molrs

__all__ = ["UnitSystem"]


# The LAMMPS unit styles are the native ``UnitPreset``'s; molpy reads their base
# units from it and owns only the presets the native core does not define.
_NATIVE_PRESETS = ("real", "metal", "si", "cgs", "electron", "micro", "nano")
_DIMENSIONS = (
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
_EXTRA_PRESETS: dict[str, dict[str, str]] = {
    "openmm": {
        "mass": "gram_per_mole",
        "length": "nanometer",
        "time": "picosecond",
        "energy": "kilojoule_per_mole",
        "temperature": "kelvin",
        "charge": "elementary_charge",
        "pressure": "bar",
        "velocity": "nanometer / picosecond",
        "force": "kilojoule_per_mole / nanometer",
        "density": "gram / centimeter ** 3",
    },
}


class UnitSystem(molrs.UnitRegistry):
    """Native unit registry with LAMMPS presets and LJ construction sugar.

    Parsing, definitions, dimensional arithmetic, and conversion all execute in
    the native core. ``base_units`` only records the user's chosen working units.
    """

    def __new__(cls, *, base_units: Mapping[str, str] | None = None) -> "UnitSystem":
        del base_units
        return super().__new__(cls)

    def __init__(self, *, base_units: Mapping[str, str] | None = None) -> None:
        super().__init__()
        # CODATA 2018; SI dimension order is L, M, T, I, Θ, N, J.
        self.define(
            "boltzmann_constant",
            1.380649e-23,
            [2, 1, -2, 0, -1, 0, 0],
            aliases=["k_B"],
            symbol="k_B",
        )
        self.base_units = {
            dimension: self.parse(expression)
            for dimension, expression in (base_units or {}).items()
        }

    @classmethod
    def preset(cls, name: str, **overrides: str) -> Self:
        """Create a unit system from a LAMMPS unit-style preset."""
        if name in _EXTRA_PRESETS:
            preset = _EXTRA_PRESETS[name]
        elif name in _NATIVE_PRESETS:
            native = molrs.UnitPreset(name)
            preset = {dim: getattr(native, dim)() for dim in _DIMENSIONS}
        else:
            raise ValueError(
                f"unknown preset {name!r}; available: {sorted(cls.preset_names())}"
            )
        return cls(base_units={**preset, **overrides})

    @classmethod
    def preset_names(cls) -> tuple[str, ...]:
        """Return registered preset names."""
        return (*_NATIVE_PRESETS, *_EXTRA_PRESETS)

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
        if name in cls.preset_names() and not overwrite:
            raise ValueError(
                f"preset {name!r} already exists; pass overwrite=True to replace it"
            )
        _EXTRA_PRESETS[name] = dict(base_units)

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

    def factor(self, source: str | molrs.Unit, target: str | molrs.Unit) -> float:
        """Magnitude of one ``source`` unit expressed in ``target`` units.

        Args:
            source: Unit string or parsed unit (e.g. ``"kilocalorie_per_mole"``).
            target: Dimensionally compatible destination unit.

        Returns:
            Conversion factor so ``value_target = value_source * factor``.
        """
        src = self.parse(source) if isinstance(source, str) else source
        dst = self.parse(target) if isinstance(target, str) else target
        return float((1.0 * src).to(dst).magnitude)
