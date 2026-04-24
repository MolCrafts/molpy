"""Unit support for MolPy.

:class:`UnitSystem` is a :class:`pint.UnitRegistry` subclass. Its
constructor accepts every argument pint's does. On top of pint it:

* Pre-registers three per-molecule energy units
  (``kilocalorie_per_mole``, ``kilojoule_per_mole``, ``gram_per_mole``)
  and four LJ reduced-unit symbols (``lj_sigma``, ``lj_epsilon``,
  ``lj_tau``, ``lj_epsilon_over_kB``). Names are prefixed so they do
  not shadow pint's ``sigma`` (Stefan–Boltzmann alias) or ``tau``
  (:math:`2\\pi`).
* Exposes a :attr:`~UnitSystem.base_units` mapping ``{dimension: Unit}``
  that records the user's chosen base unit per physical dimension. This
  is how downstream code (force fields, I/O, compute operators) learns
  "which length unit does this system use?" — without caring whether the
  system came from a preset.
* Provides :meth:`~UnitSystem.preset` (LAMMPS unit-style presets) and
  :meth:`~UnitSystem.lj` (Lennard-Jones reduced units) as optional
  factories.

Example:
    >>> from molpy import UnitSystem
    >>> real = UnitSystem.preset("real")
    >>> 1.5 * real.angstrom
    <Quantity(1.5, 'angstrom')>
    >>> real.base_units["length"]
    <Unit('angstrom')>
    >>> ar = UnitSystem.lj(
    ...     mass=39.948 * real.amu,
    ...     sigma=3.405 * real.angstrom,
    ...     epsilon=0.2381 * real.kilocalorie_per_mole,
    ... )
    >>> with ar.context("lj"):
    ...     (4.0 * ar.angstrom).to(ar.lj_sigma).magnitude  # doctest: +ELLIPSIS
    1.174743...
"""

from __future__ import annotations

from typing import Mapping

import pint

__all__ = ["UnitSystem"]


# --------------------------------------------------------------------------- #
# LAMMPS unit-style presets
# --------------------------------------------------------------------------- #
#
# Reference: https://docs.lammps.org/units.html
#
# These are *one* source of base-unit selections, not the definition of
# UnitSystem. Downstream code must never branch on preset names — always
# read from system.base_units.

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


# --------------------------------------------------------------------------- #
# UnitSystem
# --------------------------------------------------------------------------- #


class UnitSystem(pint.UnitRegistry):
    """A :class:`pint.UnitRegistry` with MolPy-specific pre-registered
    units and an optional ``base_units`` selection per dimension.

    The constructor accepts every argument :class:`pint.UnitRegistry`
    accepts, plus one MolPy-specific keyword:

    Args:
        *args: Forwarded to :class:`pint.UnitRegistry`.
        base_units: Optional ``{dimension: unit}`` mapping recording this
            system's chosen base unit for each physical dimension.
            Downstream code reads this to learn which units a system
            uses, independent of how the system was created.
        **kwargs: Forwarded to :class:`pint.UnitRegistry`.

    Example:
        >>> u = UnitSystem(base_units={"length": "nm", "time": "ps"})
        >>> u.base_units["length"]
        <Unit('nanometer')>
        >>> 2.5 * u.nanometer
        <Quantity(2.5, 'nanometer')>
    """

    def __init__(
        self,
        *args,
        base_units: Mapping[str, str] | None = None,
        **kwargs,
    ) -> None:
        # Captured here so that _after_init (run post default-unit load) can
        # resolve unit labels against the fully-populated registry.
        self._pending_base_units = base_units
        super().__init__(*args, **kwargs)

    def _after_init(self) -> None:
        # pint loads default_en.txt here; our custom units must be defined
        # *after* that, and base_units must be resolved *after* that again.
        super()._after_init()
        self._install_molpy_units()
        self.base_units: dict[str, pint.Unit] = (
            {dim: self.Unit(expr) for dim, expr in self._pending_base_units.items()}
            if self._pending_base_units
            else {}
        )
        del self._pending_base_units

    def _install_molpy_units(self) -> None:
        """Register MolPy-specific additions on top of pint's defaults."""
        # Per-molecule energy/mass. pint's kcal/mol carries a [substance]
        # dimension; the per-molecule form is what we want for MD values.
        self.define("kilocalorie_per_mole = kilocalorie / avogadro_number")
        self.define("kilojoule_per_mole   = kilojoule   / avogadro_number")
        self.define("gram_per_mole        = gram        / avogadro_number")

        # LJ reduced-unit symbols. Each lives in its own private dimension
        # so any conversion across the boundary requires an active context.
        # Prefixed with 'lj_' to avoid shadowing pint's 'sigma'
        # (Stefan–Boltzmann alias) and 'tau' (= 2π).
        self.define("lj_sigma           = [length_lj]")
        self.define("lj_epsilon         = [energy_lj]")
        self.define("lj_tau             = [time_lj]")
        self.define("lj_epsilon_over_kB = [temperature_lj]")

    # --- Factories --------------------------------------------------------

    @classmethod
    def preset(cls, name: str, **overrides: str) -> "UnitSystem":
        """Return a :class:`UnitSystem` initialised from a built-in preset.

        Currently MolPy ships the seven LAMMPS unit styles (``real``,
        ``metal``, ``si``, ``cgs``, ``electron``, ``micro``, ``nano``).
        Preset names and contents are an implementation detail — callers
        consuming the resulting :class:`UnitSystem` must only rely on
        :attr:`base_units`.

        Any keyword overrides replace individual dimension entries::

            >>> UnitSystem.preset("real", pressure="bar").base_units["pressure"]
            <Unit('bar')>
        """
        try:
            preset = _LAMMPS_PRESETS[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown preset {name!r}; available: {sorted(_LAMMPS_PRESETS)}"
            ) from exc
        base = {**preset, **overrides}
        return cls(base_units=base)

    @classmethod
    def preset_names(cls) -> tuple[str, ...]:
        """Names of all built-in presets."""
        return tuple(_LAMMPS_PRESETS)

    @classmethod
    def lj(
        cls,
        *,
        mass: pint.Quantity,
        sigma: pint.Quantity,
        epsilon: pint.Quantity,
    ) -> "UnitSystem":
        """Return a reduced Lennard-Jones unit system.

        The three scales ``mass``, ``sigma`` (length), and ``epsilon``
        (per-molecule energy) fix the system; :math:`\\tau = \\sqrt{m\\sigma^2/\\varepsilon}`
        and :math:`\\varepsilon/k_B` are derived. The returned registry
        has a pint context named ``"lj"`` registered; activate it with
        ``with system.context("lj"): ...`` to convert across the
        physical/reduced boundary.

        ``mass``/``sigma``/``epsilon`` may be Quantities from any pint
        registry — they are translated to SI magnitudes before being
        rebuilt in the new registry.
        """
        mass_kg = mass.to("kg").magnitude
        sigma_m = sigma.to("m").magnitude
        eps_J = epsilon.to("J").magnitude

        system = cls(
            base_units={
                "length": "lj_sigma",
                "energy": "lj_epsilon",
                "time": "lj_tau",
                "temperature": "lj_epsilon_over_kB",
            }
        )

        m = mass_kg * system.kilogram
        s = sigma_m * system.meter
        e = eps_J * system.joule
        tau = ((m * s**2 / e) ** 0.5).to_base_units()

        system.add_context(_build_lj_context(system, sigma=s, epsilon=e, tau=tau))
        return system


# --------------------------------------------------------------------------- #
# LJ context builder
# --------------------------------------------------------------------------- #


def _build_lj_context(
    reg: pint.UnitRegistry,
    *,
    sigma: pint.Quantity,
    epsilon: pint.Quantity,
    tau: pint.Quantity,
) -> pint.Context:
    kB = reg.boltzmann_constant
    ctx = pint.Context("lj")

    ctx.add_transformation(
        "[length]",
        "lj_sigma",
        lambda _reg, x, _s=sigma: (x / _s).to_reduced_units().magnitude * _reg.lj_sigma,
    )
    ctx.add_transformation(
        "lj_sigma",
        "[length]",
        lambda _reg, x, _s=sigma: x.magnitude * _s,
    )

    ctx.add_transformation(
        "[energy]",
        "lj_epsilon",
        lambda _reg, x, _e=epsilon: (x / _e).to_reduced_units().magnitude
        * _reg.lj_epsilon,
    )
    ctx.add_transformation(
        "lj_epsilon",
        "[energy]",
        lambda _reg, x, _e=epsilon: x.magnitude * _e,
    )

    ctx.add_transformation(
        "[time]",
        "lj_tau",
        lambda _reg, x, _t=tau: (x / _t).to_reduced_units().magnitude * _reg.lj_tau,
    )
    ctx.add_transformation(
        "lj_tau",
        "[time]",
        lambda _reg, x, _t=tau: x.magnitude * _t,
    )

    # T* = T * kB / epsilon
    ctx.add_transformation(
        "[temperature]",
        "lj_epsilon_over_kB",
        lambda _reg, x, _e=epsilon, _k=kB: (
            (x * _k / _e).to_reduced_units().magnitude * _reg.lj_epsilon_over_kB
        ),
    )
    ctx.add_transformation(
        "lj_epsilon_over_kB",
        "[temperature]",
        lambda _reg, x, _e=epsilon, _k=kB: (x.magnitude * _e / _k).to("kelvin"),
    )

    return ctx
