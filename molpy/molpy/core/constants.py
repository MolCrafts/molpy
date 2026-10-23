"""Physical and engine constants — ``molrs::core::constants``, in full.

CODATA 2018 / SI-2019 values (``AVOGADRO``, ``BOLTZMANN``,
``SPEED_OF_LIGHT``, ``BOHR_RADIUS``, …) and the constants engines and force
fields define (``COULOMB_REAL``, ``AMBER_COULOMB``, ``OPENMM_ONE_4PI_EPS0``,
``AMBER_SCEE`` / ``AMBER_SCNB``, ``OPLS_LJ_14``, ``MMFF_COULOMB``, …). Every
name and value is the Rust constant of the same name, read from Rust's
``constants::ALL`` table, so a constant added in Rust appears here with no
edit.

Unit conversions are not constants: use the unit registry,
``molpy.core.UnitRegistry().factor("kcal", "kJ")`` or
``UnitRegistry().quantity(x, "nm").to(...)``.
"""

from molpy import _core

__all__ = sorted(name for name in dir(_core.constants) if name.isupper())

globals().update({name: getattr(_core.constants, name) for name in __all__})
