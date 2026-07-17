"""Force-field model — thin re-export of the native molrs hierarchy.

molrs (the Rust extension) natively owns the entire force-field model:
``ForceField``, the ``Style`` tree, the ``Type`` tree and ``Parameters``.
molpy no longer maintains a parallel Python hierarchy; this module simply
re-exports the molrs classes and adds the handful of thin specialized
``Style`` subclasses that molrs does not ship a named class for (e.g.
``morse``, ``class2``, ``fourier``, ``periodic`` variants).

A specialized style here carries no kernel — it only fixes the style name so
callers can write ``ff.def_style(BondMorseStyle())`` instead of
``ff.def_bondstyle("morse")``. Energy/force evaluation lives entirely in
molrs via ``ff.to_potentials().calc_energy(frame)`` / ``.calc_forces(frame)``.
"""

from __future__ import annotations

from molrs import (
    AngleHarmonicStyle,
    AngleStyle,
    AngleType,
    AtomStyle,
    AtomType,
    BondHarmonicStyle,
    BondStyle,
    BondType,
    DihedralOPLSStyle,
    DihedralStyle,
    DihedralType,
    ForceField,
    ImproperStyle,
    ImproperType,
    PairCoulLongStyle,
    PairLJ126CoulCutStyle,
    PairLJ126CoulLongStyle,
    PairLJ126Style,
    PairStyle,
    PairType,
    Parameters,
    Style,
    Type,
)

# ``AtomisticForcefield`` was a molpy-only subclass; the native molrs
# ``ForceField`` now covers its responsibilities.
AtomisticForcefield = ForceField


# ---------------------------------------------------------------------------
# Thin specialized styles for kernels molrs does not ship a named class for.
# Each only pins the style name; types and params flow through molrs natively.
# ---------------------------------------------------------------------------


class BondMorseStyle(BondStyle):
    """Bond ``morse`` style (LAMMPS ``bond_style morse``)."""

    def _name_default(self) -> str:
        return "morse"


class BondClass2Style(BondStyle):
    """Bond ``class2`` style."""

    def _name_default(self) -> str:
        return "class2"


class AngleClass2Style(AngleStyle):
    """Angle ``class2`` style."""

    def _name_default(self) -> str:
        return "class2"


class AngleClass2BondBondStyle(AngleStyle):
    """Angle ``class2/bb`` cross term."""

    def _name_default(self) -> str:
        return "class2/bb"


class AngleClass2BondAngleStyle(AngleStyle):
    """Angle ``class2/ba`` cross term."""

    def _name_default(self) -> str:
        return "class2/ba"


class DihedralPeriodicStyle(DihedralStyle):
    """Dihedral ``periodic`` (CHARMM-style ``charmm``/``periodic``) style."""

    def _name_default(self) -> str:
        return "periodic"


class DihedralFourierStyle(DihedralStyle):
    """Dihedral ``fourier`` style (AMBER multi-term)."""

    def _name_default(self) -> str:
        return "fourier"


class DihedralCharmmStyle(DihedralStyle):
    """Dihedral ``charmm`` style."""

    def _name_default(self) -> str:
        return "charmm"


class DihedralMultiHarmonicStyle(DihedralStyle):
    """Dihedral ``multi/harmonic`` style."""

    def _name_default(self) -> str:
        return "multi/harmonic"


class DihedralClass2Style(DihedralStyle):
    """Dihedral ``class2`` style."""

    def _name_default(self) -> str:
        return "class2"


class ImproperPeriodicStyle(ImproperStyle):
    """Improper ``periodic`` style."""

    def _name_default(self) -> str:
        return "periodic"


class ImproperHarmonicStyle(ImproperStyle):
    """Improper ``harmonic`` style."""

    def _name_default(self) -> str:
        return "harmonic"


class ImproperCvffStyle(ImproperStyle):
    """Improper ``cvff`` style."""

    def _name_default(self) -> str:
        return "cvff"


class ImproperClass2Style(ImproperStyle):
    """Improper ``class2`` style."""

    def _name_default(self) -> str:
        return "class2"


class PairBuckStyle(PairStyle):
    """Pair ``buck`` (Buckingham) style."""

    def _name_default(self) -> str:
        return "buck"


class PairMorseStyle(PairStyle):
    """Pair ``morse`` style."""

    def _name_default(self) -> str:
        return "morse"


class PairLJClass2Style(PairStyle):
    """Pair ``lj/class2`` style."""

    def _name_default(self) -> str:
        return "lj/class2"


class PairTholeStyle(PairStyle):
    """Pair ``thole`` — Thole damped dipole-dipole / core-shell Coulomb style.

    Per-atom-type parameters: ``charge`` (e), ``alpha`` (Å³),
    ``a_thole`` (dimensionless Thole damping width, default 2.6).
    Damping: ``T(r) = 1 − (1 + s·r/2)·exp(−s·r)`` where
    ``s = a_ij / (α_i·α_j)^(1/6)``, ``a_ij = (a_i + a_j)/2``.

    Reference: Thole, *Chem. Phys.* **59** (1981) 341,
    :doi:`10.1016/0301-0104(81)85176-2`.
    """

    def _name_default(self) -> str:
        return "thole"


class PairCoulTTStyle(PairStyle):
    """Pair ``coul/tt`` — Tang−Toennies damped charge−dipole Coulomb style.

    Style-level parameters: ``b`` (1/Å, default 4.5), ``n`` (default 4),
    ``c`` (default 1.0).
    Per-atom-type parameter: ``charge`` (e).
    Damping: ``f_n(r) = 1 − c·exp(−b·r)·Σ_{k=0}^n (b·r)^k/k!``.

    Reference: Tang & Toennies, *J. Chem. Phys.* **80** (1984) 3726,
    :doi:`10.1063/1.447150`.
    """

    def _name_default(self) -> str:
        return "coul/tt"


__all__ = [
    # Core molrs hierarchy
    "ForceField",
    "AtomisticForcefield",
    "Parameters",
    "Style",
    "AtomStyle",
    "BondStyle",
    "AngleStyle",
    "DihedralStyle",
    "ImproperStyle",
    "PairStyle",
    "Type",
    "AtomType",
    "BondType",
    "AngleType",
    "DihedralType",
    "ImproperType",
    "PairType",
    # Named specialized styles (from molrs)
    "BondHarmonicStyle",
    "AngleHarmonicStyle",
    "DihedralOPLSStyle",
    "PairLJ126Style",
    "PairLJ126CoulCutStyle",
    "PairLJ126CoulLongStyle",
    "PairCoulLongStyle",
    # Thin specialized styles (molpy-defined, gap fillers)
    "BondMorseStyle",
    "BondClass2Style",
    "AngleClass2Style",
    "AngleClass2BondBondStyle",
    "AngleClass2BondAngleStyle",
    "DihedralPeriodicStyle",
    "DihedralFourierStyle",
    "DihedralCharmmStyle",
    "DihedralMultiHarmonicStyle",
    "DihedralClass2Style",
    "ImproperPeriodicStyle",
    "ImproperHarmonicStyle",
    "ImproperCvffStyle",
    "ImproperClass2Style",
    "PairBuckStyle",
    "PairMorseStyle",
    "PairLJClass2Style",
    "PairTholeStyle",
    "PairCoulTTStyle",
]
