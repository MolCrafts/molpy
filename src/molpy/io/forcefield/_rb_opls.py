"""Ryckaert-Bellemans (RB) → OPLS dihedral coefficient conversion.

Pure-Python format helper used by the XML/LAMMPS force-field I/O. This is an
I/O-format concern (translating GROMACS RB coefficients to LAMMPS ``opls``
``k1-k4``), not a force-field kernel — the kernels themselves live in molrs.
"""

from __future__ import annotations

from molpy.core.logger import get_logger

logger = get_logger(__name__)

_KJ_PER_KCAL = 4.184


def rb_to_opls(C0, C1, C2, C3, C4, C5, *, units="kJ"):
    """Convert Ryckaert-Bellemans (RB) dihedral coefficients to OPLS k1-k4.

    Performs the analytical RB → OPLS 4-cosine conversion (GROMACS manual
    Eqs. 200-201). Only C1-C4 enter the result, so forces and relative
    energies are preserved exactly when ``C5 == 0``; a non-zero
    ``C0+C1+C2+C3+C4`` is a harmless constant energy offset.

    Args:
        C0, C1, C2, C3, C4, C5: RB coefficients.
        units: Input units, either ``"kJ"`` or ``"kcal"`` (default ``"kJ"``).

    Returns:
        Tuple ``(K1, K2, K3, K4)`` in kcal/mol for the LAMMPS ``opls`` style.

    Raises:
        ValueError: If ``C5 != 0`` (no cos⁵φ term in 4-term OPLS) or the units
            string is not recognised.
    """
    if abs(C5) > 1e-6:
        raise ValueError(
            f"RB torsion uses C5 = {C5:.6f}, which cannot be represented by "
            f"a 4-term OPLS potential (no cos⁵φ term). "
            f"This RB potential cannot be converted to OPLS format."
        )

    sum_c = C0 + C1 + C2 + C3 + C4
    if abs(sum_c) > 1e-2:
        logger.warning(
            "RB coefficients do not lie on the ideal 4-term OPLS manifold "
            f"(C0+C1+C2+C3+C4 = {sum_c:.6f}, expected ≈ 0). "
            "Conversion will preserve forces and relative energies exactly, "
            f"but will introduce a constant energy offset of ΔE = {sum_c:.6f} kJ/mol. "
            "This does not affect MD simulations."
        )

    F1 = -2.0 * C1 - 1.5 * C3
    F2 = -C2 - C4
    F3 = -0.5 * C3
    F4 = -0.25 * C4

    unit = units.lower()
    if unit == "kj":
        return (
            F1 / _KJ_PER_KCAL,
            F2 / _KJ_PER_KCAL,
            F3 / _KJ_PER_KCAL,
            F4 / _KJ_PER_KCAL,
        )
    if unit == "kcal":
        return (F1, F2, F3, F4)
    raise ValueError(f"Unknown units: {units}. Must be 'kJ' or 'kcal'")


def format_lammps_dihedral_coeff(
    type_name: str,
    C0: float,
    C1: float,
    C2: float,
    C3: float,
    C4: float,
    C5: float,
    *,
    units: str = "kJ",
    precision: int = 6,
) -> str:
    """Build a LAMMPS ``dihedral_coeff`` line from RB coefficients.

    Args:
        type_name: Dihedral type name (e.g. ``"CT-CT-OH-HO"``).
        C0, C1, C2, C3, C4, C5: RB coefficients.
        units: Input units, either ``"kJ"`` or ``"kcal"`` (default ``"kJ"``).
        precision: Number of decimal places (default 6).

    Returns:
        A ``dihedral_coeff <name> K1 K2 K3 K4`` command string.
    """
    K1, K2, K3, K4 = rb_to_opls(C0, C1, C2, C3, C4, C5, units=units)
    fmt = f"{{:.{precision}f}}"
    return (
        f"dihedral_coeff {type_name} "
        f"{fmt.format(K1)} {fmt.format(K2)} {fmt.format(K3)} {fmt.format(K4)}"
    )
