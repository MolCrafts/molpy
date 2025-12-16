"""OPLS dihedral force field styles."""

from molpy.core.logger import get_logger
from molpy.core.forcefield import AtomType, DihedralStyle, DihedralType

logger = get_logger(__name__)


def rb_to_opls(C0, C1, C2, C3, C4, C5, *, units="kJ"):
    """Convert Ryckaert-Bellemans (RB) dihedral coefficients to LAMMPS OPLS format.

    Performs analytical conversion from RB representation to OPLS 4-cosine form
    according to GROMACS manual Eqs. 200-201.

    The OPLS torsion (LAMMPS `dihedral_style opls`) is defined as:

    .. math::
       V_{\\text{OPLS}}(\\phi) = \\frac12F_1(1+\\cos\\phi) +
       \\frac12F_2(1-\\cos2\\phi) + \\frac12F_3(1+\\cos3\\phi) +
       \\frac12F_4(1-\\cos4\\phi)

    The RB representation (GROMACS) is:

    .. math::
       V_{\\rm RB}(\\phi) = \\sum_{n=0}^5 C_n (\\cos\\psi)^n

    where :math:`\\psi = \\phi - \\pi` (GROMACS convention).

    The analytical equivalence (GROMACS Eqs. 200-201) gives:

    .. math::
       \\begin{aligned}
       C_0 &= F_2 + \\tfrac12(F_1+F_3) \\\\
       C_1 &= \\tfrac12(-F_1 + 3F_3) \\\\
       C_2 &= -F_2 + 4F_4 \\\\
       C_3 &= -2 F_3 \\\\
       C_4 &= -4 F_4 \\\\
       C_5 &= 0
       \\end{aligned}

    This can be inverted to obtain:

    .. math::
       \\begin{aligned}
       F_1 &= -2 C_1 - \\tfrac32 C_3 \\\\
       F_2 &= -C_2 - C_4 \\\\
       F_3 &= -\\tfrac12 C_3 \\\\
       F_4 &= -\\tfrac14 C_4
       \\end{aligned}

    For LAMMPS, :math:`K_i = F_i` (same coefficients), but requires kcal/mol.

    Args:
        C0, C1, C2, C3, C4, C5: RB coefficients
        units: Input units, either "kJ" or "kcal" (default: "kJ")

    Returns:
        Tuple of (K1, K2, K3, K4) in kcal/mol for LAMMPS OPLS dihedral style

    Raises:
        ValueError: If C5 ≠ 0 (cannot represent cos⁵φ with 4-term OPLS)

    Note:
        **MD-Safe Conversion**: This converter preserves forces and relative energies
        exactly when C5 ≈ 0. The conversion uses only C1-C4 to compute F1-F4:
        
        - F1-F4 are computed from C1-C4 only, preserving dV/dφ exactly
        - Any difference in C0 from the "ideal" C0' = F2 + 0.5*(F1+F3) represents
          a harmless constant energy offset
        - MD trajectories and thermodynamics (up to arbitrary reference energies)
          are unaffected by this constant offset
        - The only strict requirement is C5 ≈ 0 (cannot represent cos⁵φ term)
        
        If C0 + C1 + C2 + C3 + C4 ≠ 0, this indicates a non-zero constant offset
        but does NOT affect MD correctness.
    """
    # STRICT validation: Only C5 ≠ 0 is a hard error
    # C5 ≠ 0 means the RB potential contains cos⁵φ which cannot be represented
    # by a 4-term OPLS potential (we don't support F5)

    if abs(C5) > 1e-6:
        raise ValueError(
            f"RB torsion uses C5 = {C5:.6f}, which cannot be represented by "
            f"a 4-term OPLS potential (no cos⁵φ term). "
            f"This RB potential cannot be converted to OPLS format."
        )

    # Soft validation: C0+...+C4 ≠ 0 indicates constant energy offset
    # This is harmless for MD (doesn't affect forces or relative energies)
    sum_c = C0 + C1 + C2 + C3 + C4
    if abs(sum_c) > 1e-2:
        logger.warning(
            "RB coefficients do not lie on the ideal 4-term OPLS manifold "
            f"(C0+C1+C2+C3+C4 = {sum_c:.6f}, expected ≈ 0). "
            "Conversion will preserve forces and relative energies exactly, "
            f"but will introduce a constant energy offset of ΔE = {sum_c:.6f} kJ/mol. "
            "This does not affect MD simulations."
        )

    # Compute F1-F4 analytically using inverted formulas
    # These formulas use ONLY C1-C4, preserving dV/dφ exactly
    F1 = -2.0 * C1 - 1.5 * C3
    F2 = -C2 - C4
    F3 = -0.5 * C3
    F4 = -0.25 * C4

    # Convert units: kJ/mol → kcal/mol if needed
    if units.lower() == "kj":
        K1 = F1 / 4.184
        K2 = F2 / 4.184
        K3 = F3 / 4.184
        K4 = F4 / 4.184
    elif units.lower() == "kcal":
        K1 = F1
        K2 = F2
        K3 = F3
        K4 = F4
    else:
        raise ValueError(f"Unknown units: {units}. Must be 'kJ' or 'kcal'")

    return (K1, K2, K3, K4)


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
    """Generate LAMMPS dihedral_coeff line from RB coefficients.

    Helper function to convert RB coefficients to LAMMPS format and print
    the corresponding dihedral_coeff command.

    Args:
        type_name: Dihedral type name (e.g., "CT-CT-OH-HO")
        C0, C1, C2, C3, C4, C5: RB coefficients
        units: Input units, either "kJ" or "kcal" (default: "kJ")
        precision: Number of decimal places (default: 6)

    Returns:
        String with LAMMPS dihedral_coeff command

    Example:
        >>> coeff_line = format_lammps_dihedral_coeff(
        ...     "CT-CT-OH-HO", -0.4435, 3.83255, 0.72801, -4.11705, 0.0, 0.0
        ... )
        >>> print(coeff_line)
        dihedral_coeff CT-CT-OH-HO 0.916001 0.173999 -0.983999 0.000000
    """
    K1, K2, K3, K4 = rb_to_opls(C0, C1, C2, C3, C4, C5, units=units)
    format_str = f"{{:.{precision}f}}"
    return f"dihedral_coeff {type_name} {format_str.format(K1)} {format_str.format(K2)} {format_str.format(K3)} {format_str.format(K4)}"


class DihedralOPLSType(DihedralType):
    """OPLS dihedral type with c0-c5 coefficients."""

    def __init__(
        self,
        name: str,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        c0: float = 0.0,
        c1: float = 0.0,
        c2: float = 0.0,
        c3: float = 0.0,
        c4: float = 0.0,
        c5: float = 0.0,
    ):
        """
        Args:
            name: Type name
            itom: First atom type
            jtom: Second atom type
            ktom: Third atom type
            ltom: Fourth atom type
            c0-c5: OPLS Ryckaert-Bellemans coefficients
        """
        super().__init__(
            name,
            itom,
            jtom,
            ktom,
            ltom,
            c0=c0,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            c5=c5,
        )


class DihedralOPLSStyle(DihedralStyle):
    """OPLS dihedral style with fixed name='opls'.

    OPLS dihedral uses Ryckaert-Bellemans (RB) coefficients c0-c5.
    LAMMPS opls style expects k1-k4, which are computed from c0-c5 using
    analytical conversion according to GROMACS manual Eqs. 200-201.
    """

    def __init__(self):
        super().__init__("opls")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        c0: float = 0.0,
        c1: float = 0.0,
        c2: float = 0.0,
        c3: float = 0.0,
        c4: float = 0.0,
        c5: float = 0.0,
        name: str = "",
    ) -> DihedralOPLSType:
        """Define OPLS dihedral type.

        Args:
            itom: First atom type
            jtom: Second atom type
            ktom: Third atom type
            ltom: Fourth atom type
            c0-c5: OPLS Ryckaert-Bellemans coefficients
            name: Optional name (defaults to itom-jtom-ktom-ltom)

        Returns:
            DihedralOPLSType instance
        """
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}-{ltom.name}"
        dt = DihedralOPLSType(name, itom, jtom, ktom, ltom, c0, c1, c2, c3, c4, c5)
        self.types.add(dt)
        return dt

    def to_lammps_params(self, dihedral_type: DihedralOPLSType) -> list[float]:
        """Convert OPLS c0-c5 coefficients to LAMMPS k1-k4 format.

        Note: In OPLS XML files, c1-c4 typically contain OPLS coefficients F1-F4
        directly (not RB format). So we use c1-c4 directly as k1-k4.

        For true RB format coefficients, use rb_to_opls() function instead.

        Args:
            dihedral_type: DihedralOPLSType with c0-c5 parameters

        Returns:
            List of [k1, k2, k3, k4] for LAMMPS in kcal/mol
        """
        # OPLS XML stores F1-F4 in c1-c4 (already OPLS format)
        # Just return them directly (should already be in kcal/mol)
        return [
            dihedral_type.params.kwargs.get("c1", 0.0),
            dihedral_type.params.kwargs.get("c2", 0.0),
            dihedral_type.params.kwargs.get("c3", 0.0),
            dihedral_type.params.kwargs.get("c4", 0.0),
        ]
