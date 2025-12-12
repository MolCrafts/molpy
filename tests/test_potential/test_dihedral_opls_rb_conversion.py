"""Tests for RB to OPLS dihedral conversion."""

import pytest

from molpy.potential.dihedral.opls import format_lammps_dihedral_coeff, rb_to_opls


def test_rb_to_opls_exact_conversion():
    """Test that RB to OPLS conversion is exact (round-trip test)."""
    # Start with known OPLS coefficients F1-F4 (kJ/mol)
    F1, F2, F3, F4 = 1.0, 2.0, 3.0, 4.0

    # Convert to RB format using GROMACS Eqs. 200-201
    C0 = F2 + 0.5 * (F1 + F3)
    C1 = 0.5 * (-F1 + 3 * F3)
    C2 = -F2 + 4 * F4
    C3 = -2 * F3
    C4 = -4 * F4
    C5 = 0.0

    # Convert back to OPLS using rb_to_opls (returns kcal/mol)
    K1, K2, K3, K4 = rb_to_opls(C0, C1, C2, C3, C4, C5, units="kJ")

    # Verify round-trip (K values are in kcal/mol, F values in kJ/mol)
    # K = F / 4.184
    assert abs(K1 - F1 / 4.184) < 1e-10
    assert abs(K2 - F2 / 4.184) < 1e-10
    assert abs(K3 - F3 / 4.184) < 1e-10
    assert abs(K4 - F4 / 4.184) < 1e-10


def test_rb_to_opls_units_conversion():
    """Test unit conversion from kJ/mol to kcal/mol."""
    # Use valid RB coefficients (C0 + C1 + C2 + C3 + C4 = 0)
    # From: F1=4.184, F2=0, F3=0, F4=0 (kJ/mol)
    # C0 = F2 + 0.5*(F1+F3) = 2.092
    # C1 = 0.5*(-F1 + 3*F3) = -2.092
    # C2 = -F2 + 4*F4 = 0
    # C3 = -2*F3 = 0
    # C4 = -4*F4 = 0
    # C5 = 0
    C0, C1, C2, C3, C4, C5 = 2.092, -2.092, 0.0, 0.0, 0.0, 0.0

    # Convert in kJ/mol
    K1_kj, K2_kj, K3_kj, K4_kj = rb_to_opls(C0, C1, C2, C3, C4, C5, units="kJ")

    # Convert in kcal/mol (input already in kcal/mol, so same as kJ input / 4.184)
    # First convert RB coefficients to kcal/mol
    C0_kcal, C1_kcal, C2_kcal, C3_kcal, C4_kcal = (
        C0 / 4.184,
        C1 / 4.184,
        C2 / 4.184,
        C3 / 4.184,
        C4 / 4.184,
    )
    K1_kcal, K2_kcal, K3_kcal, K4_kcal = rb_to_opls(
        C0_kcal, C1_kcal, C2_kcal, C3_kcal, C4_kcal, C5, units="kcal"
    )

    # Verify unit conversion: K from kJ input should equal K from kcal input
    assert abs(K1_kj - K1_kcal) < 1e-10
    assert abs(K2_kj - K2_kcal) < 1e-10
    assert abs(K3_kj - K3_kcal) < 1e-10
    assert abs(K4_kj - K4_kcal) < 1e-10

    # Also verify that K values from kJ input are F values / 4.184
    # For our example: F1=4.184 kJ/mol, so K1 should be 1.0 kcal/mol
    assert abs(K1_kj - 1.0) < 1e-10


def test_rb_to_opls_valid_constraints():
    """Test that valid RB coefficients (C5=0, sum=0) produce no warnings."""
    import warnings

    # Valid RB coefficients
    C0, C1, C2, C3, C4, C5 = 18.96607, -18.96607, 0.0, 0.0, 0.0, 0.0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rb_to_opls(C0, C1, C2, C3, C4, C5, units="kJ")
        # Should have no warnings for valid coefficients
        assert len(w) == 0


def test_rb_to_opls_invalid_c5():
    """Test that invalid C5 raises ValueError (cannot represent cos⁵φ)."""
    # Invalid: C5 != 0 - cannot represent with 4-term OPLS
    C0, C1, C2, C3, C4, C5 = 0.0, 1.0, 2.0, 3.0, 4.0, 5.0

    with pytest.raises(ValueError, match="cos⁵φ"):
        rb_to_opls(C0, C1, C2, C3, C4, C5, units="kJ")


def test_rb_to_opls_invalid_sum():
    """Test that non-zero sum triggers warning (MD-safe constant offset)."""
    import warnings

    # Non-zero sum indicates constant energy offset (harmless for MD)
    C0, C1, C2, C3, C4, C5 = 1.0, 2.0, 3.0, 4.0, 5.0, 0.0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rb_to_opls(C0, C1, C2, C3, C4, C5, units="kJ")
        # Should have warning about constant offset
        assert len(w) > 0
        assert "constant energy offset" in str(w[0].message)


def test_format_lammps_dihedral_coeff():
    """Test formatting of LAMMPS dihedral_coeff line."""
    line = format_lammps_dihedral_coeff(
        "CT-CT-OH-HO",
        -0.4435,
        3.83255,
        0.72801,
        -4.11705,
        0.0,
        0.0,
        units="kJ",
        precision=6,
    )

    assert line.startswith("dihedral_coeff CT-CT-OH-HO")
    # Should contain 4 numbers
    parts = line.split()
    assert len(parts) == 6  # dihedral_coeff + type_name + 4 numbers
    assert all(
        part.replace("-", "").replace(".", "").isdigit()
        or part == "CT-CT-OH-HO"
        or part == "dihedral_coeff"
        for part in parts[2:]
    )


def test_rb_to_opls_zero_coefficients():
    """Test conversion with all zero coefficients."""
    K1, K2, K3, K4 = rb_to_opls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, units="kJ")

    assert K1 == 0.0
    assert K2 == 0.0
    assert K3 == 0.0
    assert K4 == 0.0


def test_rb_to_opls_known_example():
    """Test with known example from GROMACS manual."""
    # Example: RB coefficients that correspond to simple OPLS form
    # F1=4.184, F2=0, F3=0, F4=0 (in kJ/mol) = 1 kcal/mol
    # According to inverted formulas:
    # F1 = -2*C1 - 1.5*C3
    # If F2=F3=F4=0, then C2=C3=C4=0
    # F1 = -2*C1, so C1 = -F1/2 = -4.184/2 = -2.092
    # C0 = F2 + 0.5*(F1+F3) = 0 + 0.5*4.184 = 2.092

    C0, C1, C2, C3, C4, C5 = 2.092, -2.092, 0.0, 0.0, 0.0, 0.0

    K1, K2, K3, K4 = rb_to_opls(C0, C1, C2, C3, C4, C5, units="kJ")

    # Should get F1≈4.184 kJ/mol = 1.0 kcal/mol
    assert abs(K1 - 1.0) < 1e-6
    assert abs(K2) < 1e-10
    assert abs(K3) < 1e-10
    assert abs(K4) < 1e-10
