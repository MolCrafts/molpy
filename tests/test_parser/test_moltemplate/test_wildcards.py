"""Tests for the wildcard ``By Type`` rule resolver and ``replace{}`` parsing.

Covers:

* ``replace{ @atom:X @atom:Y }`` atom-type decoration.
* ``Data Bonds By Type`` wildcard matching fills in missing bond types.
* ``Data Angles / Dihedrals By Type`` propagate to auto-generated topology.
"""

from __future__ import annotations

from molpy.parser.moltemplate import build_system, parse_string


_SRC = """
SimpleFF {
  write_once("Data Masses") {
    @atom:X 12.011
    @atom:Y 1.008
  }
  write_once("Data Bonds By Type") {
    @bond:T_XX @atom:*_bX* @atom:*_bX*
    @bond:T_XY @atom:*_bX* @atom:*_bY*
  }
  write_once("Data Angles By Type") {
    @angle:T_YXY @atom:*_bY* @atom:*_bX* @atom:*_bY*
  }

  replace{ @atom:X @atom:X_bX_a*_d*_i* }
  replace{ @atom:Y @atom:Y_bY_a*_d*_i* }
}

Water inherits SimpleFF {
  write("Data Atoms") {
    $atom:ox $mol:m @atom:X 0.0  0.0 0.0 0.0
    $atom:h1 $mol:m @atom:Y 0.0  1.0 0.0 0.0
    $atom:h2 $mol:m @atom:Y 0.0 -1.0 0.0 0.0
  }
  write('Data Bond List') {
    $bond:ox_h1 $atom:ox $atom:h1
    $bond:ox_h2 $atom:ox $atom:h2
  }
}

m = new Water
"""


def _build():
    doc = parse_string(_SRC)
    return build_system(doc)


def test_replace_decorates_atom_types():
    system, _ = _build()
    atom_types = sorted({str(a.get("type", "")) for a in system.atoms})
    # Oxygen should become X_bX_a*_d*_i*; hydrogens become Y_bY_a*_d*_i*.
    assert any(t.startswith("X_bX") for t in atom_types), atom_types
    assert any(t.startswith("Y_bY") for t in atom_types), atom_types


def test_bonds_by_type_assigns_types():
    system, _ = _build()
    bond_types = sorted({str(b.get("type", "")) for b in system.bonds})
    assert "T_XY" in bond_types, bond_types


def test_angles_by_type_assigns_auto_generated_angles():
    system, _ = _build()
    # Auto-topology should generate the H-O-H angle (Y-X-Y) from two bonds.
    angles = list(system.angles)
    assert angles, "expected auto-generated angle"
    angle_types = {str(a.get("type", "")) for a in angles}
    assert "T_YXY" in angle_types, angle_types


def test_untyped_bond_list_is_resolved_via_rules():
    system, _ = _build()
    typed = [b for b in system.bonds if b.get("type")]
    assert len(typed) == len(list(system.bonds)), typed
