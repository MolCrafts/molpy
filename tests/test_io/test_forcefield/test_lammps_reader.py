"""molpy's LAMMPS force-field reader is a thin sink to the native molrs reader.

``molpy.io.read_lammps_forcefield`` and ``molpy.io.forcefield.read_lammps_forcefield``
delegate to ``molrs.read_lammps_forcefield`` — the force-field model lives in
molrs (Rust); molpy does not reimplement parsing. These tests assert the
delegation returns a ``molrs.ForceField`` with correctly unit-normalized params.
"""

import molpy.io as mpio
import molpy.io.forcefield as mpff

_FF = """\
pair_style lj/cut/coul/long 10.0 10.0
pair_coeff c3 c3 0.107800 3.397710

bond_style harmonic
bond_coeff c3-c3 228.890000 1.535400

angle_style harmonic
angle_coeff c3-c3-oh 76.790000 109.660000

dihedral_style fourier
dihedral_coeff c3-c3-oh-ho 1 0.060000 3 0.000000
"""


def _write(tmp_path):
    p = tmp_path / "melt.ff"
    p.write_text(_FF)
    return p


def test_io_read_lammps_forcefield_sinks_to_molrs(tmp_path):
    ff = mpio.read_lammps_forcefield(_write(tmp_path))
    # Sunk to molrs: the returned object is the native molrs ForceField.
    assert type(ff).__module__ == "molrs.forcefield"
    bt = ff.get_style("bond", "harmonic").get_type_by_name("c3-c3")
    assert abs(bt.params["k"] - 457.78) < 1e-2  # k = 2K


def test_forcefield_namespace_read_lammps_forcefield(tmp_path):
    ff = mpff.read_lammps_forcefield(_write(tmp_path))
    assert type(ff).__module__ == "molrs.forcefield"
    assert len(ff.get_style("dihedral", "fourier").types) == 1


def test_list_of_includes_concatenates(tmp_path):
    a = tmp_path / "styles.ff"
    a.write_text("pair_style lj/cut/coul/long 10.0 10.0\npair_coeff c3 c3 0.1 3.4\n")
    b = tmp_path / "bonds.ff"
    b.write_text("bond_style harmonic\nbond_coeff c3-c3 100.0 1.5\n")
    ff = mpio.read_lammps_forcefield([a, b])
    assert ff.get_style("pair", "lj/cut") is not None
    assert ff.get_style("bond", "harmonic").get_type_by_name("c3-c3") is not None
