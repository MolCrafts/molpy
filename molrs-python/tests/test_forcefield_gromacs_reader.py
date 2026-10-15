"""Seam smoke test for the GROMACS topology force-field reader.

The directive model (what is read, what is refused, what a skip does) is proved
in Rust (``io::gromacs::top_reader``). This test only asserts that the
public ``molrs.io.read_top_forcefield`` maps a refused directive to
``ValueError`` and that its ``skip_directives`` keyword reaches the Rust reader.
"""

import molrs
import pytest

# One OPLS-AA atom type (GROMACS v2026.3 ``ffnonbonded.itp`` row) followed by a
# ``[ constrainttypes ]`` section, which the reader refuses unless skipped.
_TOP = """\
[ defaults ]
; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ
1  3  yes  0.5  0.5

[ atomtypes ]
; name  bond_type  at.num  mass  charge  ptype  sigma  epsilon
opls_135  CT  6  12.01100  -0.180  A  3.50000e-01  2.76144e-01

[ constrainttypes ]
; i  j  funct  b0
CT  HC  1  0.10900
"""


def test_constrainttypes_is_refused_without_skip_and_read_past_with_it(tmp_path):
    path = tmp_path / "ff.top"
    path.write_text(_TOP, encoding="utf-8")
    with pytest.raises(ValueError, match="constrainttypes"):
        molrs.io.read_top_forcefield(path)

    ff = molrs.io.read_top_forcefield(path, skip_directives=["constrainttypes"])
    assert type(ff) is molrs.ff.forcefield.ForceField


_SYSTEM = """\
[ defaults ]
1  3  yes  0.5  0.5

[ atomtypes ]
opls_135  CT  6  12.01100  -0.180  A  3.50000e-01  2.76144e-01
opls_140  HC  1   1.00800   0.060  A  2.50000e-01  1.25520e-01

[ bondtypes ]
CT  HC  1  0.10900  284512.0

[ moleculetype ]
CH  3

[ atoms ]
1  opls_135  1  CH  C  1
2  opls_140  1  CH  H  1

[ bonds ]
1  2  1

[ molecules ]
CH  2
"""


def test_read_top_system_returns_the_force_field_and_a_typed_frame(tmp_path):
    path = tmp_path / "topol.top"
    path.write_text(_SYSTEM, encoding="utf-8")
    ff, frame = molrs.io.read_top_system(path)
    assert type(ff) is molrs.ff.forcefield.ForceField
    assert list(frame["bonds"]["type"]) == ["CT-HC", "CT-HC"]
    assert list(frame["bonds"]["atomi"]) == [0, 2]
    assert list(frame["atoms"]["mol_id"]) == [1, 1, 2, 2]

    with pytest.raises(ValueError, match="read_system"):
        molrs.io.read_top_forcefield(path)


def test_write_top_system_round_trips_read_top_system(tmp_path):
    path = tmp_path / "topol.top"
    path.write_text(_SYSTEM, encoding="utf-8")
    ff, frame = molrs.io.read_top_system(path)
    out = tmp_path / "out.top"
    molrs.io.write_top_system(out, ff, frame)
    ff2, frame2 = molrs.io.read_top_system(out)
    # The writer states each bond's parameters on its row; the reader names
    # such a row's type after its lookup type.
    assert all(str(t).startswith("CT-HC") for t in frame2["bonds"]["type"])
    assert list(frame2["atoms"]["type"]) == list(frame["atoms"]["type"])
    assert list(frame2["bonds"]["atomi"]) == [0, 2]
    assert list(frame2["atoms"]["mol_id"]) == [1, 1, 2, 2]
