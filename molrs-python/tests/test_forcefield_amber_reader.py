"""Seam smoke test for the AMBER prmtop whole-system reader.

The 1-4 weighting (which pairs differ, their summed weights, what is refused)
is proved in Rust (``io::amber::prmtop_forcefield``, and term by term
against sander in ``prmtop_check``). This test only asserts that
``molrs.io.amber.AmberPrmtopForcefieldReader().read_system`` returns the force field and
the structure frame with the per-pair ``pairs`` block a mixed-SCEE topology
needs, and that the structure reader alone carries none.
"""

from pathlib import Path

import molrs
import numpy as np
import pytest
from molrs.io.amber import AmberPrmtopForcefieldReader

#: The Rust prmtop fixtures (AmberTools 26.1 tleap / ParmEd chamber builds).
FIXTURES = Path(__file__).resolve().parents[2] / "molrs/src/io/amber/testdata/prmtop"


def test_glycam_mixed_scee_keeps_its_per_pair_weights():
    # Most of this file's 1-4 rows are GLYCAM's (SCEE = SCNB = 1.0), so that
    # is the field's special_bonds; the ff14SB rows beside them (1.2 / 2.0)
    # are the `pairs` that carry their own weights.
    ff, frame = AmberPrmtopForcefieldReader().read_system(FIXTURES / "glycam.parm7")
    assert type(ff) is molrs.ff.forcefield.ForceField
    lj, coul = ff.special_bonds
    assert lj[2] == pytest.approx(1.0)
    assert coul[2] == pytest.approx(1.0)

    pairs = frame["pairs"]
    assert pairs.n_rows > 0
    assert np.asarray(pairs["is_14"]).all()
    np.testing.assert_allclose(np.asarray(pairs["coul_scale"]), 1.0 / 1.2)
    np.testing.assert_allclose(np.asarray(pairs["lj_scale"]), 1.0 / 2.0)

    # The structure is the structure reader's; it alone has no `pairs`.
    structure = molrs.io.read_prmtop(FIXTURES / "glycam.parm7")
    assert "pairs" not in structure
    assert frame["atoms"].n_rows == structure["atoms"].n_rows
    assert list(frame["bonds"]["atomi"]) == list(structure["bonds"]["atomi"])


def test_a_uniform_prmtop_has_no_pairs_block():
    ff, frame = AmberPrmtopForcefieldReader().read_system(FIXTURES / "ff14sb.parm7")
    assert type(ff) is molrs.ff.forcefield.ForceField
    assert "pairs" not in frame
    assert frame["atoms"].n_rows > 0


def test_an_unreadable_prmtop_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AmberPrmtopForcefieldReader().read_system(tmp_path / "missing.parm7")


def test_text_that_is_no_prmtop_raises():
    with pytest.raises(ValueError):
        AmberPrmtopForcefieldReader().read_system(molrs.io.Text("not a prmtop\n"))
