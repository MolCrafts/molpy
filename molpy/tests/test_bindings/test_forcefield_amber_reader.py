"""Seam smoke test for the AMBER prmtop whole-system reader.

The 1-4 weighting (which pairs differ, their summed weights, what is refused)
is proved in Rust (``io::amber::prmtop_forcefield``). This test only asserts that
``mp.io.amber.AmberPrmtopForcefieldReader().read_system`` returns the force field and
the structure frame with its special pairs (``pairs``), the per-pair scales a
mixed-SCEE topology needs among them, and that the structure reader alone
carries none.
"""

from pathlib import Path

import molpy as mp
import numpy as np
import pytest
from molpy.io.amber import AmberPrmtopForcefieldReader

#: The Rust prmtop fixtures (AmberTools 26.1 tleap / ParmEd chamber builds).
FIXTURES = (
    Path(__file__).resolve().parents[3] / "crates/molrs/src/io/amber/testdata/prmtop"
)


def _stated(pairs: mp.core.Block, key: str) -> np.ndarray:
    """Which rows state ``key``."""
    if key not in pairs:
        return np.zeros(pairs.n_rows, dtype=bool)
    mask = pairs.validity(key)
    return np.ones(pairs.n_rows, dtype=bool) if mask is None else np.asarray(mask)


def test_glycam_mixed_scee_keeps_its_per_pair_weights():
    # Most of this file's 1-4 rows are GLYCAM's (SCEE = SCNB = 1.0), so that
    # is the field's special_bonds; the ff14SB rows beside them (1.2 / 2.0)
    # carry their own scales.
    ff, frame = AmberPrmtopForcefieldReader().read_system(FIXTURES / "glycam.parm7")
    assert type(ff) is mp.ff.forcefield.ForceField
    lj, coul = ff.special_bonds
    assert lj[2] == pytest.approx(1.0)
    assert coul[2] == pytest.approx(1.0)

    pairs = frame["pairs"]
    one_four = np.asarray(pairs["n_bonds"]) == 3
    own = one_four & _stated(pairs, "coul_scale")
    assert own.any()
    np.testing.assert_allclose(np.asarray(pairs["coul_scale"])[own], 1.0 / 1.2)
    np.testing.assert_allclose(np.asarray(pairs["lj_scale"])[own], 1.0 / 2.0)

    # The structure is the structure reader's; it alone has no special pairs.
    structure = mp.io.read_prmtop(FIXTURES / "glycam.parm7")
    assert "pairs" not in structure
    assert frame["atoms"].n_rows == structure["atoms"].n_rows
    assert list(frame["bonds"]["atomi"]) == list(structure["bonds"]["atomi"])


def test_a_uniform_prmtop_states_no_pair_scale():
    ff, frame = AmberPrmtopForcefieldReader().read_system(FIXTURES / "ff14sb.parm7")
    assert type(ff) is mp.ff.forcefield.ForceField
    pairs = frame["pairs"]
    assert pairs.n_rows > 0
    for key in ("coul_scale", "lj_scale"):
        assert not _stated(pairs, key).any(), key


def test_an_unreadable_prmtop_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AmberPrmtopForcefieldReader().read_system(tmp_path / "missing.parm7")


def test_text_that_is_no_prmtop_raises():
    with pytest.raises(ValueError):
        AmberPrmtopForcefieldReader().read_system(mp.io.Text("not a prmtop\n"))
