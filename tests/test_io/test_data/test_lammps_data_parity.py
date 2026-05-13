"""Parametrized parity tests between molpy.io and molrs.io.

Each scenario corresponds to one file in ``tests-data/lammps-data/`` —
the filename is the test point. The same test body runs against both
``molpy.io.read_lammps_data`` and ``molrs.io.read_lammps_data`` to prove
the molrs reader is a drop-in replacement under duck typing: same
attribute names (``frame.box.lengths``, ``frame.box.tilts``,
``block[key]``), same row counts, same coordinates.

Per-reader differences (canonical vs format-specific column naming, the
``charge``/``mol_id`` translation) are bridged by small accessor helpers
so the assertion bodies stay reader-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import molpy.io as molpy_io
import molpy.io.experimental as _experimental_io
import molrs.io as molrs_io


# ──────────────────────────────────────────────────────────────────────────
# Reader parametrization
# ──────────────────────────────────────────────────────────────────────────


def _molpy_read(path: Path, atom_style: str):
    return molpy_io.read_lammps_data(path, atom_style=atom_style)


def _molrs_read(path: Path, atom_style: str):
    return molrs_io.read_lammps_data(path, atom_style=atom_style)


def _experimental_read(path: Path, atom_style: str):
    return _experimental_io.read_lammps_data(path, atom_style=atom_style)


READERS = [
    pytest.param(_molpy_read, id="molpy"),
    pytest.param(_molrs_read, id="molrs"),
    pytest.param(_experimental_read, id="experimental"),
]


# ──────────────────────────────────────────────────────────────────────────
# Test-data fixture (re-declared locally so this file is self-contained)
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def test_files(TEST_DATA_DIR) -> dict[str, Path]:
    d = TEST_DATA_DIR / "lammps-data"
    return {
        "molid": d / "molid.lmp",
        "whitespaces": d / "whitespaces.lmp",
        "triclinic_1": d / "triclinic-1.lmp",
        "triclinic_2": d / "triclinic-2.lmp",
        "labelmap": d / "labelmap.lmp",
        "solvated": d / "solvated.lmp",
        "data_body": d / "data.body",
    }


# ──────────────────────────────────────────────────────────────────────────
# Duck-typing helpers
#
# The two readers produce different concrete column names for the same
# physical quantity (molpy canonicalizes to ``mol_id``/``charge``; molrs
# keeps the LAMMPS-shaped ``molecule_id``/``charge``). These helpers pick
# whichever is present so the assertion bodies stay reader-agnostic.
# ──────────────────────────────────────────────────────────────────────────


def _col(block, *candidates: str) -> np.ndarray:
    for name in candidates:
        if name in block:
            return np.asarray(block[name])
    raise AssertionError(
        f"none of {candidates!r} present in block; have {list(block.keys())!r}"
    )


def _mol_id(block) -> np.ndarray:
    return _col(block, "mol_id", "molecule_id")


def _charge(block) -> np.ndarray:
    return _col(block, "charge", "q")


# ──────────────────────────────────────────────────────────────────────────
# One test per file. Filename == test point.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("read", READERS)
def test_molid_file(read, test_files):
    """molid.lmp — atom_style full with mol IDs, integer box bounds 0-20."""
    frame = read(test_files["molid"], "full")
    atoms = frame["atoms"]

    assert atoms.nrows == 12
    np.testing.assert_array_almost_equal(
        np.asarray(frame.box.lengths), [20.0, 20.0, 20.0]
    )
    np.testing.assert_array_almost_equal(np.asarray(frame.box.origin), [0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(np.asarray(frame.box.tilts), [0.0, 0.0, 0.0])

    # Canonical/format-specific mol-id column must be present.
    mol_ids = _mol_id(atoms)
    assert len(mol_ids) == 12
    assert set(np.unique(mol_ids).tolist()) <= {0, 1, 2, 3}


@pytest.mark.parametrize("read", READERS)
def test_whitespaces_file(read, test_files):
    """whitespaces.lmp — extra whitespace tolerance + float box bounds."""
    frame = read(test_files["whitespaces"], "full")
    atoms = frame["atoms"]

    assert atoms.nrows == 1
    np.testing.assert_array_almost_equal(
        np.asarray(frame.box.lengths), [10.0, 10.0, 10.0]
    )
    np.testing.assert_array_almost_equal(
        [
            float(np.asarray(atoms["x"])[0]),
            float(np.asarray(atoms["y"])[0]),
            float(np.asarray(atoms["z"])[0]),
        ],
        [5.0, 5.0, 5.0],
    )


@pytest.mark.parametrize("read", READERS)
def test_triclinic_1_file(read, test_files):
    """triclinic-1.lmp — triclinic header with all-zero tilts."""
    frame = read(test_files["triclinic_1"], "atomic")

    assert frame.box is not None
    np.testing.assert_array_almost_equal(
        np.asarray(frame.box.lengths), [34.0, 34.0, 34.0]
    )
    np.testing.assert_array_almost_equal(np.asarray(frame.box.tilts), [0.0, 0.0, 0.0])


@pytest.mark.parametrize("read", READERS)
def test_triclinic_2_file(read, test_files):
    """triclinic-2.lmp — non-zero tilt factors (5, -8, 3 = xy, xz, yz)."""
    frame = read(test_files["triclinic_2"], "atomic")

    assert frame.box is not None
    np.testing.assert_array_almost_equal(np.asarray(frame.box.tilts), [5.0, -8.0, 3.0])
    # Edge-vector norms reflect the tilt: |a|=lx, |b|=sqrt(xy^2+ly^2),
    # |c|=sqrt(xz^2+yz^2+lz^2).
    np.testing.assert_array_almost_equal(
        np.asarray(frame.box.lengths),
        [
            34.0,
            np.sqrt(5.0**2 + 34.0**2),
            np.sqrt(8.0**2 + 3.0**2 + 34.0**2),
        ],
    )


@pytest.mark.parametrize("read", READERS)
def test_labelmap_file(read, test_files):
    """labelmap.lmp — atom/bond/angle/dihedral type labels + connectivity."""
    frame = read(test_files["labelmap"], "full")

    assert frame["atoms"].nrows == 16
    assert frame["bonds"].nrows == 14
    assert frame["angles"].nrows == 25
    assert frame["dihedrals"].nrows == 27


@pytest.mark.parametrize("read", READERS)
def test_solvated_file(read, test_files):
    """solvated.lmp — large system with bonds/angles/dihedrals/impropers."""
    frame = read(test_files["solvated"], "full")

    assert frame["atoms"].nrows == 7772
    assert frame["bonds"].nrows == 6248
    assert frame["angles"].nrows == 8100
    assert frame["dihedrals"].nrows == 10720
    assert frame["impropers"].nrows == 1376

    np.testing.assert_array_almost_equal(
        np.asarray(frame.box.lengths),
        [
            33.920998 - (-0.103),
            33.957998 - (-0.066),
            162.150494 - (-0.885501),
        ],
    )
    np.testing.assert_array_almost_equal(
        np.asarray(frame.box.origin), [-0.103, -0.066, -0.885501]
    )

    # Mol IDs and charges must be present (under either column name).
    assert len(_mol_id(frame["atoms"])) == 7772
    assert len(_charge(frame["atoms"])) == 7772


@pytest.mark.parametrize("read", READERS)
def test_data_body_file(read, test_files):
    """data.body — atom_style='body' must expose per-atom bodyflag + mass."""
    frame = read(test_files["data_body"], "body")
    atoms = frame["atoms"]

    assert atoms.nrows == 100
    np.testing.assert_array_almost_equal(
        np.asarray(frame.box.lengths),
        [
            15.532224567 - (-15.532224567),
            15.532224567 - (-15.532224567),
            0.5 - (-0.5),
        ],
    )

    for col in ("id", "type", "bodyflag", "mass", "x", "y", "z"):
        assert col in atoms, f"body atom_style must expose {col!r}"

    # First atom: 1 1 1 6 -15.5322 -15.5322 0 ...
    assert int(np.asarray(atoms["bodyflag"])[0]) == 1
    assert float(np.asarray(atoms["mass"])[0]) == 6.0
    np.testing.assert_array_almost_equal(
        [
            float(np.asarray(atoms["x"])[0]),
            float(np.asarray(atoms["y"])[0]),
            float(np.asarray(atoms["z"])[0]),
        ],
        [-15.5322, -15.5322, 0.0],
    )
