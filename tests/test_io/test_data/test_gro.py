"""GROMACS .gro reader and writer.

The reader scales nm to Å, infers the element from the atom name, reads the
optional velocity columns and both box-line forms, and rejects a malformed
record instead of guessing. Fixtures live in ``tests-data/gro``: two_waters
(velocities, orthogonal box), triclinic (nine-number box line),
truncated_record and no_box.
"""

from pathlib import Path

import numpy as np
import pytest

import molpy as mp


@pytest.fixture
def gro_dir(TEST_DATA_DIR: Path) -> Path:
    return TEST_DATA_DIR / "gro"


def _read(path: Path) -> mp.Frame:
    return mp.io.read_gro(path)


class TestReadGro:
    def test_records_are_split_into_canonical_columns(self, gro_dir):
        atoms = _read(gro_dir / "two_waters.gro")["atoms"]
        assert atoms.nrows == 6
        assert atoms["res_id"].tolist() == [1, 1, 1, 2, 2, 2]
        assert list(atoms["res_name"]) == ["WAT"] * 6
        assert list(atoms["name"]) == ["OW", "HW1", "HW2", "OW", "HW1", "HW2"]
        assert atoms["id"].tolist() == [1, 2, 3, 4, 5, 6]
        assert "atomic_number" not in atoms

    def test_coordinates_are_scaled_from_nm_to_angstrom(self, gro_dir):
        atoms = _read(gro_dir / "two_waters.gro")["atoms"]
        xyz = atoms["x", "y", "z"]
        assert xyz.dtype == np.float64
        np.testing.assert_allclose(xyz[0], [3.10, 8.62, 13.16])
        np.testing.assert_allclose(xyz[3], [10.0, 10.0, 10.0])

    def test_element_is_inferred_from_the_atom_name(self, gro_dir):
        atoms = _read(gro_dir / "two_waters.gro")["atoms"]
        assert list(atoms["element"]) == ["O", "H", "H", "O", "H", "H"]

    def test_velocity_columns_are_read_and_scaled(self, gro_dir):
        atoms = _read(gro_dir / "two_waters.gro")["atoms"]
        np.testing.assert_allclose(atoms["vx", "vy", "vz"][0], [1.0, 2.0, 3.0])
        assert "vx" not in _read(gro_dir / "triclinic.gro")["atoms"]

    def test_title_line_is_kept_as_meta(self, gro_dir):
        assert _read(gro_dir / "two_waters.gro").meta["title"] == "Two waters"

    def test_three_number_box_line_is_orthogonal(self, gro_dir):
        box = _read(gro_dir / "two_waters.gro").box
        np.testing.assert_allclose(box.h, np.diag([20.0, 30.0, 40.0]))

    def test_nine_number_box_line_fills_the_off_diagonals(self, gro_dir):
        # v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y); lattice
        # vectors are the columns of the matrix, in Å.
        box = _read(gro_dir / "triclinic.gro").box
        expected = [[20.0, 5.0, 6.0], [0.0, 30.0, 7.0], [0.0, 0.0, 40.0]]
        np.testing.assert_allclose(box.h, expected)

    def test_only_the_first_frame_of_a_multi_frame_file_is_returned(
        self, gro_dir, tmp_path
    ):
        two = tmp_path / "two.gro"
        two.write_text(
            (gro_dir / "triclinic.gro").read_text()
            + (gro_dir / "two_waters.gro").read_text()
        )
        frame = _read(two)
        assert frame["atoms"].nrows == 1
        assert frame.meta["title"] == "Triclinic"

    def test_missing_trailing_newline_is_tolerated(self, gro_dir, tmp_path):
        bare = tmp_path / "bare.gro"
        bare.write_text((gro_dir / "triclinic.gro").read_text().rstrip("\n"))
        assert _read(bare)["atoms"].nrows == 1

    def test_short_atom_record_raises(self, gro_dir):
        with pytest.raises(OSError, match="too short"):
            _read(gro_dir / "truncated_record.gro")

    def test_missing_box_line_raises(self, gro_dir):
        with pytest.raises(OSError, match="box"):
            _read(gro_dir / "no_box.gro")

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.gro"
        empty.write_text("")
        with pytest.raises(OSError):
            _read(empty)

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            _read(tmp_path / "nonexistent.gro")


class TestWriteGro:
    def _water(self) -> mp.Frame:
        frame = mp.Frame()
        frame["atoms"] = {
            "res_id": [1, 1, 1],
            "res_name": ["WAT", "WAT", "WAT"],
            "name": ["OW", "HW1", "HW2"],
            "x": [0.0, 1.0, -0.33],
            "y": [0.0, 0.0, 0.94],
            "z": [0.0, 0.0, 0.0],
        }
        frame.box = mp.Box(h=np.diag([20.0, 30.0, 40.0]))
        return frame

    def test_record_layout(self, tmp_path):
        path = tmp_path / "out.gro"
        mp.io.write_gro(path, self._water())
        lines = path.read_text().splitlines()
        assert len(lines) == 6
        assert lines[1].strip() == "3"
        # Fixed columns: resid(5) resname(5) name(5) serial(5) then 8.3f in nm.
        assert lines[2] == "    1WAT     OW    1   0.000   0.000   0.000"
        assert lines[3] == "    1WAT    HW1    2   0.100   0.000   0.000"

    def test_box_is_written_in_nm(self, tmp_path):
        path = tmp_path / "out.gro"
        mp.io.write_gro(path, self._water())
        assert path.read_text().splitlines()[-1].split() == [
            "2.00000",
            "3.00000",
            "4.00000",
        ]

    def test_round_trip_preserves_coordinates_and_names(self, tmp_path):
        path = tmp_path / "rt.gro"
        mp.io.write_gro(path, self._water())
        back = _read(path)["atoms"]
        np.testing.assert_allclose(
            back["x", "y", "z"], self._water()["atoms"]["x", "y", "z"], atol=5e-3
        )
        assert list(back["name"]) == ["OW", "HW1", "HW2"]
