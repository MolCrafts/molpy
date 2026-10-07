"""``mp.io.read_xyz`` and ``mp.io.write_xyz``.

molrs's reader, by identity: an n-wide extended-XYZ property is one ``(N, n)``
column and ``species`` is ``element``. Fixtures live in ``tests-data/xyz``.
"""

from pathlib import Path

import numpy as np
import pytest

import molpy as mp


@pytest.fixture
def xyz_dir(TEST_DATA_DIR: Path) -> Path:
    return TEST_DATA_DIR / "xyz"


class TestReadXyz:
    def test_coordinates_become_three_flat_columns(self, xyz_dir):
        atoms = mp.io.read_xyz(xyz_dir / "methane.xyz")["atoms"]
        assert atoms.n_rows == 5
        for key in ("x", "y", "z"):
            assert atoms[key].shape == (5,)
            assert atoms[key].dtype == np.float64
        np.testing.assert_allclose(
            atoms["x"], [0.0, 0.631716, -0.631716, 0.631716, -0.631716]
        )
        np.testing.assert_allclose(
            atoms["z"], [0.0, 0.631716, 0.631716, -0.631716, -0.631716]
        )

    def test_element_symbols_are_read(self, xyz_dir):
        atoms = mp.io.read_xyz(xyz_dir / "methane.xyz")["atoms"]
        assert list(atoms["element"]) == ["C", "H", "H", "H", "H"]
        assert mp.io.read_xyz is __import__("molrs").io.read_xyz

    def test_ragged_spacing_and_trailing_blank_lines_are_tolerated(self, xyz_dir):
        # Leading blanks on the count line, uneven columns, empty lines after
        # the last atom.
        frame = mp.io.read_xyz(xyz_dir / "ragged.xyz")
        assert frame["atoms"].n_rows == 3
        np.testing.assert_allclose(frame["atoms"]["y"], [0.1005, 0.5004, 0.7003])
        assert frame.box is None

    def test_free_text_comment_is_kept_verbatim(self, xyz_dir):
        frame = mp.io.read_xyz(xyz_dir / "ragged.xyz")
        assert (
            frame.meta["comment"] == " Too much empty new line at the end of the file"
        )
        assert mp.io.read_xyz(xyz_dir / "methane.xyz").meta["comment"] == ""

    def test_lattice_vectors_become_the_box_columns(self, xyz_dir):
        # lattice.xyz: Lattice="10 0 0 2 11 0 3 4 12" lists R1 R2 R3.
        box = mp.io.read_xyz(xyz_dir / "lattice.xyz").box
        np.testing.assert_allclose(
            box.h, [[10.0, 2.0, 3.0], [0.0, 11.0, 4.0], [0.0, 0.0, 12.0]]
        )

    def test_species_property_is_the_element(self, xyz_dir):
        atoms = mp.io.read_xyz(xyz_dir / "lattice.xyz")["atoms"]
        assert list(atoms["element"]) == ["O", "C", "H"]

    def test_two_wide_property_is_rejoined(self, xyz_dir):
        atoms = mp.io.read_xyz(xyz_dir / "lattice.xyz")["atoms"]
        assert atoms["CS"].shape == (3, 2)
        np.testing.assert_allclose(atoms["CS"][1], [-74.64, -90.59])
        assert "CS_1" not in atoms
        assert "CS_2" not in atoms

    def test_three_wide_property_is_rejoined_whole(self, xyz_dir):
        atoms = mp.io.read_xyz(xyz_dir / "velocities.xyz")["atoms"]
        assert atoms["velo"].shape == (3, 3)
        np.testing.assert_allclose(atoms["velo"], np.eye(3))
        assert not any(key.startswith("velo_") for key in atoms.keys())

    def test_comment_line_scalars_keep_their_types(self, xyz_dir):
        meta = mp.io.read_xyz(xyz_dir / "lattice.xyz").meta
        assert meta["ENERGY"] == pytest.approx(-2069.85)
        assert isinstance(meta["ENERGY"], float)
        assert meta["Natoms"] == 3
        assert isinstance(meta["Natoms"], int)
        assert meta["NAME"] == "COBHUW"
        assert meta["IsStrange"] is True


class TestWriteXyz:
    def test_written_file_reads_back_to_the_same_atoms(self, tmp_path):
        frame = mp.Frame()
        frame["atoms"] = {
            "element": np.array(["O", "H", "H"]),
            "x": np.array([0.0, 0.96, -0.24]),
            "y": np.array([0.0, 0.0, 0.93]),
            "z": np.zeros(3),
        }
        path = tmp_path / "out.xyz"
        mp.io.write_xyz(path, frame)
        lines = path.read_text().splitlines()
        assert lines[0] == "3"
        assert len(lines) == 5

        back = mp.io.read_xyz(path)["atoms"]
        assert list(back["element"]) == ["O", "H", "H"]
        np.testing.assert_allclose(back["x"], [0.0, 0.96, -0.24])
        np.testing.assert_allclose(back["y"], [0.0, 0.0, 0.93])
