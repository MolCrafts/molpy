"""``Lattice.from_cif``: a CIF read by ``mp.io.read_cif`` becomes a full-cell lattice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import molpy as mp

TRICLINIC_FRACTIONAL = np.array([[0.10, 0.20, 0.30], [0.25, 0.35, 0.15]])

MINIMAL_CIF = """\
data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
{symops}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Cu1 Cu 0.0 0.0 0.0 {occupancy}
"""

P1_LOOP = "loop_\n_symmetry_equiv_pos_as_xyz\nx,y,z"


def _cell_matrix(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Lattice vectors as rows, a along x and b in the xy plane (the CIF convention)."""
    al, be, ga = np.radians([alpha, beta, gamma])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    return np.array(
        [
            [a, 0.0, 0.0],
            [b * np.cos(ga), b * np.sin(ga), 0.0],
            [cx, cy, np.sqrt(c * c - cx * cx - cy * cy)],
        ]
    )


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "structure.cif"
    path.write_text(text, encoding="utf-8")
    return path


def _sorted_rows(points: np.ndarray) -> np.ndarray:
    return points[np.lexsort(points.T[::-1])]


class TestFromCif:
    def test_nacl_fm3m_fills_eight_atoms_per_cell(self, TEST_DATA_DIR):
        frame = mp.io.read_cif(TEST_DATA_DIR / "cif" / "nacl_fm3m.cif")
        lattice = mp.builder.Lattice.from_cif(frame)

        assert len(lattice.basis) == 8
        species = [site.species for site in lattice.basis]
        assert species.count("Na") == 4
        assert species.count("Cl") == 4
        na = {
            tuple(np.round(s.fractional, 6)) for s in lattice.basis if s.species == "Na"
        }
        assert na == {
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        }
        assert {s.label for s in lattice.basis if s.species == "Cl"} == {
            f"Cl1_{k}" for k in range(4)
        }
        np.testing.assert_allclose(lattice.box.matrix, 5.6402 * np.eye(3), atol=1e-12)

        structure = lattice.build(repeats=(2, 2, 2))
        assert len(list(structure.atoms)) == 64

    def test_triclinic_build_matches_the_file(self, TEST_DATA_DIR):
        """read_cif -> from_cif -> build puts every atom where the file says.

        The expected positions come from the cell parameters, not from molrs,
        so a transposed cell matrix (rows taken for columns) fails here.
        """
        frame = mp.io.read_cif(TEST_DATA_DIR / "cif" / "triclinic_p1bar.cif")
        structure = mp.builder.Lattice.from_cif(frame).build(repeats=(1, 1, 1))

        rows = _cell_matrix(5.0, 6.0, 7.0, 80.0, 95.0, 110.0)
        images = np.vstack([TRICLINIC_FRACTIONAL, 1.0 - TRICLINIC_FRACTIONAL])
        expected = images @ rows

        built = structure.atoms["x", "y", "z"]
        np.testing.assert_allclose(_sorted_rows(built), _sorted_rows(expected))

        # The asymmetric unit lands exactly on the coordinates read_cif reports.
        atoms = frame["atoms"]
        file_xyz = np.column_stack([atoms["x"], atoms["y"], atoms["z"]])
        labels = structure.atoms["label"].tolist()
        for name, xyz in zip(atoms["name"].tolist(), file_xyz, strict=True):
            np.testing.assert_allclose(built[labels.index(f"{name}_0")], xyz)

    def test_missing_symops_is_an_error_not_p1(self, tmp_path):
        frame = mp.io.read_cif(
            _write(tmp_path, MINIMAL_CIF.format(symops="", occupancy="1.0"))
        )
        assert mp.core.keys.SYMOPS.key not in frame.meta

        with pytest.raises(ValueError, match="symops"):
            mp.builder.Lattice.from_cif(frame)

    def test_explicit_p1_symops_are_honoured(self, tmp_path):
        frame = mp.io.read_cif(
            _write(tmp_path, MINIMAL_CIF.format(symops=P1_LOOP, occupancy="1.0"))
        )
        lattice = mp.builder.Lattice.from_cif(frame)

        assert [s.label for s in lattice.basis] == ["Cu1_0"]

    def test_missing_box_is_an_error(self):
        frame = mp.Frame()
        frame["atoms"] = {
            "x": np.zeros(1),
            "y": np.zeros(1),
            "z": np.zeros(1),
            "element": np.array(["Cu"]),
            "name": np.array(["Cu1"]),
        }
        frame.meta[mp.core.keys.SYMOPS.key] = ("x,y,z",)

        with pytest.raises(ValueError, match="box"):
            mp.builder.Lattice.from_cif(frame)

    def test_partial_occupancy_is_rejected(self, tmp_path):
        frame = mp.io.read_cif(
            _write(tmp_path, MINIMAL_CIF.format(symops=P1_LOOP, occupancy="0.5"))
        )

        with pytest.raises(ValueError, match=r"Cu1.*partial occupancy"):
            mp.builder.Lattice.from_cif(frame)
