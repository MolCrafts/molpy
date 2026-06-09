"""Parity tests: molpy experimental GRO I/O (molrs-backed) vs molpy native."""

import numpy as np
import pytest

from molpy.io import read_gro as native_read_gro
from molpy.io import write_gro as native_write_gro
from molpy.io.experimental import read_gro as exp_read_gro
from molpy.io.experimental import write_gro as exp_write_gro


@pytest.fixture
def gro_path(TEST_DATA_DIR):
    # molpy ships its own test corpus (github.com/molcrafts/tests-data), cloned
    # on demand by the session-scoped TEST_DATA_DIR fixture — no dependency on a
    # sibling molrs checkout.
    p = TEST_DATA_DIR / "gro" / "ubiquitin.gro"
    if not p.exists():
        pytest.skip(f"GRO test file not found: {p}")
    return str(p)


class TestGroReaderParity:
    def test_same_atom_count(self, gro_path):
        f_native = native_read_gro(gro_path)
        f_exp = exp_read_gro(gro_path)
        assert f_native["atoms"].nrows == f_exp["atoms"].nrows

    def test_coordinates_match(self, gro_path):
        f_native = native_read_gro(gro_path)
        f_exp = exp_read_gro(gro_path)
        for c in ["x", "y", "z"]:
            np.testing.assert_allclose(
                np.asarray(f_native["atoms"][c]),
                np.asarray(f_exp["atoms"][c]),
                rtol=1e-10,
                err_msg=f"coordinate {c} mismatch",
            )

    def test_residue_info_match(self, gro_path):
        f_native = native_read_gro(gro_path)
        f_exp = exp_read_gro(gro_path)
        assert np.array_equal(
            np.asarray(f_native["atoms"]["res_name"]),
            np.asarray(f_exp["atoms"]["res_name"]),
        )

    def test_column_names_parity(self, gro_path):
        f_native = native_read_gro(gro_path)
        f_exp = exp_read_gro(gro_path)
        native_keys = set(f_native["atoms"].keys())
        exp_keys = set(f_exp["atoms"].keys())
        missing = native_keys - exp_keys
        # Allow experimental to lack 'res_number' (molrs uses 'res_id')
        missing.discard("res_number")
        assert not missing, f"experimental reader missing columns: {missing}"

    def test_atomic_numbers_parity(self, gro_path):
        f_native = native_read_gro(gro_path)
        f_exp = exp_read_gro(gro_path)
        np.testing.assert_array_equal(
            np.asarray(f_native["atoms"]["number"]),
            np.asarray(f_exp["atoms"]["number"]),
        )


class TestGroWriterParity:
    def test_round_trip_via_experimental(self, gro_path, tmp_path):
        f_native = native_read_gro(gro_path)
        out = tmp_path / "out.gro"
        exp_write_gro(str(out), f_native)
        f_rt = native_read_gro(str(out))
        assert f_native["atoms"].nrows == f_rt["atoms"].nrows
        for c in ["x", "y", "z"]:
            np.testing.assert_allclose(
                np.asarray(f_native["atoms"][c]),
                np.asarray(f_rt["atoms"][c]),
                atol=1e-3,
                err_msg=f"round-trip coordinate {c} mismatch",
            )

    def test_native_read_experimental_write_parity(self, gro_path, tmp_path):
        f_original = native_read_gro(gro_path)
        out = tmp_path / "out.gro"
        exp_write_gro(str(out), f_original)
        f_rt = native_read_gro(str(out))
        assert f_original["atoms"].nrows == f_rt["atoms"].nrows

    def test_experimental_read_native_write_parity(self, gro_path, tmp_path):
        f_exp = exp_read_gro(gro_path)
        out = tmp_path / "out.gro"
        native_write_gro(str(out), f_exp)
        f_rt = native_read_gro(str(out))
        assert f_exp["atoms"].nrows == f_rt["atoms"].nrows
