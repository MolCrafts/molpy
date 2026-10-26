"""``CoarseGrain`` FFI seam: a Rust refusal crosses as a Python exception."""

import molpy as mp
import numpy as np
import pytest


def test_from_frame_refuses_a_2d_atoms_column_with_value_error() -> None:
    # A graph property is one value per row; an (n, 3) column has no reading.
    # The refusal must be a ValueError, not pyo3's PanicException (which
    # derives from BaseException and so escapes ``pytest.raises(ValueError)``).
    frame = mp.core.Frame()
    frame["atoms"] = mp.core.Block(
        {
            "type": np.array(["A", "B"]),
            "xyz": np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64),
        }
    )
    with pytest.raises(ValueError, match="xyz"):
        mp.core.CoarseGrain.from_frame(frame)


def _three_beads() -> tuple[mp.core.CoarseGrain, list[mp.core.Bead]]:
    cg = mp.core.CoarseGrain()
    beads = [
        cg.add_bead(bead_type="W", x=0.5, y=-1.25, z=2.0),
        cg.add_bead(bead_type="P1", x=3.0, y=4.0, z=5.0),
        cg.add_bead(bead_type="P2", x=-0.75, y=8.5, z=0.125),
    ]
    return cg, beads


def test_positions_cross_as_float64_rows_in_the_order_asked() -> None:
    cg, (h0, _, h2) = _three_beads()

    positions = cg.positions([h2, h0])

    assert isinstance(positions, np.ndarray)
    assert positions.dtype == np.float64
    assert positions.shape == (2, 3)
    np.testing.assert_array_equal(
        positions, np.array([[-0.75, 8.5, 0.125], [0.5, -1.25, 2.0]])
    )


def test_bead_types_cross_as_a_list_of_str_in_the_order_asked() -> None:
    cg, (h0, h1, h2) = _three_beads()

    types = cg.bead_types([h1, h0, h2])

    assert types == ["P1", "W", "P2"]
    assert all(type(t) is str for t in types)


def test_a_stale_bead_is_a_value_error_for_both_accessors() -> None:
    cg, (h0, h1, _) = _three_beads()
    cg.del_bead(h1)

    with pytest.raises(ValueError, match=str(h1.handle)):
        cg.positions([h0, h1])
    with pytest.raises(ValueError, match=str(h1.handle)):
        cg.bead_types([h0, h1])


def test_per_bead_reads_take_a_refs() -> None:
    cg, _ = _three_beads()
    assert cg.bead_types(cg.beads) == ["W", "P1", "P2"]
    assert cg.positions(cg.beads[1:]).shape == (2, 3)


def test_a_bead_of_another_graph_is_a_value_error() -> None:
    cg, _ = _three_beads()
    _, (foreign, _, _) = _three_beads()
    with pytest.raises(ValueError, match="another graph"):
        cg.positions([foreign])
