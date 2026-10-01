"""LAMMPSEngine helpers: style lines from a force field, coordinate splicing."""

import numpy as np
import pytest

import molpy as mp
from molpy.engine.lammps import _splice_coords, _style_lines


def test_style_lines_name_the_bonded_styles_only():
    ff = mp.ForceField("t", units="real")
    ct = ff.def_style("atom", "full").def_type("CT", mass=12.011)
    ff.def_style("bond", "harmonic").def_type("CT-CT", ct, ct, k=1.0, r0=1.5)
    ff.def_style("pair", "lj/cut", {"cutoff": 10.0}).def_type(
        "CT", ct, epsilon=0.1, sigma=3.0
    )
    assert _style_lines(ff) == ["bond_style harmonic"]


def _frame(ids, x):
    frame = mp.Frame()
    frame["atoms"] = {
        "id": np.array(ids),
        "x": np.array(x, dtype=float),
        "y": np.zeros(len(ids)),
        "z": np.zeros(len(ids)),
        "type": np.array(["A"] * len(ids)),
    }
    frame.box = mp.Box.cube(10.0)
    return frame


def test_splice_matches_relaxed_coordinates_by_id():
    original = _frame([1, 2], [0.0, 1.0])
    relaxed = _frame([2, 1], [7.0, 5.0])  # reversed order, moved atoms
    out = _splice_coords(original, relaxed)
    assert out["atoms"]["x"].tolist() == [5.0, 7.0]
    assert list(out["atoms"]["type"]) == ["A", "A"]
    assert original["atoms"]["x"].tolist() == [0.0, 1.0], "input is not mutated"
    np.testing.assert_allclose(out.box.h, original.box.h)


def test_splice_rejects_a_changed_atom_count():
    with pytest.raises(RuntimeError, match="atom count changed"):
        _splice_coords(_frame([1, 2], [0.0, 1.0]), _frame([1], [0.0]))
