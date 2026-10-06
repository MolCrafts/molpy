"""LAMMPSEngine: the relaxation deck's styles, coordinate splicing."""

import os
import shutil

import numpy as np
import pytest

import molpy as mp
from molpy.engine import LAMMPSEngine
from molpy.engine.lammps import _splice_coords


def test_relaxation_styles_come_from_the_molrs_include(tmp_path, monkeypatch):
    """The script sets only ``pair_style``; every other ``*_style`` line is in
    the settings molrs writes, after ``read_data``, so any style molrs can
    write (``hybrid`` included) reaches LAMMPS."""
    ff = mp.ForceField("t", units="real")
    ct = ff.def_style("atom", "full").def_type("CT", mass=12.011)
    ff.def_style("bond", "harmonic").def_type("CT-CT", ct, ct, k=1.0, r0=1.5)
    ff.def_style("pair", "lj/cut", {"cutoff": 10.0}).def_type(
        "CT", ct, epsilon=0.1, sigma=3.0
    )
    frame = mp.Frame(
        blocks={
            "atoms": {
                "id": [1, 2],
                "type": ["CT", "CT"],
                "x": [0.0, 1.5],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "charge": [0.0, 0.0],
                "mol_id": [1, 1],
            },
            "bonds": {"atomi": [0], "atomj": [1], "type": ["CT-CT"]},
        }
    )
    frame.box = mp.Box.cube(10.0)
    seen = {}

    def run(script, *, workdir, **_kw):
        seen["script"] = script.text
        shutil.copy(workdir / "system.data", workdir / "relaxed.data")

    engine = LAMMPSEngine("lmp", check_executable=False)
    monkeypatch.setattr(engine, "run", run)
    engine.minimize(frame, ff, workdir=tmp_path)
    script = seen["script"].splitlines()
    assert [line for line in script if line.split()[0].endswith("_style")] == [
        "atom_style full",
        "pair_style lj/cut/coul/cut 10.0",
        "thermo_style custom step temp pe ke etotal press",
    ]
    settings = (tmp_path / "system.in.settings").read_text().splitlines()
    assert "bond_style harmonic" in settings
    assert not [line for line in settings if line.startswith("pair_style")]
    # Only the pair_style line is the script's: the force field's 1-4 weights
    # and mixing rule stay in the include (LAMMPS's defaults are 0 0 0 and
    # geometric).
    lj, coul = ff.special_bonds
    assert (
        "special_bonds lj {:.6f} {:.6f} {:.6f} coul {:.6f} {:.6f} {:.6f}".format(
            *lj, *coul
        )
        in settings
    )
    assert "pair_modify mix arithmetic" in settings


@pytest.mark.skipif(shutil.which("lmp") is None, reason="needs the lmp executable")
def test_relaxation_runs_with_the_force_field_special_bonds(tmp_path, monkeypatch):
    """A real minimisation: LAMMPS reads the include after the script's
    ``pair_style`` and keeps the force field's ``special_bonds``."""
    ff = mp.ForceField("t", units="real")
    ff.set_special_bonds([0.0, 0.0, 0.5], [0.0, 0.0, 0.8333])
    ct = ff.def_style("atom", "full").def_type("CT", mass=12.011)
    ff.def_style("bond", "harmonic").def_type("CT-CT", ct, ct, k=300.0, r0=1.5)
    ff.def_style("pair", "lj/cut", {"cutoff": 10.0}).def_type(
        "CT", ct, epsilon=0.1, sigma=3.0
    )
    frame = mp.Frame(
        blocks={
            "atoms": {
                "id": [1, 2],
                "type": ["CT", "CT"],
                "x": [10.0, 11.6],
                "y": [10.0, 10.0],
                "z": [10.0, 10.0],
                "charge": [0.0, 0.0],
                "mol_id": [1, 1],
            },
            "bonds": {"atomi": [0], "atomj": [1], "type": ["CT-CT"]},
        }
    )
    frame.box = mp.Box.cube(30.0)
    # A singleton run: inside a Slurm step, MPI must not join the step's PMI.
    for key in [
        k for k in os.environ if k.startswith(("PMI", "PMIX", "SLURM", "OMPI"))
    ]:
        monkeypatch.delenv(key)
    relaxed = LAMMPSEngine("lmp").minimize(frame, ff, workdir=tmp_path)
    log = (tmp_path / "log.lammps").read_text()
    assert (
        "special_bonds lj 0.000000 0.000000 0.500000 coul 0.000000 0.000000 0.833300"
        in log
    )
    assert "pair_modify mix arithmetic" in log
    bond = np.diff(np.asarray(relaxed["atoms"]["x"]))[0]
    assert bond == pytest.approx(1.5, abs=1e-3)


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
