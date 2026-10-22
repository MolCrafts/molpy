"""LammpsEngine: the relaxation deck's styles, coordinate splicing."""

import shutil

import numpy as np
import pytest

import molpy as mp
from molpy.engine.lammps import _splice_coords


def _lmp():
    """A config naming the ``lmp`` binary (not the auto-detected one)."""
    return mp.config.load_config({"engine": {"lammps": {"executable": "lmp"}}})


def _minimize_deck(tmp_path, monkeypatch, ff, frame, **kw):
    """Run ``minimize`` with LAMMPS stubbed out; return the input script."""
    seen = {}

    def run(script, *, workdir, **_kw):
        seen["script"] = script.text
        shutil.copy(workdir / "system.data", workdir / "relaxed.data")

    engine = mp.engine.LammpsEngine(config=_lmp(), check_executable=False)
    monkeypatch.setattr(engine, "run", run)
    engine.minimize(frame, ff, workdir=tmp_path, **kw)
    return seen["script"].splitlines()


def _ct_dimer() -> mp.Frame:
    frame = mp.Frame(
        blocks={
            "atoms": {
                "id": [1, 2],
                "type": ["CT", "CT"],
                "x": [0.0, 1.5],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "charge": [0.1, -0.1],
                "mol_id": [1, 1],
            },
            "bonds": {"atomi": [0], "atomj": [1], "type": ["CT-CT"]},
        }
    )
    frame.box = mp.Box(mp.core.Cell.lengths([10.0] * 3))
    return frame


def _ct_forcefield(*pair_styles: tuple[str, dict]) -> mp.ff.forcefield.ForceField:
    ff = mp.ff.forcefield.ForceField("t", units="real")
    ct = ff.def_style("atom", "full").def_type("CT", mass=12.011)
    ff.def_style("bond", "harmonic").def_type("CT-CT", ct, ct, k=1.0, r0=1.5)
    for name, params in pair_styles:
        style = ff.def_style("pair", name, params)
        if name == "lj/cut":
            style.def_type("CT", ct, epsilon=0.1, sigma=3.0)
        elif name == "buck":
            style.def_type("CT", ct, a=1000.0, rho=0.3, c=10.0)
    return ff


def _styled(lines):
    return [line for line in lines if line and line.split()[0].endswith("_style")]


def test_minimize_keeps_the_force_fields_own_pair_style(tmp_path, monkeypatch):
    """A 12 Å ``lj/cut/coul/cut`` force field minimises at 12 Å with a cut
    Coulomb: the deck writes the force field's pair style, not one of its own."""
    ff = _ct_forcefield(("lj/cut", {"cutoff": 12.0}), ("coul/cut", {"cutoff": 12.0}))
    script = _minimize_deck(tmp_path, monkeypatch, ff, _ct_dimer())
    init = (tmp_path / "system.in.init").read_text(encoding="utf-8").splitlines()
    settings = (tmp_path / "system.in.settings").read_text(encoding="utf-8")
    assert _styled(init) == ["atom_style full"]
    assert "pair_style lj/cut/coul/cut 12.000000 12.000000" in settings.splitlines()
    assert not [line for line in script if line.startswith("kspace_style")]


def test_minimize_keeps_a_buckingham_pair_style(tmp_path, monkeypatch):
    """A ``buck`` force field's ``pair_coeff`` rows are read as Buckingham."""
    ff = _ct_forcefield(("buck", {"cutoff": 10.0}))
    _minimize_deck(tmp_path, monkeypatch, ff, _ct_dimer())
    settings = (tmp_path / "system.in.settings").read_text(encoding="utf-8")
    assert "pair_style buck 10.000000" in settings.splitlines()


def test_a_long_range_force_field_gets_its_kspace_solver(tmp_path, monkeypatch):
    ff = _ct_forcefield(
        ("lj/cut", {"cutoff": 10.0}), ("coul/long/pme", {"cutoff": 10.0})
    )
    script = _minimize_deck(tmp_path, monkeypatch, ff, _ct_dimer())
    settings = (tmp_path / "system.in.settings").read_text(encoding="utf-8")
    assert "pair_style lj/cut/coul/long 10.000000 10.000000" in settings.splitlines()
    assert script.index("kspace_style pppm 1.0e-4") > script.index(
        "include system.in.settings"
    )


def test_an_explicit_pair_style_replaces_the_force_fields(tmp_path, monkeypatch):
    ff = _ct_forcefield(
        ("lj/cut", {"cutoff": 10.0}), ("coul/long/pme", {"cutoff": 10.0})
    )
    script = _minimize_deck(
        tmp_path, monkeypatch, ff, _ct_dimer(), pair_style="lj/cut/coul/cut 8.0"
    )
    init = (tmp_path / "system.in.init").read_text(encoding="utf-8").splitlines()
    settings = (tmp_path / "system.in.settings").read_text(encoding="utf-8")
    assert "pair_style lj/cut/coul/cut 8.0" in init
    assert "pair_style" not in settings
    assert not [line for line in script if line.startswith("kspace_style")]


def test_relaxation_styles_come_from_the_molrs_include(tmp_path, monkeypatch):
    """The deck (``generate_inputs``) sets only ``atom_style`` in the init;
    every other ``*_style`` line is in the settings molrs writes, after
    ``read_data``, so any style molrs can write (``hybrid`` included) reaches
    LAMMPS."""
    ff = mp.ff.forcefield.ForceField("t", units="real")
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
    frame.box = mp.Box(mp.core.Cell.lengths([10.0] * 3))
    seen = {}

    def run(script, *, workdir, **_kw):
        seen["script"] = script.text
        shutil.copy(workdir / "system.data", workdir / "relaxed.data")

    engine = mp.engine.LammpsEngine(config=_lmp(), check_executable=False)
    monkeypatch.setattr(engine, "run", run)
    engine.minimize(frame, ff, workdir=tmp_path)
    script = seen["script"].splitlines()
    init = (tmp_path / "system.in.init").read_text(encoding="utf-8").splitlines()
    styled = [
        line for line in init + script if line and line.split()[0].endswith("_style")
    ]
    assert styled == [
        "atom_style full",
        "thermo_style custom step temp pe ke etotal press",
    ]
    assert "include system.in.init" in script
    assert "write_data relaxed.data nocoeff" in script
    settings = (
        (tmp_path / "system.in.settings").read_text(encoding="utf-8").splitlines()
    )
    assert "bond_style harmonic" in settings
    assert "pair_style lj/cut 10.000000" in settings
    # The force field's 1-4 weights and mixing rule are in the include too
    # (LAMMPS's defaults are 0 0 0 and geometric).
    lj, coul = ff.special_bonds
    assert (
        "special_bonds lj {:.6f} {:.6f} {:.6f} coul {:.6f} {:.6f} {:.6f}".format(
            *lj, *coul
        )
        in settings
    )
    assert "pair_modify mix arithmetic" in settings


def _frame(ids, x):
    frame = mp.Frame()
    frame["atoms"] = {
        "id": np.array(ids),
        "x": np.array(x, dtype=float),
        "y": np.zeros(len(ids)),
        "z": np.zeros(len(ids)),
        "type": np.array(["A"] * len(ids)),
    }
    frame.box = mp.Box(mp.core.Cell.lengths([10.0] * 3))
    return frame


def test_splice_matches_relaxed_coordinates_by_id():
    original = _frame([1, 2], [0.0, 1.0])
    relaxed = _frame([2, 1], [7.0, 5.0])  # reversed order, moved atoms
    out = _splice_coords(original, relaxed)
    assert out["atoms"]["x"].tolist() == [5.0, 7.0]
    assert list(out["atoms"]["type"]) == ["A", "A"]
    assert original["atoms"]["x"].tolist() == [0.0, 1.0], "input is not mutated"
    np.testing.assert_allclose(out.box.matrix, original.box.matrix)


def test_splice_rejects_a_changed_atom_count():
    with pytest.raises(RuntimeError, match="atom count changed"):
        _splice_coords(_frame([1, 2], [0.0, 1.0]), _frame([1], [0.0]))
