"""What every engine writer passes over, generates, or refuses.

- The keys that price nothing — an estimate's provenance (``estimated``,
  ``estimate_penalty``, ``estimate_method``, ``estimate_analog``) and an
  annotation — pass every force-field writer, and a style parameter at its
  spec default (``coul/cut``'s ``delta = 0``) needs no column; any other
  parameter a format cannot place is refused by name.
- A null cell (what ``Frame.concat`` leaves where a part lacks a column) is
  never written as a ``0`` or an empty field: an unset label is generated, an
  ``id``, charge or coordinate is refused by column and rows (``OSError``
  from a frame writer, as its other refusals).

The Rust tests (``molrs/src/ff/engine_codec_check.rs``,
``molrs/src/io/null_cells.rs``) own the cases; these prove the seam.
"""

from __future__ import annotations

from pathlib import Path

import molrs
import numpy as np
import pytest

ForceField = molrs.ff.forcefield.ForceField


def _ethylene() -> molrs.core.Atomistic:
    mol = molrs.core.Atomistic()
    c1 = mol.def_atom(element="C", type="c2", charge=-0.2, x=0.0, y=0.0, z=0.0)
    c2 = mol.def_atom(element="C", type="c2", charge=-0.2, x=1.3, y=0.0, z=0.0)
    mol.def_bond(c1, c2)
    for c, (x, y) in ((c1, (-0.6, 0.9)), (c1, (-0.6, -0.9)), (c2, (1.9, 0.9)), (c2, (1.9, -0.9))):
        mol.def_bond(c, mol.def_atom(element="H", type="ha", charge=0.1, x=x, y=y, z=0.0))
    return mol


def _native_gaff() -> tuple[ForceField, molrs.core.Frame]:
    """Ethylene typed by the native GAFF route: its impropers are parmchk2
    estimates, carrying their provenance."""
    gaff = molrs.ff.typifier.GaffTypifier(parameter_set="gaff")
    frame = gaff.typify(_ethylene()).to_frame()
    ff = gaff.forcefield()
    for style in ("lj/cut", "coul/cut"):
        ff.get_style("pair", style)["cutoff"] = 10.0
    return ff, frame


def _every_writer(ff: ForceField, frame: molrs.core.Frame, tmp_path: Path) -> dict[str, str]:
    top = tmp_path / "system.top"
    molrs.io.write_top_system(top, ff, frame)
    return {
        "LAMMPS": molrs.io.write_lammps_forcefield_str(ff, frame),
        "GROMACS": top.read_text(encoding="utf-8"),
        "OpenMM": molrs.io.write_openmm_forcefield_str(ff),
        "frcmod": molrs.io.write_frcmod_str(ff),
    }


def test_a_native_gaff_field_with_estimates_writes_to_every_engine(tmp_path: Path) -> None:
    ff, frame = _native_gaff()
    (improper,) = ff.get_types("improper")
    assert improper.params["estimated"] == 1.0
    assert ff.get_style("pair", "coul/cut").params["delta"] == 0.0
    written = _every_writer(ff, frame, tmp_path)
    for engine in ("LAMMPS", "GROMACS", "OpenMM"):
        assert "estimate" not in written[engine], engine
    # frcmod states the provenance where parmchk2 does: the row's comment.
    assert "penalty score=" in written["frcmod"]


def test_a_parameter_no_format_holds_is_refused_by_every_writer(tmp_path: Path) -> None:
    ff = ForceField("bogus", units="real")
    atoms = ff.def_style("atom", "full")
    a, b = atoms.def_type("A", mass=12.0), atoms.def_type("B", mass=1.0)
    lj = ff.def_style("pair", "lj/cut", {"cutoff": 10.0, "mixing": "arithmetic"})
    lj.def_type("A", a, epsilon=0.1, sigma=3.4)
    lj.def_type("B", b, epsilon=0.02, sigma=2.6)
    ff.def_style("bond", "harmonic").def_type("A-B", a, b, k=340.0, r0=1.09, bogus=1.0)
    frame = _part(["C", "H"], ["A", "B"], [-0.1, 0.1], [(0, 1)])
    frame["atoms"]["mol_id"] = np.array([1, 1], dtype=np.uint64)
    with pytest.raises(ValueError, match="bogus"):
        molrs.io.write_lammps_forcefield_str(ff, frame)
    with pytest.raises(ValueError, match="bogus"):
        molrs.io.write_top_system(tmp_path / "x.top", ff, frame)
    with pytest.raises(ValueError, match="bogus"):
        molrs.io.write_openmm_forcefield_str(ff)
    with pytest.raises(ValueError, match="bogus"):
        molrs.io.write_frcmod_str(ff)


def test_coul_cut_delta_is_written_at_its_default_and_refused_off_it(tmp_path: Path) -> None:
    ff, frame = _native_gaff()
    _every_writer(ff, frame, tmp_path)
    ff.get_style("pair", "coul/cut")["delta"] = 0.5
    with pytest.raises(ValueError, match="delta"):
        molrs.io.write_top_system(tmp_path / "x.top", ff, frame)
    with pytest.raises(ValueError, match="delta"):
        molrs.io.write_lammps_forcefield_str(ff, frame)


# ---------------------------------------------------------------------------
# Null cells after Frame.concat
# ---------------------------------------------------------------------------


def _part(elements: list[str], types: list[str], charges: list[float], bonds) -> molrs.core.Frame:
    n = len(elements)
    atoms = molrs.core.Block()
    atoms.insert("element", elements)
    atoms.insert("type", types)
    atoms.insert("charge", np.array(charges))
    for d, key in enumerate("xyz"):
        atoms.insert(key, np.arange(n, dtype=float) + d)
    b = molrs.core.Block()
    b.insert("atomi", np.array([i for i, _ in bonds], dtype=np.uint64))
    b.insert("atomj", np.array([j for _, j in bonds], dtype=np.uint64))
    b.insert("type", [f"{types[i]}-{types[j]}" for i, j in bonds])
    frame = molrs.core.Frame()
    frame["atoms"] = atoms
    frame["bonds"] = b
    return frame


def _concat() -> molrs.core.Frame:
    """A water stated whole, then a C-H stating no id, name or residue."""
    water = _part(["O", "H", "H"], ["OW", "HW", "HW"], [-0.8, 0.4, 0.4], [(0, 1), (0, 2)])
    water["atoms"]["id"] = np.array([1, 2, 3], dtype=np.uint64)
    water["atoms"]["name"] = ["OW", "HW1", "HW2"]
    water["atoms"]["res_name"] = ["SOL"] * 3
    water["atoms"]["res_id"] = np.array([7, 7, 7], dtype=np.uint64)
    ch = _part(["C", "H"], ["CT", "HC"], [-0.1, 0.1], [(0, 1)])
    joined = molrs.core.Frame.concat([water, ch])
    assert list(joined["atoms"].validity("id")) == [True, True, True, False, False]
    joined["atoms"]["mol_id"] = np.array([1, 1, 1, 2, 2], dtype=np.uint64)
    return joined


def test_the_lammps_data_writer_refuses_a_null_id_and_numbers_a_missing_one() -> None:
    frame = _concat()
    with pytest.raises(OSError, match=r"atoms\.id is null in 2 of 5 rows \(rows 3-4"):
        molrs.io.write_lammps_data_str(frame)
    del frame["atoms"]["id"]
    text = molrs.io.write_lammps_data_str(frame)
    atoms = text.split("Atoms")[1].strip().splitlines()[2:7]
    assert [line.split()[0] for line in atoms] == ["1", "2", "3", "4", "5"]


def test_a_null_charge_is_refused_by_column_and_rows() -> None:
    frame = _concat()
    del frame["atoms"]["id"]
    frame["atoms"].set_validity("charge", [True, True, True, False, True])
    with pytest.raises(OSError, match=r"atoms\.charge is null in 1 of 5 rows \(row 3"):
        molrs.io.write_lammps_data_str(frame)


def test_gro_and_pdb_generate_unset_names_and_residues() -> None:
    frame = _concat()
    with pytest.raises(OSError, match=r"atoms\.id is null"):
        molrs.io.write_gro_str(frame)
    del frame["atoms"]["id"]
    lines = molrs.io.write_gro_str(frame).splitlines()[2:7]
    labels = [(line[0:5].strip(), line[5:10].strip(), line[10:15].strip()) for line in lines]
    assert labels == [
        ("7", "SOL", "OW"),
        ("7", "SOL", "HW1"),
        ("7", "SOL", "HW2"),
        ("2", "MOL", "C1"),
        ("2", "MOL", "H1"),
    ]
    pdb = [line for line in molrs.io.write_pdb_str(frame).splitlines() if line.startswith("ATOM")]
    assert (pdb[3][12:16].strip(), pdb[3][17:20]) == ("C1", "MOL")
