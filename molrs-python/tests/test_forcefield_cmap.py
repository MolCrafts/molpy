"""The ``cmap`` category and array-valued params across the FFI seam.

A cmap type names five atom types (two consecutive dihedrals) and carries its
correction table as the array param ``grid``. Array params cross as float64
numpy arrays, compare exactly, survive the ``forcefield`` section and a
``*.mrec`` store, and pickle; a frame's ``cmaps`` block renumbers
``atomi`` … ``atomm``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import molrs
import numpy as np
import pytest


def _lammps_include(
    ff: molrs.ff.forcefield.ForceField, frame: molrs.core.Frame, **options: object
) -> str:
    """The LAMMPS force-field include ``ff`` writes for ``frame``."""
    buf = bytearray()
    molrs.io.lammps.LammpsForcefieldWriter(frame, **options).write(buf, ff)  # type: ignore[arg-type]
    return buf.decode()


ENDS = ("C", "NH1", "CT1", "C2", "NH2")


def _grid(n: int = 24, scale: float = 0.125) -> np.ndarray:
    return scale * np.arange(n * n, dtype=np.float64).reshape(n, n)


def _cmap_ff(grid: np.ndarray | None = None) -> molrs.ff.forcefield.ForceField:
    ff = molrs.ff.forcefield.ForceField("charmm", units="real")
    atoms = ff.def_style("atom", "full")
    ends = [atoms.def_type(name, mass=12.0) for name in ENDS]
    ff.def_style("cmap", "charmm").def_type(
        "-".join(ENDS), *ends, grid=_grid() if grid is None else grid
    )
    return ff


def test_a_cmap_style_takes_five_endpoints_and_a_grid() -> None:
    ff = _cmap_ff()
    style = ff.get_style("cmap", "charmm")
    assert isinstance(style, molrs.ff.forcefield.CmapStyle)
    (cmap,) = style.get_types()
    assert isinstance(cmap, molrs.ff.forcefield.CmapType)
    assert cmap.category == "cmap"
    assert [t.name for t in cmap.endpoints] == list(ENDS)
    assert (cmap.itom.name, cmap.mtom.name) == ("C", "NH2")
    grid = cmap["grid"]
    assert isinstance(grid, np.ndarray) and grid.dtype == np.float64
    np.testing.assert_array_equal(grid, _grid())
    assert ff.get_types(molrs.ff.forcefield.CmapType) == [cmap]
    assert ff.get_styles("cmap") == [style]


def test_an_array_param_is_taken_from_any_numeric_array_or_nested_list() -> None:
    ff = _cmap_ff(grid=[[1, 2], [3, 4]])
    (cmap,) = ff.get_types("cmap")
    np.testing.assert_array_equal(cmap["grid"], [[1.0, 2.0], [3.0, 4.0]])
    assert cmap["grid"].dtype == np.float64
    cmap["grid"] = np.zeros((3, 3), dtype=np.int32)
    assert cmap["grid"].shape == (3, 3) and cmap["grid"].dtype == np.float64
    with pytest.raises(TypeError):
        cmap["grid"] = [[1.0, 2.0], [3.0]]
    # A 0-d array is a number.
    cmap["scale"] = np.float64(0.5)
    assert cmap["scale"] == 0.5


def test_a_restatement_compares_its_arrays_exactly() -> None:
    ff = _cmap_ff()
    style = ff.get_style("cmap", "charmm")
    ends = ff.get_style("atom", "full").get_types()
    name = "-".join(ENDS)
    style.def_type(name, *ends, grid=_grid())
    nudged = _grid()
    nudged[3, 4] = np.nextafter(nudged[3, 4], np.inf)
    with pytest.raises(ValueError):
        style.def_type(name, *ends, grid=nudged)
    with pytest.raises(TypeError):
        style.def_type("short", *ends[:4], grid=_grid())


def test_a_cmap_grid_round_trips_through_the_section_and_a_store(
    tmp_path: Path,
) -> None:
    ff = _cmap_ff()
    section = molrs.io.mrec.ForceFieldSection.from_forcefield(ff)
    table = section.table("cmap", "charmm")
    assert list(table["mtom"]) == ["NH2"]
    assert table["grid"].shape == (1, 24, 24)

    path = tmp_path / "ff.mrec"
    molrs.io.write_mrec(path, forcefield=ff)
    back = molrs.io.mrec.MrecFile(path).forcefield().to_forcefield()
    (cmap,) = back.get_types("cmap")
    assert cmap["grid"].tobytes() == _grid().tobytes()


def test_a_grid_the_section_cannot_hold_is_refused() -> None:
    with pytest.raises(ValueError, match="cmap grid"):
        molrs.io.mrec.ForceFieldSection.from_forcefield(_cmap_ff(grid=np.zeros((2, 3))))


def test_a_cmap_force_field_pickles_with_its_grid() -> None:
    ff = _cmap_ff()
    back = pickle.loads(pickle.dumps(ff, protocol=pickle.HIGHEST_PROTOCOL))
    (cmap,) = back.get_types("cmap")
    assert cmap["grid"].tobytes() == _grid().tobytes()
    assert [t.name for t in cmap.endpoints] == list(ENDS)


def _two_cmaps() -> molrs.core.Frame:
    return molrs.core.Frame(
        {
            "atoms": {"x": np.arange(6, dtype=np.float64)},
            "cmaps": {
                key: np.array([i, i + 1], dtype=np.uint64)
                for i, key in enumerate(["atomi", "atomj", "atomk", "atoml", "atomm"])
            }
            | {"type": np.array(["c1", "c2"])},
        }
    )


def test_a_cmaps_block_renumbers_atomi_through_atomm(tmp_path: Path) -> None:
    assert [str(key) for key in molrs.core.keys.ENDPOINTS][-1] == "atomm"
    assert str(molrs.core.keys.ATOMM) == "atomm"
    assert molrs.core.schema.CMAPS == "cmaps"
    assert molrs.core.schema.relation_endpoints("cmaps", [])[-1] == ("atomm", "atoms")
    frame = _two_cmaps()
    frame.validate()

    out = frame.subset([1, 2, 3, 4, 5])
    assert list(out["cmaps"]["atomm"]) == [4]
    assert list(out["cmaps"]["type"]) == ["c2"]
    two = frame.replicate(2)
    assert list(two["cmaps"]["atomm"]) == [4, 5, 10, 11]

    path = tmp_path / "cmaps.mrec"
    molrs.io.write_mrec(path, frame=frame)
    back = molrs.io.mrec.MrecFile(path).frame()
    assert list(back["cmaps"]["atomm"]) == [4, 5]


# ── CMAP crossterms: assign_cmaps, the kernel, LAMMPS fix cmap I/O ──────────

ALANINE = (
    Path(__file__).resolve().parents[2]
    / "molrs/src/ff/potential/cmap/testdata/charmm36_alanine.cmap"
)
ALA = ("C", "NH1", "CT1", "C", "NH1")

# Five backbone atoms, φ ≈ −60°, ψ ≈ −40°.
BACKBONE = np.array(
    [
        [0.3, 1.4, 0.2],
        [0.0, 0.0, 0.0],
        [1.46, 0.0, 0.0],
        [2.0, -0.8, 1.2],
        [3.3, -0.9, 1.3],
    ]
)


def _number_lines(text: str) -> list[str]:
    return [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _alanine_ff() -> molrs.ff.forcefield.ForceField:
    (row,) = (
        molrs.io.lammps.LammpsForcefieldReader().read_cmap(ALANINE).get_types("cmap")
    )
    ff = molrs.ff.forcefield.ForceField("charmm", units="real")
    atoms = ff.def_style("atom", "full")
    by_name = {name: atoms.def_type(name, mass=12.0) for name in set(ALA)}
    ff.def_style("cmap", "charmm").def_type(
        "ala", *(by_name[name] for name in ALA), grid=row["grid"]
    )
    return ff


def _backbone() -> molrs.core.Frame:
    frame = molrs.core.Frame()
    atoms = molrs.core.Block()
    for k, key in enumerate("xyz"):
        atoms.insert(key, BACKBONE[:, k].copy())
    atoms.insert("type", list(ALA))
    atoms.insert("mol_id", np.ones(5, dtype=np.uint64))
    atoms.insert("charge", np.zeros(5))
    frame["atoms"] = atoms
    dihedrals = molrs.core.Block()
    for p, key in enumerate(("atomi", "atomj", "atomk", "atoml")):
        dihedrals.insert(key, np.array([p, p + 1], dtype=np.uint64))
    frame["dihedrals"] = dihedrals
    return frame


def test_read_lammps_cmap_names_rows_by_crossterm_type() -> None:
    ff = molrs.io.lammps.LammpsForcefieldReader().read_cmap(ALANINE)
    assert ff.units == "real"
    (row,) = ff.get_types("cmap")
    assert row.name == "1"
    assert row["grid"].shape == (24, 24)
    assert row["grid"][0, 0] == 0.12679


def test_assign_cmaps_builds_the_block_the_kernel_prices() -> None:
    ff, frame = _alanine_ff(), _backbone()
    assert molrs.ff.typifier.assign_cmaps(frame, ff) == 1
    cmaps = frame["cmaps"]
    assert [int(cmaps[k][0]) for k in ("atomi", "atomm")] == [0, 4]
    assert list(cmaps["type"]) == ["ala"]
    compiler = molrs.ff.compile.PotentialCompiler(ff)
    energy, forces = compiler.compile(frame).evaluate(frame)
    assert np.isfinite(energy) and energy != 0.0
    np.testing.assert_allclose(forces.reshape(-1, 3).sum(axis=0), 0.0, atol=1e-10)


def test_lammps_fix_cmap_files_round_trip(tmp_path: Path) -> None:
    ff, frame = _alanine_ff(), _backbone()
    molrs.ff.typifier.assign_cmaps(frame, ff)
    del frame["dihedrals"]

    cmap = tmp_path / "charmm.cmap"
    molrs.io.lammps.LammpsForcefieldWriter(frame).write_cmap(cmap, ff)
    assert _number_lines(cmap.read_text(encoding="utf-8")) == _number_lines(
        ALANINE.read_text(encoding="utf-8")
    )

    include = _lammps_include(ff, frame, skip_pair_style=True, cmap_file="charmm.cmap")
    assert "fix cmap all cmap charmm.cmap\nfix_modify cmap energy yes\n" in include
    with pytest.raises(ValueError, match="cmap_file"):
        _lammps_include(ff, frame, skip_pair_style=True)

    # A data file states a mass per atom type and the writer derives none:
    # the CHARMM masses of C / NH1 / CT1, in atom order.
    atoms = frame["atoms"]
    atoms.insert("mass", np.array([12.011, 14.007, 12.011, 12.011, 14.007]))
    frame["atoms"] = atoms
    data = tmp_path / "data.lmp"
    molrs.io.write_lammps_data(data, frame)
    assert "\n1 crossterms\n" in data.read_text(encoding="utf-8")
    back = molrs.io.read_lammps_data(data)
    assert [int(back["cmaps"][k][0]) for k in ("atomi", "atomm", "type_id")] == [
        0,
        4,
        1,
    ]
