"""OpenMM force-field XML across the FFI seam, and its 1-4 pairs compiled.

The fixtures are the Rust check's (``crates/molrs/src/ff/testdata/openmm``): XML
cut from CHARMM36, AMBER ff14SB and OPLS-AA, whose energies the Rust tests
hold to OpenMM's and LAMMPS's.
"""

from __future__ import annotations

from pathlib import Path

import molpy as mp
import numpy as np
import pytest
from molpy.io.openmm_xml import OpenmmXmlReader, OpenmmXmlWriter

FIXTURES = Path(__file__).resolve().parents[3] / "crates/molrs/src/ff/testdata/openmm"


def test_a_charmm_port_reads_every_section() -> None:
    ff = OpenmmXmlReader().read(FIXTURES / "charmm.xml")
    lj = ff.get_style("pair", "lj/charmm")
    assert lj.params["one_four"] == "epsilon14"
    assert lj.params["mixing"] == "arithmetic"
    assert ff.get_style("pair", "coul/charmm") is not None
    assert ff.get_style("angle", "charmm") is not None
    assert ff.get_style("improper", "harmonic") is not None
    (cmap,) = ff.get_types("cmap")
    assert cmap.params["grid"].shape == (24, 24)
    assert ff.special_bonds == ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])


@pytest.mark.parametrize("case", ["charmm", "amber", "opls"])
def test_written_xml_reads_back_with_the_same_types(case: str, tmp_path: Path) -> None:
    ff = OpenmmXmlReader().read(FIXTURES / f"{case}.xml")
    path = tmp_path / f"{case}.xml"
    OpenmmXmlWriter().write(path, ff)
    back = OpenmmXmlReader().read(path)
    for category in ("bond", "angle", "dihedral", "improper", "pair", "cmap"):
        names = sorted(t.name for t in ff.get_types(category))
        assert sorted(t.name for t in back.get_types(category)) == names, category


def test_xml_written_into_a_buffer_reads_back_as_text(tmp_path: Path) -> None:
    ff = OpenmmXmlReader().read(FIXTURES / "amber.xml")
    path = tmp_path / "amber.xml"
    OpenmmXmlWriter().write(path, ff)
    buf = bytearray()
    OpenmmXmlWriter().write(buf, ff)
    assert bytes(buf) == path.read_bytes()
    back = OpenmmXmlReader().read(mp.io.Text(buf.decode()))
    for category in ("bond", "angle", "dihedral", "improper", "pair"):
        names = sorted(t.name for t in ff.get_types(category))
        assert sorted(t.name for t in back.get_types(category)) == names, category


def _chain() -> mp.core.Frame:
    """ACE's CH3-C-N-CA (CT3 C NH1 CT1), bonded in a row: (0, 3) is the one
    1-4 pair, typed with the CHARMM fixture's rows."""
    frame = mp.core.Frame()
    atoms = mp.core.Block()
    x = np.array([[0.0, 0.0, 0.0], [1.5, 0.2, 0.0], [2.2, 1.5, 0.3], [3.6, 1.9, 0.1]])
    for k, key in enumerate("xyz"):
        atoms.insert(key, x[:, k].copy())
    atoms.insert("type", ["CT3", "C", "NH1", "CT1"])
    atoms.insert("atomic_number", np.array([6, 6, 7, 6], dtype=np.uint64))
    atoms.insert("charge", np.array([0.2, -0.2, 0.2, -0.2]))
    frame["atoms"] = atoms
    for name, rows, labels in (
        ("bonds", [[0, 1], [1, 2], [2, 3]], ["CT3-C", "NH1-C", "NH1-CT1"]),
        ("angles", [[0, 1, 2], [1, 2, 3]], ["NH1-C-CT3", "CT1-NH1-C"]),
        ("dihedrals", [[0, 1, 2, 3]], ["CT3-C-NH1-CT1"]),
    ):
        block = mp.core.Block()
        for k, key in enumerate(("atomi", "atomj", "atomk", "atoml")[: len(rows[0])]):
            block.insert(key, np.array([r[k] for r in rows], dtype=np.uint64))
        block.insert("type", labels)
        frame[name] = block
    return frame


def _lj_charmm_14(ff: mp.ff.forcefield.ForceField, frame: mp.core.Frame) -> float:
    """The CHARMM fixture's 1-4 pricing of ``frame``: everything but the
    Lennard-Jones and Coulomb pair styles' pricing, with no switch."""
    for name in ("lj/charmm", "coul/charmm"):
        style = ff.get_style("pair", name)
        style.set(inner=900.0)
        style.set(cutoff=1000.0)
    return mp.ff.PotentialCompiler(ff).compile(frame).evaluate(frame)[0]


def test_the_one_four_pair_is_priced_at_epsilon14_once_generated() -> None:
    ff = OpenmmXmlReader().read(FIXTURES / "charmm.xml")
    frame = _chain()
    with pytest.raises(ValueError, match="generate_topology"):
        _lj_charmm_14(ff, frame)
    graph = mp.core.Atomistic.from_frame(frame)
    graph.generate_topology(gen_angle=False, gen_dihedral=False)
    typed = graph.to_frame()
    pairs = typed["pairs"]
    rows = list(zip(pairs["atomi"], pairs["atomj"], pairs["n_bonds"], strict=True))
    assert [tuple(int(v) for v in r) for r in rows] == [
        (0, 1, 1),
        (0, 2, 2),
        (0, 3, 3),
        (1, 2, 1),
        (1, 3, 2),
        (2, 3, 1),
    ]
    assert np.isfinite(_lj_charmm_14(ff, typed))
