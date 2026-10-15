"""The in-memory doors of ``molrs.io`` are their path doors on memory.

``write_<fmt>_str`` / ``_bytes`` return the file the path writer writes, and
``read_<fmt>_str`` / ``_bytes`` read it back to what the path reader reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import molrs.io as mio

XYZ = (
    '2\nLattice="10 0 0 0 11 0 0 0 12" Properties=species:S:1:pos:R:3:type:S:1 '
    'pbc="T T T"\nO 1.0 2.0 3.0 OW\nH 1.5 2.0 3.0 HW\n'
)

FIXTURES = Path(__file__).resolve().parents[2] / "molrs/src/ff/testdata/openmm"

XSF = "CRYSTAL\nPRIMVEC\n 10 0 0\n 0 11 0\n 0 0 12\nPRIMCOORD\n 2 1\n 8 1.0 2.0 3.0\n 1 1.5 2.0 3.0\n"


@pytest.fixture
def frame():
    return mio.read_xyz_str(XYZ)


def _n(frame) -> int:
    return frame["atoms"].n_rows


TEXT_FORMATS = [
    ("pdb", mio.write_pdb, mio.write_pdb_str, mio.read_pdb, mio.read_pdb_str),
    ("xyz", mio.write_xyz, mio.write_xyz_str, mio.read_xyz, mio.read_xyz_str),
    ("gro", mio.write_gro, mio.write_gro_str, mio.read_gro, mio.read_gro_str),
    ("mol2", mio.write_mol2, mio.write_mol2_str, mio.read_mol2, mio.read_mol2_str),
    ("cif", mio.write_cif, mio.write_cif_str, mio.read_cif, mio.read_cif_str),
    (
        "data",
        mio.write_lammps_data,
        mio.write_lammps_data_str,
        mio.read_lammps_data,
        mio.read_lammps_data_str,
    ),
    (
        "poscar",
        mio.write_poscar,
        mio.write_poscar_str,
        mio.read_poscar,
        mio.read_poscar_str,
    ),
]


@pytest.mark.parametrize(
    ("ext", "write_path", "write_str", "read_path", "read_str"),
    TEXT_FORMATS,
    ids=[f[0] for f in TEXT_FORMATS],
)
def test_text_doors_are_the_path_doors_in_memory(
    tmp_path, frame, ext, write_path, write_str, read_path, read_str
):
    path = tmp_path / f"a.{ext}"
    write_path(path, frame)
    text = write_str(frame)
    assert path.read_text(encoding="utf-8") == text
    assert _n(read_str(text)) == _n(read_path(path)) == 2


def test_xsf_text_doors(tmp_path):
    frame = mio.read_xsf_str(XSF)
    assert _n(frame) == 2
    path = tmp_path / "a.xsf"
    mio.write_xsf(path, frame)
    assert path.read_text(encoding="utf-8") == mio.write_xsf_str(frame)


def test_lammps_dump_str_is_one_snapshot_of_the_trajectory(tmp_path, frame):
    path = tmp_path / "a.dump"
    mio.write_lammps_dump_trajectory(path, [frame], columns=["type", "x", "y", "z"])
    text = mio.write_lammps_dump_str(frame, columns=["type", "x", "y", "z"])
    assert path.read_text(encoding="utf-8") == text
    assert _n(mio.read_lammps_dump_str(text)) == 2
    assert _n(mio.read_lammps_dump_bytes(text.encode())) == 2


def test_bytes_window_readers_read_the_text(frame):
    assert _n(mio.read_pdb_bytes(mio.write_pdb_str(frame).encode())) == 2
    assert _n(mio.read_xyz_bytes(mio.write_xyz_str(frame).encode())) == 2
    data = mio.write_lammps_data_str(frame)
    assert _n(mio.read_lammps_data_bytes(data.encode())) == 2


@pytest.mark.parametrize("fmt", ["dcd", "trr", "xtc"])
def test_binary_bytes_doors_are_one_frame_files(tmp_path, frame, fmt):
    write_bytes = getattr(mio, f"write_{fmt}_bytes")
    read_bytes = getattr(mio, f"read_{fmt}_bytes")
    write_path = getattr(mio, f"write_{fmt}_trajectory")
    path = tmp_path / f"a.{fmt}"
    write_path(path, [frame])
    data = write_bytes(frame)
    assert isinstance(data, bytes)
    assert path.read_bytes() == data
    assert _n(read_bytes(data)) == 2


def test_forcefield_text_doors_are_the_path_doors_in_memory(tmp_path):
    source = FIXTURES / "amber.xml"
    ff = mio.read_openmm_forcefield_str(source.read_text(encoding="utf-8"))
    from_path = mio.read_openmm_forcefield(source)
    xml = mio.write_openmm_forcefield_str(ff)
    assert xml == mio.write_openmm_forcefield_str(from_path)
    path = tmp_path / "ff.xml"
    mio.write_openmm_forcefield(path, ff)
    assert path.read_text(encoding="utf-8") == xml
    # Its long atom types do not fit frcmod: both frcmod doors refuse alike.
    with pytest.raises(ValueError, match="two-character") as by_path:
        mio.write_frcmod(tmp_path / "ff.frcmod", ff)
    with pytest.raises(ValueError) as by_str:
        mio.write_frcmod_str(ff)
    assert str(by_str.value) == str(by_path.value)


def test_the_force_field_xml_holds_what_openmm_xml_read(tmp_path):
    source = FIXTURES / "amber.xml"
    ff = mio.read_openmm_forcefield(source)
    xml = mio.write_xml_forcefield_str(ff)
    path = tmp_path / "ff.xml"
    mio.write_xml_forcefield(path, ff)
    assert path.read_text(encoding="utf-8") == xml
    back = mio.read_xml_forcefield(path)
    assert mio.write_xml_forcefield_str(back) == xml
    assert mio.write_xml_forcefield_str(mio.read_xml_forcefield_str(xml)) == xml
    # An OpenMM document is not this one: the reader names the door that reads it.
    with pytest.raises(ValueError, match="read_openmm_forcefield"):
        mio.read_xml_forcefield(source)
