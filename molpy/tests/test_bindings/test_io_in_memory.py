"""Every ``mp.io`` door reads a path, bytes and text alike, and writes the
same bytes to a path as into a ``bytearray``.

A ``bytearray`` sink is cleared, then holds exactly the file the path writer
writes; ``bytes`` and :class:`mp.io.Text` sources read back what the path
reader reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import molpy.io as mio
from molpy.core import Element
from molpy.io.amber import AmberFrcmodWriter
from molpy.io.openmm_xml import OpenmmXmlReader, OpenmmXmlWriter

XYZ = (
    '2\nLattice="10 0 0 0 11 0 0 0 12" Properties=species:S:1:pos:R:3:type:S:1 '
    'pbc="T T T"\nO 1.0 2.0 3.0 OW\nH 1.5 2.0 3.0 HW\n'
)

FIXTURES = Path(__file__).resolve().parents[3] / "crates/molrs/src/ff/testdata/openmm"

XSF = "CRYSTAL\nPRIMVEC\n 10 0 0\n 0 11 0\n 0 0 12\nPRIMCOORD\n 2 1\n 8 1.0 2.0 3.0\n 1 1.5 2.0 3.0\n"


@pytest.fixture
def frame():
    frame = mio.read_xyz(mio.Text(XYZ))
    # GRO and MOL2 spell an atom by its ``name`` and a LAMMPS data file
    # states a ``mass`` per type; the writers read both and guess neither
    # from the ``element`` the XYZ reader keeps, so the frame states them.
    atoms = frame["atoms"]
    symbols = list(atoms["element"])
    atoms.insert("name", symbols)
    atoms.insert(
        "mass", np.array([Element.by_symbol(s).mass for s in symbols], dtype=np.float64)
    )
    frame["atoms"] = atoms
    return frame


def _n(frame) -> int:
    return frame["atoms"].n_rows


TEXT_FORMATS = [
    ("pdb", mio.write_pdb, mio.read_pdb),
    ("xyz", mio.write_xyz, mio.read_xyz),
    ("gro", mio.write_gro, mio.read_gro),
    ("mol2", mio.write_mol2, mio.read_mol2),
    ("cif", mio.write_cif, mio.read_cif),
    ("data", mio.write_lammps_data, mio.read_lammps_data),
    ("poscar", mio.write_poscar, mio.read_poscar),
]


@pytest.mark.parametrize(
    ("ext", "write", "read"),
    TEXT_FORMATS,
    ids=[f[0] for f in TEXT_FORMATS],
)
def test_a_buffer_holds_the_file_and_every_source_reads_it(
    tmp_path, frame, ext, write, read
):
    path = tmp_path / f"a.{ext}"
    write(path, frame)
    buf = bytearray(b"stale bytes")
    write(buf, frame)
    assert path.read_bytes() == bytes(buf)
    text = bytes(buf).decode("utf-8")
    assert _n(read(mio.Text(text))) == _n(read(bytes(buf))) == _n(read(path)) == 2


def test_xsf_doors_in_memory(tmp_path):
    frame = mio.read_xsf(mio.Text(XSF))
    assert _n(frame) == 2
    path = tmp_path / "a.xsf"
    mio.write_xsf(path, frame)
    buf = bytearray()
    mio.write_xsf(buf, frame)
    assert path.read_bytes() == bytes(buf)


def test_lammps_dump_door_is_one_snapshot_of_the_writer(tmp_path, frame):
    columns = ["type", "x", "y", "z"]
    path = tmp_path / "a.dump"
    with mio.lammps.LammpsDumpWriter(path) as writer:
        writer.set_columns(columns)
        writer.write(frame)
    buf = bytearray()
    mio.write_lammps_dump(buf, frame, columns=columns)
    assert path.read_bytes() == bytes(buf)
    assert _n(mio.read_lammps_dump(mio.Text(bytes(buf).decode()))) == 2
    assert _n(mio.read_lammps_dump(bytes(buf))) == 2


def test_bytes_sources_read_the_text(frame):
    for write, read in [
        (mio.write_pdb, mio.read_pdb),
        (mio.write_xyz, mio.read_xyz),
        (mio.write_lammps_data, mio.read_lammps_data),
    ]:
        buf = bytearray()
        write(buf, frame)
        assert _n(read(bytes(buf))) == 2


@pytest.mark.parametrize("fmt", ["dcd", "trr", "xtc"])
def test_binary_doors_are_one_frame_files(tmp_path, frame, fmt):
    write = getattr(mio, f"write_{fmt}")
    read = getattr(mio, f"read_{fmt}")
    module = getattr(mio, fmt)
    writer_class = getattr(module, f"{fmt.capitalize()}Writer")
    path = tmp_path / f"a.{fmt}"
    with writer_class(path) as writer:
        writer.write(frame)
    buf = bytearray()
    write(buf, frame)
    assert path.read_bytes() == bytes(buf)
    assert _n(read(bytes(buf))) == 2


def test_a_str_is_neither_a_source_nor_a_sink(frame):
    with pytest.raises(TypeError, match=r"pathlib\.Path\(s\).*molpy\.io\.Text\(s\)"):
        mio.read_xyz(XYZ)
    with pytest.raises(TypeError, match=r"pathlib\.Path\(s\)"):
        mio.write_xyz("out.xyz", frame)
    with pytest.raises(TypeError, match="bytearray"):
        mio.write_xyz(b"", frame)


def test_forcefield_doors_read_a_path_and_text_alike(tmp_path):
    source = FIXTURES / "amber.xml"
    reader = OpenmmXmlReader()
    writer = OpenmmXmlWriter()
    ff = reader.read(mio.Text(source.read_text(encoding="utf-8")))
    from_path = reader.read(source)
    xml = bytearray()
    writer.write(xml, ff)
    again = bytearray()
    writer.write(again, from_path)
    assert xml == again
    path = tmp_path / "ff.xml"
    writer.write(path, ff)
    assert path.read_bytes() == bytes(xml)
    # Its long atom types do not fit frcmod: both sinks refuse alike.
    with pytest.raises(ValueError, match="two-character") as by_path:
        AmberFrcmodWriter().write(tmp_path / "ff.frcmod", ff)
    with pytest.raises(ValueError) as by_buffer:
        AmberFrcmodWriter().write(bytearray(), ff)
    assert str(by_buffer.value) == str(by_path.value)


def test_the_force_field_xml_holds_what_openmm_xml_read(tmp_path):
    source = FIXTURES / "amber.xml"
    ff = OpenmmXmlReader().read(source)
    reader = mio.xml.XmlForcefieldReader()
    writer = mio.xml.XmlForcefieldWriter()
    xml = bytearray()
    writer.write(xml, ff)
    path = tmp_path / "ff.xml"
    writer.write(path, ff)
    assert path.read_bytes() == bytes(xml)
    back = reader.read(path)
    again = bytearray()
    writer.write(again, back)
    assert again == xml
    writer.write(again, reader.read(bytes(xml)))
    assert again == xml
    # An OpenMM document is not this one: the reader names the reader of it.
    with pytest.raises(ValueError, match="OpenmmXmlReader"):
        reader.read(source)
