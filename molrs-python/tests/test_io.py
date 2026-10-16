"""FFI smoke tests for top-level IO readers/writers.

Self-contained: every fixture is written by molrs itself. No external corpus,
no third-party scientific software.
"""

from __future__ import annotations

from pathlib import Path

import molrs
import numpy as np
import pytest


class TestErrorMessages:
    def test_pyo3_type_error_names_the_argument(self):
        with pytest.raises(TypeError, match="center"):
            molrs.core.Sphere("not-an-array", 1.0)


class TestReadPdb:
    def test_basic(self, water_pdb):
        frame = molrs.io.read_pdb(water_pdb)
        assert "atoms" in frame
        assert frame["atoms"].n_rows == 3

    def test_has_coordinates(self, water_pdb):
        frame = molrs.io.read_pdb(water_pdb)
        atoms = frame["atoms"]
        assert atoms["x"] is not None
        assert atoms["y"] is not None
        assert atoms["z"] is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.read_pdb(Path("/nonexistent/path.pdb"))

    def test_missing_file_names_the_path(self):
        with pytest.raises(OSError, match="missing.pdb"):
            molrs.io.read_pdb(Path("missing.pdb"))


class TestReadGro:
    def test_native_basic(self, water_gro):
        frames = molrs.io.gro.GroReader(water_gro)
        assert len(frames) == 1
        f0 = frames[0]
        assert "atoms" in f0
        assert f0["atoms"].n_rows == 3
        assert f0.box is not None

    def test_native_columns(self, water_gro):
        frames = molrs.io.gro.GroReader(water_gro)
        atoms = frames[0]["atoms"]
        # The reader emits the canonical column names (`res_id`, `id`), not
        # the format-native spellings `resid` / `atom_id`.
        for col in ["res_id", "res_name", "name", "id", "x", "y", "z"]:
            assert col in atoms, f"missing column: {col}"

    def test_facade_canonical_columns(self, water_gro):
        atoms = molrs.io.read_gro(water_gro)["atoms"]
        for col in ["res_id", "res_name", "name", "id", "x", "y", "z"]:
            assert col in atoms, f"missing canonical column: {col}"

    def test_facade_no_format_native_columns(self, water_gro):
        atoms = molrs.io.read_gro(water_gro)["atoms"]
        for col in ["resid", "atom_name", "atom_id"]:
            assert col not in atoms, f"format-native column leaked: {col}"

    def test_round_trip(self, water_gro, tmp_path):
        f0 = molrs.io.read_gro(water_gro)
        out = tmp_path / "out.gro"
        molrs.io.write_gro(out, f0)
        # A write must not rename the caller's columns.
        assert "res_name" in f0["atoms"] and "resname" not in f0["atoms"]
        f1 = molrs.io.read_gro(out)
        assert f0["atoms"].n_rows == f1["atoms"].n_rows
        assert list(f1["atoms"]["res_id"]) == list(f0["atoms"]["res_id"])
        assert list(f1["atoms"]["id"]) == list(f0["atoms"]["id"])

    def test_trajectory_round_trip(self, water_gro, tmp_path):
        f0 = molrs.io.read_gro(water_gro)
        out = tmp_path / "traj.gro"
        with molrs.io.gro.GroWriter(out) as writer:
            writer.write(f0)
            writer.write(f0)
        frames = molrs.io.gro.GroReader(out)
        assert len(frames) == 2
        assert frames[1]["atoms"].n_rows == f0["atoms"].n_rows

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.gro.GroReader(Path("/nonexistent/path.gro"))


class TestReadXyz:
    def test_basic(self, water_xyz):
        frame = molrs.io.read_xyz(water_xyz)
        assert "atoms" in frame
        assert frame["atoms"].n_rows == 3

    def test_has_coordinates(self, water_xyz):
        frame = molrs.io.read_xyz(water_xyz)
        atoms = frame["atoms"]
        assert atoms["x"] is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.read_xyz(Path("/nonexistent/path.xyz"))

    def test_trajectory_round_trip(self, water_xyz, tmp_path):
        frame = molrs.io.read_xyz(water_xyz)
        out = tmp_path / "traj.xyz"
        with molrs.io.xyz.XyzWriter(out) as writer:
            for _ in range(3):
                writer.write(frame)
        with molrs.io.xyz.XyzReader(out) as reader:
            assert len(reader) == 3
            assert reader[2]["atoms"].n_rows == frame["atoms"].n_rows


def test_every_frame_reader_class_has_its_writer() -> None:
    """Names pair: a format module's ``<Fmt>Reader`` has a ``<Fmt>Writer``,
    except the formats the core only reads (SDF, Gaussian cube)."""
    pairs = {
        "pdb": "Pdb",
        "xyz": "Xyz",
        "gro": "Gro",
        "mol2": "Mol2",
        "cif": "Cif",
        "lammps": "LammpsDump",
        "dcd": "Dcd",
        "trr": "Trr",
        "xtc": "Xtc",
    }
    for module, stem in pairs.items():
        names = getattr(molrs.io, module).__all__
        assert f"{stem}Reader" in names and f"{stem}Writer" in names, module
    assert "SdfReader" in molrs.io.sdf.__all__
    assert "SdfWriter" not in molrs.io.sdf.__all__
    assert molrs.io.cube.__all__ == ["CubeReader"]


def test_read_stl_gives_a_watertight_mesh(tmp_path) -> None:
    import numpy as np

    # A closed tetrahedron in ASCII STL, written in-process.
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    lines = ["solid tet"]
    for f in faces:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for i in f:
            lines.append("      vertex {} {} {}".format(*verts[i]))
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid tet")
    path = tmp_path / "tet.stl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mesh = molrs.io.read_stl(path)
    assert isinstance(mesh, molrs.core.TriMesh)
    assert mesh.n_faces == 4 and mesh.n_vertices == 4
    assert mesh.is_watertight()
    tet = molrs.core.Polyhedron(mesh.scaled(2.0))
    assert tet.contains(np.array([[0.2, 0.2, 0.2]]))[0]
    assert not tet.contains(np.array([[3.0, 3.0, 3.0]]))[0]


class TestCsvBlock:
    """``molrs.io.read_csv_block`` / ``write_csv_block`` — CSV for one Block,
    read from a path, bytes or text and written to a path or a buffer."""

    def test_headered_round_trip(self):
        src = molrs.core.Block(
            {
                "x": [1.0, 2.0],
                "id": np.array([10, 20], dtype=np.uint32),
                "name": ["p", "q"],
            }
        )
        buf = bytearray()
        molrs.io.write_csv_block(buf, src)
        rt = molrs.io.read_csv_block(bytes(buf))
        np.testing.assert_allclose(rt["x"], [1.0, 2.0])
        np.testing.assert_array_equal(rt["id"], [10, 20])
        assert list(rt["name"]) == ["p", "q"]

    def test_dtype_inference(self):
        rt = molrs.io.read_csv_block(molrs.io.Text("a,b,c\n1,1.5,x\n2,2.5,y\n"))
        assert str(rt["a"].dtype).startswith("int")
        assert str(rt["b"].dtype).startswith("float")
        assert list(rt["c"]) == ["x", "y"]

    def test_headerless_with_names(self):
        rt = molrs.io.read_csv_block(molrs.io.Text("1,2\n3,4\n"), header=["a", "b"])
        np.testing.assert_array_equal(rt["a"], [1, 3])
        np.testing.assert_array_equal(rt["b"], [2, 4])

    def test_empty_csv_raises_value_error(self):
        with pytest.raises(ValueError):
            molrs.io.read_csv_block(molrs.io.Text(""))

    def test_no_header(self):
        b = molrs.core.Block({"count": np.array([1, 2], dtype=np.int64)})
        buf = bytearray()
        molrs.io.write_csv_block(buf, b, header=False)
        text = buf.decode()
        assert "count" not in text.splitlines()[0]

    def test_writes_and_reads_a_file(self, tmp_path):
        path = tmp_path / "out.csv"
        assert molrs.io.write_csv_block(path, molrs.core.Block({"x": [1.0, 2.0]})) is None
        np.testing.assert_allclose(molrs.io.read_csv_block(path)["x"], [1.0, 2.0])

    def test_a_one_character_delimiter_is_used(self):
        rt = molrs.io.read_csv_block(molrs.io.Text("a;b\n1;2\n"), delimiter=";")
        np.testing.assert_array_equal(rt["b"], [2])
        buf = bytearray()
        molrs.io.write_csv_block(buf, rt, delimiter=";")
        text = buf.decode()
        assert text.splitlines()[0] == "a;b"

    @pytest.mark.parametrize("delimiter", ["", "::", ", "])
    def test_a_delimiter_not_one_character_raises_value_error(self, delimiter):
        block = molrs.core.Block({"x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="one character"):
            molrs.io.read_csv_block(molrs.io.Text("a,b\n1,2\n"), delimiter=delimiter)
        with pytest.raises(ValueError, match="one character"):
            molrs.io.write_csv_block(bytearray(), block, delimiter=delimiter)


class TestCanonicalNativeColumns:
    """Every reader emits the canonical column names in Rust, so the readers
    are re-exported by identity: there is no Python renaming layer."""

    def test_readers_and_writers_are_the_compiled_functions(self):
        for name in (
            "read_pdb",
            "write_pdb",
            "read_gro",
            "write_gro",
            "read_mol2",
            "write_mol2",
            "read_xyz",
            "read_lammps_data",
            "read_lammps_molecule",
        ):
            assert getattr(molrs.io, name) is getattr(molrs._native, name), name

    def test_pdb_format_names_become_canonical(self, tmp_path):
        path = tmp_path / "w.pdb"
        path.write_text(
            "ATOM      1  OW  SOL A   7       0.000   0.000   0.000  1.00  0.00           O\n"
            "ATOM      2  HW1 SOL A   7       0.957   0.000   0.000  1.00  0.00           H\n"
            "END\n", encoding="utf-8"
        )
        atoms = molrs.io.read_pdb(path)["atoms"]
        assert list(atoms["element"]) == ["O", "H"]
        assert list(atoms["res_id"]) == [7, 7]
        assert list(atoms["res_name"]) == ["SOL", "SOL"]
        for native in ("symbol", "resname", "res_seq"):
            assert native not in atoms

    def test_lammps_dump_q_and_mol_become_charge_and_mol_id(self, tmp_path):
        path = tmp_path / "d.lammpstrj"
        path.write_text(
            "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id type mol q x y z\n"
            "1 1 1 -0.5 0 0 0\n2 1 1 0.5 1 0 0\n", encoding="utf-8"
        )
        atoms = molrs.io.lammps.LammpsDumpReader(path)[0]["atoms"]
        np.testing.assert_allclose(atoms["charge"], [-0.5, 0.5])
        assert list(atoms["mol_id"]) == [1, 1]
        for native in ("q", "mol"):
            assert native not in atoms

    def test_lammps_molecule_charges_become_charge(self, tmp_path):
        path = tmp_path / "m.mol"
        path.write_text(
            "two atoms\n\n2 atoms\n\nCoords\n\n1 0 0 0\n2 0 0 1\n\n"
            "Types\n\n1 1\n2 1\n\nCharges\n\n1 0.1\n2 -0.1\n", encoding="utf-8"
        )
        atoms = molrs.io.read_lammps_molecule(path)["atoms"]
        np.testing.assert_allclose(atoms["charge"], [0.1, -0.1])
        assert "q" not in atoms

    def test_mol2_columns_are_canonical_and_round_trip(self, tmp_path):
        path = tmp_path / "x.mol2"
        path.write_text(
            "@<TRIPOS>MOLECULE\nX\n2 1 2 0 0\nSMALL\nNO_CHARGES\n@<TRIPOS>ATOM\n"
            "1 C1 0.0 0.0 0.0 C.ar 7 BEN 0.0\n2 N1 1.4 0.0 0.0 N.am 9 AMD 0.0\n"
            "@<TRIPOS>BOND\n1 1 2 am\n", encoding="utf-8"
        )
        frame = molrs.io.read_mol2(path)
        atoms = frame["atoms"]
        assert list(atoms["type"]) == ["C.ar", "N.am"]
        assert list(atoms["res_id"]) == [7, 9]
        assert list(frame["bonds"]["type"]) == ["am"]
        out = tmp_path / "y.mol2"
        molrs.io.write_mol2(out, frame)
        back = molrs.io.read_mol2(out)
        assert list(back["atoms"]["res_name"]) == ["BEN", "AMD"]

    def test_extxyz_species_is_element_and_wide_properties_stay_whole(self, tmp_path):
        path = tmp_path / "f.xyz"
        path.write_text(
            "2\nProperties=species:S:1:pos:R:3:forces:R:3\n"
            "O 0 0 0 0.1 0.2 0.3\nH 1 0 0 -0.1 -0.2 -0.3\n", encoding="utf-8"
        )
        atoms = molrs.io.read_xyz(path)["atoms"]
        assert list(atoms["element"]) == ["O", "H"]
        assert atoms["forces"].shape == (2, 3)
        assert "forces_1" not in atoms


class TestLammpsDataLabels:
    LABELLED = (
        "test\n\n3 atoms\n1 bonds\n2 atom types\n1 bond types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Atom Type Labels\n\n1 OW\n2 HW\n\nBond Type Labels\n\n1 OW-HW\n\n"
        "Masses\n\n1 15.999\n2 1.008\n\n"
        "Atoms # full\n\n1 1 1 -0.8 0 0 0\n2 1 2 0.4 1 0 0\n3 1 2 0.4 0 1 0\n\n"
        "Bonds\n\n1 1 1 2\n"
    )

    def test_typed_blocks_carry_string_types(self, tmp_path):
        path = tmp_path / "w.data"
        path.write_text(self.LABELLED, encoding="utf-8")
        frame = molrs.io.read_lammps_data(path)
        assert list(frame["atoms"]["type"]) == ["OW", "HW", "HW"]
        assert list(frame["atoms"]["type_id"]) == [1, 2, 2]
        assert list(frame["bonds"]["type"]) == ["OW-HW"]

    def test_atom_style_sets_the_atoms_layout(self, tmp_path):
        path = tmp_path / "a.data"
        path.write_text(
            "t\n\n1 atoms\n1 atom types\n\n0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n"
            "Atoms # full\n\n1 1 0.1 0.2 0.3\n", encoding="utf-8"
        )
        atoms = molrs.io.read_lammps_data(path, atom_style="atomic")["atoms"]
        assert "charge" not in atoms and "mol_id" not in atoms
        with pytest.raises(OSError, match="atom style"):
            molrs.io.read_lammps_data(path, atom_style="nonsense")

    def test_write_declares_extra_type_labels(self, tmp_path):
        src = tmp_path / "w.data"
        src.write_text(self.LABELLED, encoding="utf-8")
        frame = molrs.io.read_lammps_data(src)
        out = tmp_path / "o.data"
        molrs.io.write_lammps_data(out, frame, type_labels={"bonds": ["OX-HW"]})
        text = out.read_text(encoding="utf-8")
        assert "2 bond types" in text
        assert "Bond Type Labels\n\n1 OW-HW\n2 OX-HW\n" in text
        # An atom type no row uses has no mass the frame states, and the
        # writer invents none: the error names the explicit route.
        with pytest.raises(OSError, match=r"atom type 3 \(OX\) has no mass"):
            molrs.io.write_lammps_data(out, frame, type_labels={"atoms": ["OX"]})
        # Stating it through the frame meta is that route.
        masses = dict(frame.meta["lammps_type_masses"])
        masses["OX"] = 15.9994
        frame.meta["lammps_type_masses"] = masses
        molrs.io.write_lammps_data(out, frame, type_labels={"atoms": ["OX"]})
        masses_section = out.read_text(encoding="utf-8").split("Masses")[1]
        assert "15.9994" in masses_section
        with pytest.raises(ValueError, match="empty"):
            molrs.io.write_lammps_data(out, frame, type_labels={"atoms": [""]})


class TestInpcrdIntoFrame:
    TEXT = (
        "two atoms\n  2\n"
        "  1.0000000   2.0000000   3.0000000   4.0000000   5.0000000   6.0000000\n"
    )

    def test_coordinates_go_into_the_given_frame(self, tmp_path):
        path = tmp_path / "x.inpcrd"
        path.write_text(self.TEXT, encoding="utf-8")
        frame = molrs.core.Frame()
        atoms = molrs.core.Block()
        atoms.insert("x", np.zeros(2))
        atoms.insert("charge", np.array([0.5, -0.5]))
        frame["atoms"] = atoms
        same = molrs.io.read_inpcrd(path, frame)
        assert same is frame
        np.testing.assert_allclose(frame["atoms"]["z"], [3.0, 6.0])
        np.testing.assert_allclose(frame["atoms"]["charge"], [0.5, -0.5])
        assert frame.meta["title"] == "two atoms"

    def test_a_count_mismatch_leaves_the_frame_alone(self, tmp_path):
        path = tmp_path / "x.inpcrd"
        path.write_text(self.TEXT, encoding="utf-8")
        frame = molrs.core.Frame()
        atoms = molrs.core.Block()
        atoms.insert("x", np.zeros(3))
        frame["atoms"] = atoms
        with pytest.raises(OSError, match="rows"):
            molrs.io.read_inpcrd(path, frame)
        assert "y" not in frame["atoms"]
