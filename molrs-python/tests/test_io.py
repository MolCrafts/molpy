"""FFI smoke tests for top-level IO readers/writers.

Self-contained: every fixture is written by molrs itself. No external corpus,
no third-party scientific software.
"""

from __future__ import annotations


import molrs
import numpy as np
import pytest


class TestErrorMessages:
    def test_pyo3_type_error_names_the_argument(self):
        with pytest.raises(TypeError, match="center"):
            molrs.core.Sphere("not-an-array", 1.0)


class TestReadPdb:
    def test_basic(self, water_pdb):
        frame = molrs.io.read_pdb(str(water_pdb))
        assert "atoms" in frame
        assert frame["atoms"].n_rows == 3

    def test_has_coordinates(self, water_pdb):
        frame = molrs.io.read_pdb(str(water_pdb))
        atoms = frame["atoms"]
        assert atoms["x"] is not None
        assert atoms["y"] is not None
        assert atoms["z"] is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.read_pdb("/nonexistent/path.pdb")

    def test_missing_file_names_the_path(self):
        with pytest.raises(OSError, match="missing.pdb"):
            molrs.io.read_pdb("missing.pdb")


class TestReadGro:
    def test_native_basic(self, water_gro):
        frames = molrs.io.read_gro_trajectory(str(water_gro))
        assert len(frames) == 1
        f0 = frames[0]
        assert "atoms" in f0
        assert f0["atoms"].n_rows == 3
        assert f0.box is not None

    def test_native_columns(self, water_gro):
        frames = molrs.io.read_gro_trajectory(str(water_gro))
        atoms = frames[0]["atoms"]
        # The reader emits the canonical column names (`res_id`, `id`), not
        # the format-native spellings `resid` / `atom_id`.
        for col in ["res_id", "res_name", "name", "id", "x", "y", "z"]:
            assert col in atoms, f"missing column: {col}"

    def test_facade_canonical_columns(self, water_gro):
        atoms = molrs.io.read_gro(str(water_gro))["atoms"]
        for col in ["res_id", "res_name", "name", "id", "x", "y", "z"]:
            assert col in atoms, f"missing canonical column: {col}"

    def test_facade_no_format_native_columns(self, water_gro):
        atoms = molrs.io.read_gro(str(water_gro))["atoms"]
        for col in ["resid", "atom_name", "atom_id"]:
            assert col not in atoms, f"format-native column leaked: {col}"

    def test_round_trip(self, water_gro, tmp_path):
        f0 = molrs.io.read_gro(str(water_gro))
        out = tmp_path / "out.gro"
        molrs.io.write_gro(out, f0)
        # A write must not rename the caller's columns.
        assert "res_name" in f0["atoms"] and "resname" not in f0["atoms"]
        f1 = molrs.io.read_gro(out)
        assert f0["atoms"].n_rows == f1["atoms"].n_rows
        assert list(f1["atoms"]["res_id"]) == list(f0["atoms"]["res_id"])
        assert list(f1["atoms"]["id"]) == list(f0["atoms"]["id"])

    def test_trajectory_round_trip(self, water_gro, tmp_path):
        f0 = molrs.io.read_gro(str(water_gro))
        out = tmp_path / "traj.gro"
        molrs.io.write_gro_trajectory(out, [f0, f0])
        frames = molrs.io.read_gro_trajectory(out)
        assert len(frames) == 2
        assert frames[1]["atoms"].n_rows == f0["atoms"].n_rows

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.read_gro_trajectory("/nonexistent/path.gro")


class TestReadXyz:
    def test_basic(self, water_xyz):
        frame = molrs.io.read_xyz(str(water_xyz))
        assert "atoms" in frame
        assert frame["atoms"].n_rows == 3

    def test_has_coordinates(self, water_xyz):
        frame = molrs.io.read_xyz(str(water_xyz))
        atoms = frame["atoms"]
        assert atoms["x"] is not None

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.read_xyz("/nonexistent/path.xyz")

    def test_trajectory_round_trip(self, water_xyz, tmp_path):
        frame = molrs.io.read_xyz(str(water_xyz))
        out = tmp_path / "traj.xyz"
        molrs.io.write_xyz_trajectory(out, [frame, frame, frame])
        with molrs.io.read_xyz_trajectory(out) as reader:
            assert reader.n_frames == 3
            assert reader[2]["atoms"].n_rows == frame["atoms"].n_rows


def test_every_trajectory_reader_has_its_writer() -> None:
    """Names pair: a ``read_X_trajectory`` door has a ``write_X_trajectory``."""
    public = set(molrs.io.__all__)
    for name in public:
        if name.startswith("read_") and name.endswith("_trajectory"):
            assert "write_" + name[len("read_") :] in public, name


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

    mesh = molrs.io.read_stl(str(path))
    assert isinstance(mesh, molrs.core.TriMesh)
    assert mesh.n_faces == 4 and mesh.n_vertices == 4
    assert mesh.is_watertight()
    tet = molrs.core.Polyhedron(mesh.scaled(2.0))
    assert tet.contains(np.array([[0.2, 0.2, 0.2]]))[0]
    assert not tet.contains(np.array([[3.0, 3.0, 3.0]]))[0]


class TestCsvBlock:
    """``molrs.io.read_csv_block[_str]`` / ``write_csv_block[_str]`` — CSV for
    one Block."""

    def test_headered_round_trip(self):
        src = molrs.core.Block(
            {
                "x": [1.0, 2.0],
                "id": np.array([10, 20], dtype=np.uint32),
                "name": ["p", "q"],
            }
        )
        rt = molrs.io.read_csv_block_str(molrs.io.write_csv_block_str(src))
        np.testing.assert_allclose(rt["x"], [1.0, 2.0])
        np.testing.assert_array_equal(rt["id"], [10, 20])
        assert list(rt["name"]) == ["p", "q"]

    def test_dtype_inference(self):
        rt = molrs.io.read_csv_block_str("a,b,c\n1,1.5,x\n2,2.5,y\n")
        assert str(rt["a"].dtype).startswith("int")
        assert str(rt["b"].dtype).startswith("float")
        assert list(rt["c"]) == ["x", "y"]

    def test_headerless_with_names(self):
        rt = molrs.io.read_csv_block_str("1,2\n3,4\n", header=["a", "b"])
        np.testing.assert_array_equal(rt["a"], [1, 3])
        np.testing.assert_array_equal(rt["b"], [2, 4])

    def test_empty_csv_raises_value_error(self):
        with pytest.raises(ValueError):
            molrs.io.read_csv_block_str("")

    def test_no_header(self):
        b = molrs.core.Block({"count": np.array([1, 2], dtype=np.int64)})
        text = molrs.io.write_csv_block_str(b, header=False)
        assert "count" not in text.splitlines()[0]

    def test_writes_and_reads_a_file(self, tmp_path):
        path = tmp_path / "out.csv"
        assert molrs.io.write_csv_block(path, molrs.core.Block({"x": [1.0, 2.0]})) is None
        np.testing.assert_allclose(molrs.io.read_csv_block(path)["x"], [1.0, 2.0])

    def test_a_one_character_delimiter_is_used(self):
        rt = molrs.io.read_csv_block_str("a;b\n1;2\n", delimiter=";")
        np.testing.assert_array_equal(rt["b"], [2])
        text = molrs.io.write_csv_block_str(rt, delimiter=";")
        assert text.splitlines()[0] == "a;b"

    @pytest.mark.parametrize("delimiter", ["", "::", ", "])
    def test_a_delimiter_not_one_character_raises_value_error(self, delimiter):
        block = molrs.core.Block({"x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="one character"):
            molrs.io.read_csv_block_str("a,b\n1,2\n", delimiter=delimiter)
        with pytest.raises(ValueError, match="one character"):
            molrs.io.write_csv_block_str(block, delimiter=delimiter)


class TestCanonicalNativeColumns:
    """Every reader emits the canonical column names in Rust, so the readers
    are re-exported by identity: there is no Python renaming layer."""

    def test_readers_and_writers_are_the_compiled_functions(self):
        for name in (
            "read_pdb",
            "write_pdb",
            "read_pdb_trajectory",
            "read_gro",
            "write_gro",
            "read_gro_trajectory",
            "write_gro_trajectory",
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
        atoms = molrs.io.read_lammps_dump_trajectory(path).read_frame(0)["atoms"]
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

    def test_mol2_bond_orders_survive_without_the_type_column(self, tmp_path):
        path = tmp_path / "x.mol2"
        path.write_text(
            "@<TRIPOS>MOLECULE\nX\n3 2 1 0 0\nSMALL\nNO_CHARGES\n@<TRIPOS>ATOM\n"
            "1 C1 0.0 0.0 0.0 C.ar 1 X\n2 C2 1.4 0.0 0.0 C.2 1 X\n"
            "3 O1 2.6 0.0 0.0 O.2 1 X\n@<TRIPOS>BOND\n1 1 2 ar\n2 2 3 2\n",
            encoding="utf-8",
        )
        frame = molrs.io.read_mol2(path)
        del frame["bonds"]["type"]
        back = molrs.io.read_mol2_str(molrs.io.write_mol2_str(frame))
        assert list(back["bonds"]["type"]) == ["ar", "2"]
        assert list(back["bonds"]["bond_type"]) == [4, 2]

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
        molrs.io.write_lammps_data(out, frame, type_labels={"atoms": ["OX"]})
        text = out.read_text(encoding="utf-8")
        assert "3 atom types" in text
        assert "Atom Type Labels\n\n1 HW\n2 OW\n3 OX\n" in text
        with pytest.raises(ValueError, match="empty"):
            molrs.io.write_lammps_data(out, frame, type_labels={"atoms": [""]})


class TestLammpsDataStyleAndUnits:
    """The data file owns ``atom_style`` and ``units``; the script repeats them."""

    FULL = (
        "t\n\n2 atoms\n1 bonds\n1 atom types\n1 bond types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\nMasses\n\n1 12.0\n\n"
        "Atoms # full\n\n1 1 1 -0.5 1 2 3\n2 1 1 0.5 4 5 6\n\n"
        "Velocities\n\n1 0.1 0.2 0.3\n2 -0.1 0 0.5\n\nBonds\n\n1 1 1 2\n"
    )
    LJ_FLUID = (
        "t\n\n2 atoms\n1 atom types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\nMasses\n\n1 39.948\n\n"
        "Atoms # atomic\n\n1 1 1 2 3\n2 1 4 5 6\n"
    )

    def test_inference_is_public_and_matches_the_default_write(self):
        for text, style in [(self.FULL, "full"), (self.LJ_FLUID, "atomic")]:
            frame = molrs.io.read_lammps_data_str(text)
            assert molrs.io.lammps.infer_lammps_atom_style(frame) == style
            out = molrs.io.write_lammps_data_str(frame)
            assert f"Atoms # {style}\n" in out
            # Naming the inferred style, or nothing, is the old output.
            assert molrs.io.write_lammps_data_str(frame, atom_style=style) == out
            assert molrs.io.write_lammps_data_str(frame, atom_style=None, units=None) == out

    def test_a_named_style_missing_columns_raises(self, tmp_path):
        lj = molrs.io.read_lammps_data_str(self.LJ_FLUID)
        with pytest.raises(OSError, match=r"mol_id.*charge|charge.*mol_id"):
            molrs.io.write_lammps_data(tmp_path / "lj.data", lj, atom_style="full")
        with pytest.raises(OSError, match="frame\\['bonds'\\]"):
            molrs.io.write_lammps_data_str(
                molrs.io.read_lammps_data_str(self.FULL), atom_style="charge"
            )

    def test_a_named_style_is_written_exactly(self, tmp_path):
        frame = molrs.io.read_lammps_data_str(self.FULL)
        out = tmp_path / "m.data"
        molrs.io.write_lammps_data(out, frame, atom_style="molecular")
        text = out.read_text(encoding="utf-8")
        assert "Atoms # molecular\n\n1 1 1 1 2 3\n2 1 1 4 5 6\n" in text

    def test_units_convert_and_round_trip_through_the_reader(self):
        frame = molrs.io.read_lammps_data_str(self.FULL)
        text = molrs.io.write_lammps_data_str(frame, units="metal")
        assert text.startswith("# LAMMPS data file generated by molrs, units = metal\n")
        metal = molrs.io.read_lammps_data_str(text)
        assert metal.meta["lammps_units"] == "metal"
        # Å/fs -> Å/ps; Å, g/mol and e are metal's too.
        np.testing.assert_allclose(metal["atoms"]["vx"], [100.0, -100.0])
        np.testing.assert_allclose(metal["atoms"]["x"], frame["atoms"]["x"])
        lj = molrs.io.read_lammps_data_str(
            molrs.io.write_lammps_data_str(frame, units="lj")
        )
        # One e at the canonical sigma = 1 Å, epsilon = 1 kcal/mol.
        np.testing.assert_allclose(lj["atoms"]["charge"], [-9.1112, 9.1112], rtol=1e-4)
        for there in (metal, lj):
            home = molrs.io.read_lammps_data_str(
                molrs.io.write_lammps_data_str(there, units="real")
            )
            for key in ("x", "charge", "vx", "vz"):
                np.testing.assert_allclose(home["atoms"][key], frame["atoms"][key])

    def test_an_unsupported_unit_style_is_a_value_error(self, tmp_path):
        frame = molrs.io.read_lammps_data_str(self.FULL)
        for units in ("nano", "si"):
            with pytest.raises(ValueError, match=units):
                molrs.io.write_lammps_data(tmp_path / "x.data", frame, units=units)
            with pytest.raises(ValueError, match=units):
                molrs.io.write_lammps_data_str(frame, units=units)


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


P21C_CIF = """\
data_p21c
_space_group_name_H-M_alt 'P 21/c'
_cell_length_a 5.0
_cell_length_b 6.0
_cell_length_c 7.0
_cell_angle_alpha 90.0
_cell_angle_beta 100.0
_cell_angle_gamma 90.0
loop_
_space_group_symop_id
_space_group_symop_operation_xyz
1 'x, y, z'
2 '-x, y+1/2, -z+1/2'
3 '-x, -y, -z'
4 'x, -y+1/2, z+1/2'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.2 0.3
"""
P21C_OPS = ("x, y, z", "-x, y+1/2, -z+1/2", "-x, -y, -z", "x, -y+1/2, z+1/2")


class TestCrystalSymmetryMeta:
    """``read_cif`` keeps the symmetry operators and the space group; nothing
    is expanded."""

    def test_meta_keys_are_projected(self):
        assert molrs.core.keys.SYMOPS == "symops"
        assert molrs.core.keys.SPACE_GROUP == "space_group"

    def test_read_cif_keeps_symops_as_a_sequence_of_str(self, tmp_path):
        path = tmp_path / "p21c.cif"
        path.write_text(P21C_CIF, encoding="utf-8")
        frame = molrs.io.read_cif(path)
        symops = frame.meta["symops"]
        assert isinstance(symops, tuple)
        assert all(isinstance(op, str) for op in symops)
        assert symops == P21C_OPS
        assert frame.meta["space_group"] == "P 21/c"
        assert frame["atoms"].n_rows == 1

    def test_symmetry_round_trips_through_write_cif(self, tmp_path):
        frame = molrs.io.read_cif_str(P21C_CIF)
        out = tmp_path / "back.cif"
        molrs.io.write_cif(out, frame)
        back = molrs.io.read_cif(out)
        assert back.meta["symops"] == P21C_OPS
        assert back.meta["space_group"] == "P 21/c"

    def test_symops_set_from_python_are_written(self):
        frame = molrs.io.read_cif_str(P21C_CIF)
        frame.meta["symops"] = ["x,y,z", "-x,-y,-z"]
        frame.meta["space_group"] = "P -1"
        back = molrs.io.read_cif_str(molrs.io.write_cif_str(frame))
        assert back.meta["symops"] == ("x,y,z", "-x,-y,-z")
        assert back.meta["space_group"] == "P -1"

    def test_malformed_symops_are_refused_on_write(self):
        frame = molrs.io.read_cif_str(P21C_CIF)
        frame.meta["symops"] = "x,y,z"
        with pytest.raises(OSError, match="symops"):
            molrs.io.write_cif_str(frame)

    def test_read_pdb_keeps_the_cryst1_space_group(self):
        frame = molrs.io.read_pdb_str(
            "CRYST1   52.000   58.600   61.900  90.00  90.00  90.00 P 21 21 21    8\n"
            "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 20.00           N  \n"
            "END\n"
        )
        assert frame.meta["space_group"] == "P 21 21 21"


TRICLINIC_PDB = (
    "CRYST1   10.000   11.000   12.000  70.00  80.00  65.00 P 1           1\n"
    "ATOM      1  O   HOH A   1       1.000   2.000   3.000  1.00  0.00           O  \n"
    "END\n"
)


def test_read_pdb_builds_the_triclinic_cell_cryst1_states():
    frame = molrs.io.read_pdb_str(TRICLINIC_PDB)
    np.testing.assert_allclose(frame.box.lengths, [10.0, 11.0, 12.0])
    np.testing.assert_allclose(frame.box.angles, [70.0, 80.0, 65.0])
    assert frame.box.style == "triclinic"
    assert "70.00  80.00  65.00" in molrs.io.write_pdb_str(frame)
    back = molrs.io.read_pdb_str(molrs.io.write_pdb_str(frame))
    assert back.box.approx_eq(frame.box, 1e-9)


def test_a_frame_without_a_box_writes_no_cryst1_and_reads_back_without_one():
    frame = molrs.io.read_pdb_str(TRICLINIC_PDB)
    frame.box = None
    text = molrs.io.write_pdb_str(frame)
    assert "CRYST1" not in text
    assert molrs.io.read_pdb_str(text).box is None
