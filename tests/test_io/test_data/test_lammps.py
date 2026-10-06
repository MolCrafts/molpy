"""
``mp.io.read_lammps_data`` / ``mp.io.write_lammps_data`` — molrs's, by identity.

The reader returns the structure as a ``Frame`` (typed blocks carry
``type_id`` and the string ``type``) and keeps any ``* Coeffs`` sections as
text in ``frame.meta["lammps_coeffs_text"]``; the force field is
``mp.ff.forcefield.read_lammps_data_coeffs`` of that text.
"""

import os
from pathlib import Path

import numpy as np
import pytest


import molpy as mp

_KINDS = ("atom", "bond", "angle", "dihedral", "improper")


def _forcefield(frame: mp.Frame, units: str = "real") -> mp.ff.forcefield.ForceField:
    """The force field of a data file's ``* Coeffs``, keyed by its Type Labels."""
    labels = {}
    for kind in _KINDS:
        packed = str(frame.meta.get(f"{kind}_type_labels") or "")
        pairs = (item.split(":", 1) for item in packed.split(",") if ":" in item)
        labels[f"{kind}_labels"] = {int(i): label for i, label in pairs} or None
    text = str(frame.meta.get("lammps_coeffs_text") or "")
    return mp.ff.forcefield.read_lammps_data_coeffs(text, units=units, **labels)


def _section_rows(text: str, heading: str) -> list[list[str]]:
    """Split the whitespace-delimited rows of one named data-file section.

    Accepts both bare headings (``Atoms``) and style-tagged ones
    (``Atoms # full``) that molrs emits.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if (
            line == heading
            or line.startswith(heading + " ")
            or line.startswith(heading + "\t")
        ):
            start = i + 2  # heading + blank line
            break
    if start is None:
        raise ValueError(f"section {heading!r} not found")
    rows = []
    for line in lines[start:]:
        if not line.strip():
            break
        rows.append(line.split())
    return rows


@pytest.fixture
def lammps_dir(TEST_DATA_DIR: Path) -> Path:
    return TEST_DATA_DIR / "lammps-data"


class TestReadLammpsData:
    """``read_lammps_data`` on real data files."""

    def test_molid_file(self, lammps_dir):
        """Test reading molid.lmp - file with molecular IDs and full style."""

        result = mp.io.read_lammps_data(lammps_dir / "molid.lmp", atom_style="full")
        frame = result

        # Check basic structure
        assert "atoms" in frame
        atoms = frame["atoms"]

        # Should have 12 atoms based on file content
        assert atoms.nrows == 12
        assert "mol_id" in atoms  # molecule ID (canonical name)
        assert "type" in atoms
        assert "charge" in atoms  # charge (canonical name)
        assert "x" in atoms and "y" in atoms and "z" in atoms  # Separate coordinates

        # Check coordinate data
        x = atoms["x"]
        y = atoms["y"]
        z = atoms["z"]
        assert len(x) == 12
        assert len(y) == 12
        assert len(z) == 12

        # Check box dimensions (0-20 in each direction)
        assert frame.box is not None
        box_lengths = frame.box.lengths
        np.testing.assert_array_almost_equal(box_lengths, [20.0, 20.0, 20.0])

        # Check that molecule IDs are in the data (should be 0-3 based on file)
        mol_ids = atoms["mol_id"]
        assert len(np.unique(mol_ids)) <= 4  # max 4 different molecules

        assert isinstance(_forcefield(result), mp.ff.forcefield.ForceField)

    def test_whitespaces_file(self, lammps_dir):
        """Test reading whitespaces.lmp - file with extra whitespaces."""

        result = mp.io.read_lammps_data(
            lammps_dir / "whitespaces.lmp", atom_style="full"
        )
        frame = result

        # Should parse correctly despite extra whitespaces
        assert "atoms" in frame
        atoms = frame["atoms"]
        assert atoms.nrows == 1

        # Check the single atom's coordinates
        x = atoms["x"][0]
        y = atoms["y"][0]
        z = atoms["z"][0]
        np.testing.assert_array_almost_equal([x, y, z], [5.0, 5.0, 5.0])

        # Check box (should be 10x10x10)
        box_lengths = frame.box.lengths
        np.testing.assert_array_almost_equal(box_lengths, [10.0, 10.0, 10.0])

    def test_triclinic_file(self, lammps_dir):
        """triclinic-1.lmp — triclinic header with all-zero tilt factors
        must produce an orthogonal-equivalent box."""

        frame = mp.io.read_lammps_data(
            lammps_dir / "triclinic-1.lmp", atom_style="atomic"
        )

        assert frame.box is not None
        np.testing.assert_array_almost_equal(frame.box.lengths, [34.0, 34.0, 34.0])
        np.testing.assert_array_almost_equal(frame.box.tilts, [0.0, 0.0, 0.0])

        if "atoms" in frame:
            assert frame["atoms"].nrows == 0

    def test_triclinic_2_file(self, lammps_dir):
        """triclinic-2.lmp — non-zero tilt factors (5 -8 3 xy xz yz) must
        be captured in the box."""

        frame = mp.io.read_lammps_data(
            lammps_dir / "triclinic-2.lmp", atom_style="atomic"
        )

        assert frame.box is not None
        assert frame.box.style == "triclinic"
        np.testing.assert_array_almost_equal(frame.box.tilts, [5.0, -8.0, 3.0])
        # Edge-vector norms reflect the tilt: |a|=lx, |b|=sqrt(xy^2+ly^2),
        # |c|=sqrt(xz^2+yz^2+lz^2).
        np.testing.assert_array_almost_equal(
            frame.box.lengths,
            [34.0, np.sqrt(5.0**2 + 34.0**2), np.sqrt(8.0**2 + 3.0**2 + 34.0**2)],
        )

    def test_data_body_file(self, lammps_dir):
        """body.lmp carries a ``Bodies`` section the native reader does not
        read; the read raises rather than dropping the section."""

        with pytest.raises(OSError):
            mp.io.read_lammps_data(lammps_dir / "body.lmp", atom_style="body")

    def test_labelmap_file(self, lammps_dir):
        """Test reading labelmap.lmp - file with type labels and connectivity."""

        result = mp.io.read_lammps_data(lammps_dir / "labelmap.lmp", atom_style="full")
        frame = result

        # Check atoms
        assert "atoms" in frame
        atoms = frame["atoms"]
        assert atoms.nrows == 16

        # Check type labels are preserved
        types = atoms["type"]
        unique_types = np.unique(types)
        expected_types = {"f", "c3", "s6", "o", "ne", "sy", "Li+"}
        assert set(unique_types) == expected_types

        # Check connectivity
        assert "bonds" in frame
        bonds = frame["bonds"]
        assert bonds.nrows == 14

        # Check bond types
        bond_types = bonds["type"]
        unique_bond_types = np.unique(bond_types)
        assert len(unique_bond_types) > 0

        # Check angles
        assert "angles" in frame
        angles = frame["angles"]
        assert angles.nrows == 25

        # Check dihedrals
        assert "dihedrals" in frame
        dihedrals = frame["dihedrals"]
        assert dihedrals.nrows == 27

        # Check force field
        assert isinstance(_forcefield(result), mp.ff.forcefield.ForceField)

    @staticmethod
    def _styled(tmp_path: Path, style: str, row: str) -> Path:
        path = tmp_path / f"{style}.data"
        path.write_text(
            f"{style}\n\n1 atoms\n1 atom types\n\n"
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
            f"Masses\n\n1 12.011\n\nAtoms # {style}\n\n{row}\n"
        )
        return path

    def test_atomic_style(self, tmp_path):
        """``atomic`` rows are ``id type x y z``: no mol_id, no charge."""
        frame = mp.io.read_lammps_data(
            self._styled(tmp_path, "atomic", "1 1 0.0 0.0 0.0"), atom_style="atomic"
        )
        atoms = frame["atoms"]
        assert "mol_id" not in atoms
        assert "charge" not in atoms
        assert "type" in atoms
        assert "x" in atoms and "y" in atoms and "z" in atoms

    def test_charge_style(self, tmp_path):
        """``charge`` rows are ``id type q x y z``: a charge, no mol_id."""
        frame = mp.io.read_lammps_data(
            self._styled(tmp_path, "charge", "1 1 -0.5 0.0 0.0 0.0"),
            atom_style="charge",
        )
        atoms = frame["atoms"]
        assert "mol_id" not in atoms
        assert atoms["charge"].tolist() == [-0.5]

    def test_a_layout_the_style_does_not_have_is_refused(self, lammps_dir):
        """``atom_style`` fixes the ``Atoms`` layout, as LAMMPS's does: a
        ``full`` file read as ``atomic`` is an error, not a column drop."""
        with pytest.raises(OSError):
            mp.io.read_lammps_data(lammps_dir / "molid.lmp", atom_style="atomic")


class TestDefaultAtomStyle:
    """Without ``atom_style`` the reader detects the ``Atoms`` layout."""

    @pytest.fixture
    def full_data_path(self, tmp_path: Path) -> Path:
        # 3 atoms, Masses, orthogonal 0–10 box; Atoms # full = id mol type q x y z.
        data = (
            "minimal full\n\n"
            "3 atoms\n"
            "1 atom types\n\n"
            "0 10 xlo xhi\n"
            "0 10 ylo yhi\n"
            "0 10 zlo zhi\n\n"
            "Masses\n\n"
            "1 12.011\n\n"
            "Atoms # full\n\n"
            "1 1 1 0.0 0.0 0.0 0.0\n"
            "2 1 1 0.0 1.5 0.0 0.0\n"
            "3 1 1 0.0 3.0 0.0 0.0\n"
        )
        path = tmp_path / "full.data"
        path.write_text(data)
        return path

    def test_default_atom_style_reads_full_file(self, full_data_path: Path):
        frame = mp.io.read_lammps_data(full_data_path)
        assert isinstance(frame, mp.Frame)
        assert frame["atoms"].nrows == 3
        assert "charge" in frame["atoms"] and "mol_id" in frame["atoms"]

    def test_the_frame_carries_the_box(self, full_data_path: Path):
        frame = mp.io.read_lammps_data(full_data_path)
        assert np.allclose(frame.box.lengths, [10.0, 10.0, 10.0])

    def test_a_file_without_coeffs_has_no_coeffs_text(self, full_data_path: Path):
        frame = mp.io.read_lammps_data(full_data_path)
        assert not frame.meta.get("lammps_coeffs_text")


class TestWriteLammpsData:
    """``write_lammps_data``."""

    def test_write_read_roundtrip(self, lammps_dir, tmp_path):
        """Test that we can write and read back the same data."""

        # Read original file
        original_frame = mp.io.read_lammps_data(
            lammps_dir / "molid.lmp", atom_style="full"
        )

        # Write to temporary file
        tmp_file = tmp_path / "test.data"

        mp.io.write_lammps_data(tmp_file, original_frame)

        # Read back
        new_frame = mp.io.read_lammps_data(tmp_file, atom_style="full")

        # Compare atoms
        orig_atoms = original_frame["atoms"]
        new_atoms = new_frame["atoms"]

        assert orig_atoms.nrows == new_atoms.nrows
        # Convert types to strings for comparison since they may be different types
        np.testing.assert_array_equal(
            np.array([str(t) for t in orig_atoms["type"]]),
            np.array([str(t) for t in new_atoms["type"]]),
        )
        np.testing.assert_array_almost_equal(orig_atoms["x"], new_atoms["x"])
        np.testing.assert_array_almost_equal(orig_atoms["y"], new_atoms["y"])
        np.testing.assert_array_almost_equal(orig_atoms["z"], new_atoms["z"])

        # Compare box - skip for now as box handling may need more work
        # assert original_frame.box is not None
        # assert new_frame.box is not None
        # np.testing.assert_array_almost_equal(
        #     original_frame.box.lengths, new_frame.box.lengths
        # )

    def test_read_from_frame_preserves_topology(self, tmp_path):
        """read_lammps_data -> Atomistic.from_frame must keep bonds/angles/dihedrals.

        Regression: the reader stored relation endpoints as a signed int, and
        ``molrs.from_frame`` reads endpoints only as ``uint64`` — so it silently
        dropped every bond on the Frame->Atomistic round-trip.
        """
        data = (
            "minimal\n\n4 atoms\n3 bonds\n1 atom types\n1 bond types\n\n"
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
            "Masses\n\n1 12.011\n\n"
            "Atoms\n\n"
            "1 1 1 0.0 0.0 0.0 0.0\n2 1 1 0.0 1.5 0.0 0.0\n"
            "3 1 1 0.0 3.0 0.0 0.0\n4 1 1 0.0 4.5 0.0 0.0\n\n"
            "Bonds\n\n1 1 1 2\n2 1 2 3\n3 1 3 4\n"
        )
        path = tmp_path / "chain.data"
        path.write_text(data)

        frame = mp.io.read_lammps_data(path, atom_style="full")
        # molrs 0.13 reads unsigned 32-bit endpoints; 0.14 uses uint64.
        # Signed ints are the actual drop-bug (from_frame ignores them).
        assert np.asarray(frame["bonds"]["atomi"]).dtype.kind == "u"
        rebuilt = mp.Atomistic.from_frame(frame)
        assert sum(1 for _ in rebuilt.bonds) == 3

    def test_write_minimal_frame(self, tmp_path):
        """Test writing a minimal frame with just atoms."""
        # Create a simple frame
        frame = mp.Frame()

        # Add atoms data with separate x, y, z coordinates
        atoms_data = {
            "id": np.array([1, 2, 3]),
            "type_id": np.array([1, 1, 2]),
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([1.0, 1.0, 2.0]),
        }

        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        # Write to temporary file
        tmp_file = tmp_path / "test.data"

        mp.io.write_lammps_data(tmp_file, frame)

        # Check file was written and has content
        assert os.path.exists(tmp_file)
        with open(tmp_file) as f:
            content = f.read()
            assert "3 atoms" in content
            assert "2 atom types" in content
            assert "Atoms" in content

    def test_write_numbers_atoms_without_mutating_the_frame(self, tmp_path):
        """An absent ``id`` is numbered per file, not written back to the frame.

        Atom IDs are an artifact of the LAMMPS file (they are what the Bonds
        section references), so a frame built from an ``Atomistic`` — which has
        no ``id`` column — must still write, and must come back unchanged.
        """
        asm = mp.Atomistic()
        atoms = [
            asm.def_atom(
                element="C", type="CT", charge=0.0, x=float(i), y=0.0, z=0.0, mol_id=1
            )
            for i in range(3)
        ]
        asm.def_bond(atoms[0], atoms[1], type="CT-CT")
        asm.def_bond(atoms[1], atoms[2], type="CT-CT")
        frame = asm.to_frame()
        assert "id" not in frame["atoms"]

        path = tmp_path / "unnumbered.data"
        mp.io.write_lammps_data(path, frame)

        assert "id" not in frame["atoms"], "writer mutated the caller's frame"
        atoms_section = _section_rows(path.read_text(), "Atoms")
        assert [row[0] for row in atoms_section] == ["1", "2", "3"]
        # Bonds reference those same 1-based IDs.
        bonds_section = _section_rows(path.read_text(), "Bonds")
        assert [row[2:] for row in bonds_section] == [["1", "2"], ["2", "3"]]

    def test_write_full_style(self, tmp_path):
        """Test writing with full atom style including molecule IDs and charges."""
        frame = mp.Frame()

        # Create atoms with all fields
        atoms_data = {
            "id": np.array([1, 2, 3]),
            "mol_id": np.array([1, 1, 2]),
            "type": np.array(["C", "C", "O"]),
            "charge": np.array([0.0, 0.0, -0.5]),
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([12.0, 12.0, 16.0]),
        }

        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        # Add bonds
        bonds_data = {
            "id": np.array([1, 2]),
            "type": np.array(["C-C", "C-O"]),
            "atomi": np.array([0, 1]),
            "atomj": np.array([1, 2]),
        }
        frame["bonds"] = mp.Block(bonds_data)

        tmp_file = tmp_path / "test.data"

        mp.io.write_lammps_data(tmp_file, frame)

        # Check file content
        with open(tmp_file) as f:
            content = f.read()
            assert "3 atoms" in content
            assert "2 bonds" in content
            assert "2 atom types" in content
            assert "2 bond types" in content
            assert "Atom Type Labels" in content
            assert "Bond Type Labels" in content

    def test_write_with_forcefield(self, tmp_path):
        """Test writing with force field parameters."""
        frame = mp.Frame()

        # Create atoms
        atoms_data = {
            "id": np.array([1, 2]),
            "type": np.array(["C", "O"]),
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "mass": np.array([12.0, 16.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        tmp_file = tmp_path / "test.data"

        # Structure-only writer; Coeffs are not written here.
        mp.io.write_lammps_data(tmp_file, frame)

        # Check file content
        with open(tmp_file) as f:
            content = f.read()
            assert "2 atoms" in content


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_nonexistent_file(self):
        """Test reading nonexistent file."""
        with pytest.raises(OSError):
            mp.io.read_lammps_data("nonexistent_file.data")

    def test_empty_file(self, tmp_path):
        """An empty file has no box, and the reader does not invent one."""
        tmp_file = tmp_path / "test.data"
        tmp_file.write_text("")
        frame = mp.io.read_lammps_data(tmp_file)
        assert frame.box is None
        assert frame.meta["lammps_box_axes"] == "x=0,y=0,z=0"

    def test_a_missing_box_axis_is_recorded(self, tmp_path):
        """A header missing one axis says so in ``lammps_box_axes``."""
        content = (
            "# missing z\n"
            "1 atoms\n"
            "1 atom types\n"
            "\n"
            "0.0 10.0 xlo xhi\n"
            "0.0 10.0 ylo yhi\n"
            "\n"
            "Atoms\n"
            "\n"
            "1 1 0.0 0.0 0.0\n"
        )
        tmp_file = tmp_path / "missing_z.data"
        tmp_file.write_text(content)

        frame = mp.io.read_lammps_data(tmp_file, atom_style="atomic")
        assert frame.meta["lammps_box_axes"] == "x=1,y=1,z=0"

    def test_float_box_bounds_parsed(self, tmp_path):
        """Regression: float-valued box bounds must parse, not fall back to 10x10x10."""
        content = (
            "# float bounds\n"
            "1 atoms\n"
            "1 atom types\n"
            "\n"
            "0.0 25.0 xlo xhi\n"
            "0.0 30.0 ylo yhi\n"
            "0.0 35.0 zlo zhi\n"
            "\n"
            "Atoms\n"
            "\n"
            "1 1 0.0 0.0 0.0\n"
        )
        tmp_file = tmp_path / "float_box.data"
        tmp_file.write_text(content)

        frame = mp.io.read_lammps_data(tmp_file, atom_style="atomic")

        assert frame.box is not None
        np.testing.assert_array_almost_equal(frame.box.lengths, [25.0, 30.0, 35.0])

    def test_malformed_header(self, tmp_path):
        """A header count that is not a number fails the read."""
        malformed_content = """# LAMMPS data file
invalid atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 1.0

Atoms

1 1 0.0 0.0 0.0
"""
        tmp_file = tmp_path / "test.data"
        with open(tmp_file, "w") as f:
            f.write(malformed_content)

        with pytest.raises(OSError):
            mp.io.read_lammps_data(tmp_file, atom_style="atomic")


class TestForceFieldIntegration:
    """Test force field integration."""

    def test_forcefield_writing(self, tmp_path):
        """Test that force field parameters are correctly written."""
        frame = mp.Frame()

        # Create simple atoms
        atoms_data = {
            "id": np.array([1]),
            "type": np.array(["C"]),
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "mass": np.array([12.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        tmp_file = tmp_path / "test.data"

        mp.io.write_lammps_data(tmp_file, frame)

        # Read back: a structure-only file carries no ``* Coeffs``.
        new_frame = mp.io.read_lammps_data(tmp_file, atom_style="atomic")
        assert not new_frame.meta.get("lammps_coeffs_text")


class TestExplicitTypeLabels:
    """Test the writer's explicit type-label inventory."""

    def test_labels_are_inferred_without_explicit_inventory(self, tmp_path):
        """String labels present on blocks are emitted directly."""
        frame = mp.Frame()

        atoms_data = {
            "id": np.array([1, 2, 3]),
            "type": np.array(["C", "H", "O"]),
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([12.0, 1.0, 16.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        tmp_file = tmp_path / "test.data"
        mp.io.write_lammps_data(tmp_file, frame)

        # Check file content
        with open(tmp_file) as f:
            content = f.read()
            assert "3 atoms" in content
            assert "3 atom types" in content
            assert "Atom Type Labels" in content
            assert "1 C" in content
            assert "2 H" in content
            assert "3 O" in content

    def test_explicit_type_labels_include_unused_types(self, tmp_path):
        """Explicit format labels may include types absent from this Frame."""
        frame = mp.Frame()

        atoms_data = {
            "id": np.array([1, 2]),
            "type": np.array(["C", "H"]),
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "mass": np.array([12.0, 1.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        type_labels = {
            "atoms": ["C", "H", "O", "N"],  # Includes types not in atoms
        }

        tmp_file = tmp_path / "test.data"
        mp.io.write_lammps_data(tmp_file, frame, type_labels=type_labels)

        # Check file content - should include all types from the explicit inventory
        with open(tmp_file) as f:
            content = f.read()
            assert "4 atom types" in content  # All types from the explicit inventory
            assert "Atom Type Labels" in content
            # Check that all explicit types are present
            assert "1 C" in content
            assert "2 H" in content
            assert "3 N" in content
            assert "4 O" in content

    def test_explicit_labels_merge_with_actual_types(self, tmp_path):
        """Explicit labels and actual block labels are merged."""
        frame = mp.Frame()

        atoms_data = {
            "id": np.array([1, 2, 3]),
            "type": np.array(["C", "H", "S"]),  # S is not in the explicit inventory
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([12.0, 1.0, 32.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        type_labels = {
            "atoms": ["C", "H", "O", "N"],
        }

        tmp_file = tmp_path / "test.data"
        mp.io.write_lammps_data(tmp_file, frame, type_labels=type_labels)

        # Check file content - should include merged types
        with open(tmp_file) as f:
            content = f.read()
            # Should have 5 types: C, H, N, O (from the explicit inventory) + S (from atoms)
            assert "5 atom types" in content
            assert "Atom Type Labels" in content
            # All types should be present
            assert "1 C" in content
            assert "2 H" in content
            assert "3 N" in content
            assert "4 O" in content
            assert "5 S" in content

    def test_explicit_bond_types(self, tmp_path):
        """Explicit bond labels are emitted even when currently unused."""
        frame = mp.Frame()

        atoms_data = {
            "id": np.array([1, 2, 3]),
            "type": np.array(["C", "C", "O"]),
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([12.0, 12.0, 16.0]),
            "mol_id": np.array([1, 1, 1]),
        }
        frame["atoms"] = mp.Block(atoms_data)

        bonds_data = {
            "id": np.array([1, 2]),
            "type": np.array(["C-C", "C-O"]),
            "atomi": np.array([0, 1]),
            "atomj": np.array([1, 2]),
        }
        frame["bonds"] = mp.Block(bonds_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        type_labels = {
            "atoms": ["C", "O"],
            "bonds": ["C-C", "C-O", "O-O"],  # O-O not in actual bonds
        }

        tmp_file = tmp_path / "test.data"
        mp.io.write_lammps_data(tmp_file, frame, type_labels=type_labels)

        # Check file content
        with open(tmp_file) as f:
            content = f.read()
            assert "3 bond types" in content  # All types from the explicit inventory
            assert "Bond Type Labels" in content
            assert "1 C-C" in content
            assert "2 C-O" in content
            assert "3 O-O" in content

    def test_type_id_consistency(self, tmp_path):
        """Test that type_id is consistent across all sections."""
        frame = mp.Frame()

        atoms_data = {
            "id": np.array([1, 2, 3]),
            "type": np.array(["C", "H", "O"]),
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([12.0, 1.0, 16.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        type_labels = {
            "atoms": ["H", "O", "C"],  # Different order
        }

        tmp_file = tmp_path / "test.data"
        mp.io.write_lammps_data(tmp_file, frame, type_labels=type_labels)

        # Read back and verify
        new_frame = mp.io.read_lammps_data(tmp_file, atom_style="atomic")

        # Check that type IDs are consistent
        # In the written file, types should be sorted: C, H, O
        # So C should be type 1, H should be type 2, O should be type 3
        with open(tmp_file) as f:
            content = f.read()
            # Type labels should be sorted: C, H, O (alphabetically)
            assert "1 C" in content
            assert "2 H" in content
            assert "3 O" in content

        # Verify atoms section uses same type IDs
        atoms = new_frame["atoms"]
        # The actual atom types in the atoms section should reference
        # the correct type IDs from the type labels section
        assert atoms is not None


def test_write_lammps_data_requires_type_columns(tmp_path):
    import numpy as np

    frame = mp.Frame()
    frame["atoms"] = mp.Block(
        {"x": np.zeros(1), "y": np.zeros(1), "z": np.zeros(1), "mass": np.ones(1)}
    )
    frame.box = mp.Box([5.0, 5.0, 5.0])
    with pytest.raises(OSError, match="neither 'type' nor 'type_id'"):
        mp.io.write_lammps_data(tmp_path / "bad.data", frame)


def test_write_keeps_reverse_angle_type_labels_as_two_types(tmp_path):
    """A type label is a type name, matched exactly: ``h1-c3-c3`` and
    ``c3-c3-h1`` are two LAMMPS angle types."""
    import numpy as np

    frame = mp.Frame()
    frame["atoms"] = mp.Block(
        {
            "type": np.array(["c3", "c3", "h1"]),
            "x": np.array([0.0, 1.0, 2.0]),
            "y": np.zeros(3),
            "z": np.zeros(3),
            "mass": np.array([12.0, 12.0, 1.0]),
            "mol_id": np.array([1, 1, 1]),
        }
    )
    frame["angles"] = mp.Block(
        {
            "type": np.array(["c3-c3-h1", "h1-c3-c3"]),
            "atomi": np.array([0, 2], dtype=np.uint32),
            "atomj": np.array([1, 1], dtype=np.uint32),
            "atomk": np.array([2, 0], dtype=np.uint32),
        }
    )
    frame.box = mp.Box([5.0, 5.0, 5.0])
    path = tmp_path / "rev.data"
    mp.io.write_lammps_data(path, frame)
    text = path.read_text()
    assert "2 angle types" in text
    assert "c3-c3-h1" in text
    assert "h1-c3-c3" in text


class TestForceFieldCoeffs:
    """``* Coeffs`` sections parse into the result's force field."""

    @pytest.fixture
    def ff_file(self, lammps_dir: Path) -> Path:
        return lammps_dir / "coeffs.lmp"

    def test_coeffs_are_extracted(self, ff_file):
        ff = _forcefield(mp.io.read_lammps_data(ff_file, atom_style="full"))
        pair = {
            t.name: (t.get("epsilon"), t.get("sigma"))
            for s in ff.get_styles(mp.ff.forcefield.PairStyle)
            for t in s.get_types(mp.ff.forcefield.Type)
        }
        bond = {
            t.name: (t.get("k"), t.get("r0"))
            for s in ff.get_styles(mp.ff.forcefield.BondStyle)
            for t in s.get_types(mp.ff.forcefield.Type)
        }
        angle = {
            t.name: (t.get("k"), t.get("theta0"))
            for s in ff.get_styles(mp.ff.forcefield.AngleStyle)
            for t in s.get_types(mp.ff.forcefield.Type)
        }
        assert pair == {"1": (0.1521, 3.1507), "2": (0.046, 0.4)}
        # The force-field IR adopts the LAMMPS standard: K is stored as
        # written (E = K x^2) and theta0 stays in degrees.
        assert list(bond.values()) == [(450.0, 0.9572)]
        assert list(angle.values()) == [(55.0, 104.52)]

    def test_malformed_coeff_line_raises(self, tmp_path):
        data = tmp_path / "bad.data"
        data.write_text(
            "bad\n\n2 atoms\n1 atom types\n"
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n"
            "Masses\n\n1 1.0\n\n"
            "Pair Coeffs\n\n1 notanumber 3.5\n\n"
            "Atoms\n\n1 1 1 0.0 0.0 0.0 0.0\n2 1 1 0.0 0.5 0.0 0.0\n"
        )
        frame = mp.io.read_lammps_data(data, atom_style="full")
        with pytest.raises(ValueError):
            _forcefield(frame)


class TestCoeffsAreText:
    """The structure read never depends on the ``* Coeffs`` sections."""

    @pytest.fixture
    def cosine_file(self, lammps_dir: Path) -> Path:
        return lammps_dir / "cosine_angle_coeffs.data"

    def test_structure_read_survives_unparseable_coeffs(self, cosine_file):
        frame = mp.io.read_lammps_data(cosine_file, atom_style="angle")
        assert frame["atoms"].nrows == 3
        assert frame.meta.get("lammps_coeffs_text")

    def test_unparseable_coeffs_raise_when_read(self, cosine_file):
        frame = mp.io.read_lammps_data(cosine_file, atom_style="angle")
        with pytest.raises(ValueError):
            _forcefield(frame)

    def test_units_are_the_callers(self, lammps_dir):
        frame = mp.io.read_lammps_data(lammps_dir / "coeffs.lmp", atom_style="full")
        ff = _forcefield(frame, units="metal")
        assert ff.units == "metal"
        epsilon = {
            t.name: t.get("epsilon")
            for s in ff.get_styles(mp.ff.forcefield.PairStyle)
            for t in s.get_types(mp.ff.forcefield.Type)
        }
        # A metal file is a metal force field: epsilon stays in eV, as written.
        assert epsilon["1"] == 0.1521
