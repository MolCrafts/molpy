"""``molpy.io.convert`` and the ``molpy convert`` command."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

import molpy as mp
from molpy.cli import main
from molpy.core import Element
from molpy.io.formats import FORMATS, Format, convert, register, resolve


def _frame(n: int, x0: float) -> mp.Frame:
    frame = mp.Frame()
    frame["atoms"] = {
        "element": np.array(["O", "H"][:n]),
        "name": np.array(["O", "H"][:n]),
        "mass": np.array(
            [Element.by_symbol(s).mass for s in ("O", "H")[:n]], dtype=np.float64
        ),
        "x": np.array([x0, x0 + 0.96][:n]),
        "y": np.array([0.0, 0.0][:n]),
        "z": np.zeros(n),
    }
    frame.box = mp.Box.ortho((10.0, 11.0, 12.0))
    return frame


def _xs(path: Path) -> list[float]:
    return [float(value) for value in mp.io.read_xyz(path)["atoms"]["x"]]


class TestResolve:
    def test_suffix_and_gzip_and_poscar_name(self) -> None:
        assert resolve(Path("a.xyz"), None).name == "xyz"
        assert resolve(Path("a.xyz.gz"), None).name == "xyz"
        assert resolve(Path("traj.mrec.zip"), None).name == "mrec"
        assert resolve(Path("POSCAR"), None).name == "poscar"
        assert resolve(Path("a.data"), None).name == "lammps-data"

    def test_explicit_name_wins_over_the_suffix(self) -> None:
        assert resolve(Path("a.xyz"), "pdb").name == "pdb"

    def test_unknown_name_and_suffix_name_the_table(self) -> None:
        with pytest.raises(ValueError, match="unknown format 'nope'"):
            resolve(Path("a.xyz"), "nope")
        with pytest.raises(ValueError, match="cannot tell the format"):
            resolve(Path("a.nope"), None)

    def test_a_second_registration_of_the_same_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="xyz"):
            register(Format("xyz", (".xyz-again",), read=mp.io.read_xyz))

    def test_sniff_picks_the_one_row_that_claims_the_bytes(
        self, tmp_path: Path
    ) -> None:
        before = len(FORMATS)

        def yes(path: Path) -> bool:
            return path.read_bytes().startswith(b"YES")

        def no(path: Path) -> bool:
            return path.read_bytes().startswith(b"NO")

        register(Format("sniff-yes", (".sniffme",), read=mp.io.read_xyz, sniff=yes))
        register(Format("sniff-no", (".sniffme",), read=mp.io.read_xyz, sniff=no))
        path = tmp_path / "a.sniffme"
        path.write_bytes(b"YES")
        try:
            assert resolve(path, None).name == "sniff-yes"
        finally:
            del FORMATS[before:]


class TestConvert:
    def test_xyz_round_trip_keeps_coordinates(self, tmp_path: Path) -> None:
        source = tmp_path / "in.xyz"
        mp.io.write_xyz(source, _frame(2, 1.0))
        destination = tmp_path / "out.xyz"
        convert(source, destination)
        np.testing.assert_allclose(_xs(destination), [1.0, 1.96])

    def test_explicit_formats_ignore_the_file_names(self, tmp_path: Path) -> None:
        source = tmp_path / "in.bin"
        mp.io.write_xyz(source, _frame(2, 1.0))
        destination = tmp_path / "out.bin"
        convert(source, destination, in_format="xyz", out_format="pdb")
        assert mp.io.read_pdb(destination)["atoms"].n_rows == 2

    def test_one_mrec_snapshot_reads_back(self, tmp_path: Path) -> None:
        source = tmp_path / "in.xyz"
        mp.io.write_xyz(source, _frame(2, 1.0))
        destination = tmp_path / "out.mrec"
        convert(source, destination)
        back = mp.io.mrec.MrecFile(destination).frame()
        assert back is not None
        np.testing.assert_allclose(back["atoms"]["x"], [1.0, 1.96])

    def test_a_packed_record_round_trips(self, tmp_path: Path) -> None:
        source = tmp_path / "in.xyz"
        mp.io.write_xyz(source, _frame(2, 2.0))
        destination = tmp_path / "out.mrec.zip"
        convert(source, destination)
        back = tmp_path / "back.xyz"
        convert(destination, back)
        np.testing.assert_allclose(_xs(back), [2.0, 2.96])

    def test_two_frames_become_a_trajectory(self, tmp_path: Path) -> None:
        source = tmp_path / "in.xyz"
        with mp.io.xyz.XyzWriter(source) as writer:
            writer.write(_frame(2, 1.0))
            writer.write(_frame(2, 3.0))
        record = tmp_path / "out.mrec"
        convert(source, record)
        reader = mp.io.mrec.MrecReader(record)
        assert len(reader) == 2
        np.testing.assert_allclose(reader[1]["atoms"]["x"], [3.0, 3.96])
        back = tmp_path / "back.xyz"
        convert(record, back)
        assert [frame["atoms"].n_rows for frame in mp.io.xyz.XyzReader(back)] == [2, 2]

    def test_a_single_frame_format_refuses_a_trajectory(self, tmp_path: Path) -> None:
        source = tmp_path / "in.xyz"
        with mp.io.xyz.XyzWriter(source) as writer:
            writer.write(_frame(2, 1.0))
            writer.write(_frame(2, 3.0))
        with pytest.raises(ValueError, match="writes one frame"):
            convert(source, tmp_path / "out.poscar")

    def test_frame_selects_one_snapshot(self, tmp_path: Path) -> None:
        source = tmp_path / "in.xyz"
        with mp.io.xyz.XyzWriter(source) as writer:
            writer.write(_frame(2, 1.0))
            writer.write(_frame(2, 3.0))
        destination = tmp_path / "out.xyz"
        convert(source, destination, frame=1)
        np.testing.assert_allclose(_xs(destination), [3.0, 3.96])

    def test_a_missing_file_and_a_past_the_end_frame(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            convert(tmp_path / "missing.xyz", tmp_path / "out.pdb")
        source = tmp_path / "in.xyz"
        mp.io.write_xyz(source, _frame(2, 1.0))
        with pytest.raises(ValueError, match="past the end"):
            convert(source, tmp_path / "out.xyz", frame=3)

    def test_gzip_suffix_is_read(self, tmp_path: Path) -> None:
        plain = tmp_path / "in.xyz"
        mp.io.write_xyz(plain, _frame(2, 1.0))
        packed = tmp_path / "in.xyz.gz"
        packed.write_bytes(gzip.compress(plain.read_bytes()))
        destination = tmp_path / "out.xyz"
        convert(packed, destination)
        np.testing.assert_allclose(_xs(destination), [1.0, 1.96])

    def test_a_gzip_destination_round_trips(self, tmp_path: Path) -> None:
        source = tmp_path / "in.xyz"
        with mp.io.xyz.XyzWriter(source) as writer:
            writer.write(_frame(2, 1.0))
            writer.write(_frame(2, 3.0))
        destination = tmp_path / "out.xyz.gz"
        convert(source, destination)
        assert destination.read_bytes()[:2] == b"\x1f\x8b"
        back = tmp_path / "back.xyz"
        convert(destination, back)
        xs = [float(value) for frame in mp.io.xyz.XyzReader(back) for value in frame["atoms"]["x"]]
        np.testing.assert_allclose(xs, [1.0, 1.96, 3.0, 3.96])

    def test_a_mixed_case_archive_is_the_requested_path(self, tmp_path: Path) -> None:
        source = tmp_path / "in.xyz"
        with mp.io.xyz.XyzWriter(source) as writer:
            writer.write(_frame(2, 2.0))
            writer.write(_frame(2, 4.0))
        destination = tmp_path / "Out.MREC.ZIP"
        convert(source, destination)
        assert destination.is_file()
        assert not (tmp_path / "Out.MREC.mrec.zip").exists()
        assert not (tmp_path / "Out.mrec").exists()
        back = tmp_path / "back.xyz"
        convert(destination, back)
        np.testing.assert_allclose(_xs(back), [2.0, 2.96])

    def test_a_mask_that_appears_on_a_later_frame_is_kept(self, tmp_path: Path) -> None:
        first = _frame(2, 1.0)
        first["atoms"]["charge"] = np.array([0.0, 0.0])
        second = _frame(2, 3.0)
        second["atoms"]["charge"] = np.array([0.1, -0.1])
        second["atoms"].set_validity("charge", [True, False])
        frames = [first, second]

        def read_frames(_path: Path):
            return iter(frames)

        before = len(FORMATS)
        register(
            Format(
                "het",
                (".het",),
                read_frames=read_frames,
                read=lambda _path: frames[0],
            )
        )
        source = tmp_path / "in.het"
        source.write_bytes(b"x")
        destination = tmp_path / "out.mrec"
        try:
            convert(source, destination)
            reader = mp.io.mrec.MrecReader(destination)
            assert len(reader) == 2
            assert reader[0]["atoms"].validity("charge") is None
            assert list(reader[1]["atoms"].validity("charge")) == [True, False]
        finally:
            del FORMATS[before:]


class TestCommand:
    def test_convert_writes_the_output(self, tmp_path: Path) -> None:
        source = tmp_path / "in.xyz"
        mp.io.write_xyz(source, _frame(2, 1.0))
        destination = tmp_path / "out.pdb"
        assert main(["convert", "--in", str(source), "--out", str(destination)]) == 0
        assert mp.io.read_pdb(destination)["atoms"].n_rows == 2

    def test_a_bad_format_returns_failure(self, tmp_path: Path, capsys) -> None:
        source = tmp_path / "in.xyz"
        mp.io.write_xyz(source, _frame(2, 1.0))
        status = main(
            [
                "convert",
                "--in",
                str(source),
                "--out",
                str(tmp_path / "out.pdb"),
                "--in-format",
                "nope",
            ]
        )
        assert status == 1
        assert "nope" in capsys.readouterr().err
