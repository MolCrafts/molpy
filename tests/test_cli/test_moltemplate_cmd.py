"""Tests for the ``molpy moltemplate`` CLI subcommand."""

from __future__ import annotations

import json


class TestInfo:
    def test_info_prints_counts(self, run_cli, tip3p_lt):
        rc, out, _ = run_cli("moltemplate", "info", str(tip3p_lt))
        assert rc == 0
        assert "atom types:" in out
        assert "atoms:" in out
        assert "bonds:" in out


class TestParse:
    def test_parse_summary(self, run_cli, tip3p_lt):
        rc, out, _ = run_cli("moltemplate", "parse", str(tip3p_lt))
        assert rc == 0
        assert "ClassDef" in out
        assert "NewStmt" in out

    def test_parse_json(self, run_cli, tip3p_lt, tmp_path):
        out_json = tmp_path / "ir.json"
        rc, _, _ = run_cli(
            "moltemplate", "parse", str(tip3p_lt), "--json", str(out_json)
        )
        assert rc == 0
        data = json.loads(out_json.read_text())
        assert "statements" in data


class TestRun:
    def test_run_lammps(self, run_cli, tip3p_lt, tmp_path):
        rc, out, _ = run_cli(
            "moltemplate",
            "run",
            str(tip3p_lt),
            "--emit",
            "lammps",
            "--out-dir",
            str(tmp_path),
            "--prefix",
            "w",
        )
        assert rc == 0
        assert "[lammps]" in out
        assert (tmp_path / "w.data").exists()
        assert (tmp_path / "w.in").exists()

    def test_run_all(self, run_cli, tip3p_lt, tmp_path):
        rc, out, _ = run_cli(
            "moltemplate",
            "run",
            str(tip3p_lt),
            "--emit",
            "all",
            "--out-dir",
            str(tmp_path),
            "--prefix",
            "w",
        )
        assert rc == 0
        for engine in ("lammps", "openmm", "gromacs", "xml"):
            assert f"[{engine}]" in out

    def test_run_unknown_engine(self, run_cli, tip3p_lt, tmp_path):
        rc, _, err = run_cli(
            "moltemplate",
            "run",
            str(tip3p_lt),
            "--emit",
            "nonexistent",
            "--out-dir",
            str(tmp_path),
        )
        assert rc == 2
        assert "unknown engine" in err


class TestConvert:
    def test_convert_lt_to_xml(self, run_cli, tip3p_lt, tmp_path):
        out_xml = tmp_path / "tip3p.xml"
        rc, out, _ = run_cli("moltemplate", "convert", str(tip3p_lt), str(out_xml))
        assert rc == 0
        assert out_xml.exists()
        assert "<ForceField" in out_xml.read_text()


class TestEntry:
    def test_no_command_prints_help(self, run_cli):
        rc, out, _ = run_cli()
        assert rc == 0
        assert "MolPy CLI" in out
