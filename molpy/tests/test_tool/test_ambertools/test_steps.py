"""The shared AmberTools pipeline: the leap script and the stale-output policy.

``subprocess.run`` is faked as in ``tests/test_ff/test_ambertools.py``; a
program either writes its outputs or exits 0 having written nothing.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path, PureWindowsPath
from unittest.mock import patch

import pytest

from molpy.tool.ambertools import (
    AmberSteps,
    leap_mol2,
    leap_path,
    leap_script,
    leap_sequence,
    run_directory,
)
import molpy as mp


class _Fake:
    """Writes each program's outputs, unless it is told to write nothing."""

    def __init__(self, *, silent: bool = False) -> None:
        self.silent = silent
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], *, cwd: str, **_: object):
        self.calls.append(argv)
        if not self.silent:
            args = argv[1:]
            if argv[0] == "tleap":
                script = (Path(cwd) / args[1]).read_text(encoding="utf-8")
                save = next(
                    line for line in script.splitlines() if line.startswith("save")
                )
                for target in shlex.split(save)[2:]:
                    Path(target).write_text("new\n", encoding="utf-8")
            else:
                target = Path(args[args.index("-o") + 1])
                (target if target.is_absolute() else Path(cwd) / target).write_text(
                    "new\n", encoding="utf-8"
                )
        return subprocess.CompletedProcess(argv, 0, "", "")


def _run(fake: _Fake):
    return (
        patch.object(
            mp.tool.Tool, "resolve_executable", return_value="/amber/bin/tool"
        ),
        patch("molpy.tool.process.subprocess.run", side_effect=fake),
    )


class TestLeapScript:
    def test_every_path_is_quoted_in_load_order(self, tmp_path):
        work = tmp_path / "my work"
        portable = work.as_posix()
        script = leap_script(
            "gaff2",
            leap_sequence(["HEO", "EO", "TEO"]),
            work / "polymer.prmtop",
            work / "polymer.inpcrd",
            frcmods=[work / "EO.frcmod"],
            preps=[work / "HEO.prepi", work / "EO.prepi"],
        )
        assert script.splitlines() == [
            "source leaprc.gaff2",
            f'loadamberparams "{portable}/EO.frcmod"',
            f'loadamberprep "{portable}/HEO.prepi"',
            f'loadamberprep "{portable}/EO.prepi"',
            "MOL = sequence { HEO EO TEO }",
            f'saveamberparm MOL "{portable}/polymer.prmtop" "{portable}/polymer.inpcrd"',
            "quit",
        ]
        save = shlex.split(script.splitlines()[-2])
        assert save[2:] == [
            (work / "polymer.prmtop").as_posix(),
            (work / "polymer.inpcrd").as_posix(),
        ]

    def test_loadmol2_quotes_its_path(self, tmp_path):
        assert (
            leap_mol2(tmp_path / "a b.mol2")
            == f'loadmol2 "{(tmp_path / "a b.mol2").as_posix()}"'
        )

    def test_windows_paths_use_portable_separators(self):
        path = PureWindowsPath("C:/my work/a b.mol2")
        assert leap_mol2(path) == 'loadmol2 "C:/my work/a b.mol2"'

    def test_a_double_quote_cannot_be_written(self, tmp_path):
        with pytest.raises(ValueError, match="double quote"):
            leap_path(tmp_path / 'a"b.mol2')


class TestStalePolicy:
    def test_a_step_deletes_its_outputs_before_it_runs(self, tmp_path):
        steps = AmberSteps(tmp_path, mp.config.load_config())
        prmtop, inpcrd = tmp_path / "MOL.prmtop", tmp_path / "MOL.inpcrd"
        prmtop.write_text("old\n", encoding="utf-8")
        inpcrd.write_text("old\n", encoding="utf-8")
        script = leap_script("gaff", leap_mol2(tmp_path / "in.mol2"), prmtop, inpcrd)
        fake = _Fake(silent=True)
        first, second = _run(fake)
        with (
            first,
            second,
            pytest.raises(mp.tool.ToolError, match="left MOL.prmtop unwritten"),
        ):
            steps.tleap(script, prmtop, inpcrd)
        assert not prmtop.exists()
        assert not inpcrd.exists()

    def test_tleap_must_write_the_coordinates_too(self, tmp_path):
        steps = AmberSteps(tmp_path, mp.config.load_config())
        prmtop, inpcrd = tmp_path / "MOL.prmtop", tmp_path / "MOL.inpcrd"
        script = leap_script("gaff", leap_mol2(tmp_path / "in.mol2"), prmtop, prmtop)
        first, second = _run(_Fake())
        with (
            first,
            second,
            pytest.raises(mp.tool.ToolError, match="left MOL.inpcrd unwritten"),
        ):
            steps.tleap(script, prmtop, inpcrd)

    def test_antechamber_writes_the_format_its_output_names(self, tmp_path):
        steps = AmberSteps(tmp_path, mp.config.load_config())
        fake = _Fake()
        first, second = _run(fake)
        with first, second:
            steps.antechamber(
                tmp_path / "in.mol2",
                tmp_path / "out.ac",
                parameter_set="gaff",
                charge_method="bcc",
                net_charge=0,
            )
            steps.parmchk2(
                tmp_path / "out.ac", tmp_path / "out.frcmod", parameter_set="gaff"
            )
        ante, parmchk2 = fake.calls[0][1:], fake.calls[1][1:]
        assert ante[ante.index("-fo") + 1] == "ac"
        assert parmchk2[parmchk2.index("-f") + 1] == "ac"
        assert parmchk2[parmchk2.index("-s") + 1] == "gaff"

    def test_the_am1_step_asks_antechamber_for_mulliken_charges_only(self, tmp_path):
        steps = AmberSteps(tmp_path, mp.config.load_config())
        fake = _Fake()
        first, second = _run(fake)
        with first, second:
            steps.am1(tmp_path / "in.mol2", tmp_path / "sqm.ac", net_charge=-1)
        args = fake.calls[0][1:]
        assert args[args.index("-c") + 1] == "mul"
        assert args[args.index("-fo") + 1] == "ac"
        assert args[args.index("-nc") + 1] == "-1"

    def test_an_unknown_suffix_is_refused(self, tmp_path):
        steps = AmberSteps(tmp_path, mp.config.load_config())
        with pytest.raises(ValueError, match=r"\.ac or \.mol2"):
            steps.parmchk2(
                tmp_path / "a.pdb", tmp_path / "a.frcmod", parameter_set="gaff"
            )


def test_the_run_directory_is_keyed_by_every_part(tmp_path):
    base = run_directory(tmp_path, "mol2 text", "gaff2", "bcc")
    assert base.is_dir()
    assert run_directory(tmp_path, "mol2 text", "gaff2", "bcc") == base
    assert run_directory(tmp_path, "mol2 text", "gaff", "bcc") != base
    assert run_directory(tmp_path, "mol2 text", "gaff2", "gas") != base
    assert run_directory(tmp_path, "mol2 text", "gaff2bcc") != base
