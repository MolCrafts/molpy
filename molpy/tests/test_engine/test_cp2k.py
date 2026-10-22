"""Cp2kEngine: identity, input extension, and the command it builds."""

import subprocess
from unittest.mock import patch

import pytest
import molpy as mp


def test_identity_and_extension():
    engine = mp.engine.Cp2kEngine(check_executable=False)
    assert engine.key == "engine.cp2k"
    assert engine._get_default_extension() == ".inp"


def test_a_failure_carries_the_tail_of_cp2k_out(tmp_path):
    engine = mp.engine.Cp2kEngine(check_executable=False)

    def fails(argv, *, cwd, **_):
        (cwd / "cp2k.out").write_text(" *** ABORT: bad &GLOBAL ***\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 1, None, None)

    with (
        patch("subprocess.run", side_effect=fails),
        pytest.raises(mp.tool.ToolError, match="ABORT: bad &GLOBAL"),
    ):
        engine.run("&GLOBAL\n", workdir=tmp_path)


def test_run_invokes_the_binary_with_input_and_log_flags(tmp_path):
    config = mp.config.load_config(
        {"engine": {"cp2k": {"launcher": ["mpirun", "-np", "2"]}}}
    )
    engine = mp.engine.Cp2kEngine(config=config, check_executable=False)
    script = mp.engine.Script.from_text("input", "&GLOBAL\n&END GLOBAL\n")
    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        engine.run(script, workdir=tmp_path)
    argv = run.call_args.args[0]
    assert argv[:4] == ["mpirun", "-np", "2", "cp2k.psmp"]
    assert argv[argv.index("-i") + 1].endswith("input.inp")
    assert argv[argv.index("-o") + 1] == "cp2k.out"
