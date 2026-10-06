"""CP2KEngine: identity, input extension, and the command it builds."""

from unittest.mock import patch

import pytest

from molpy.engine import Script
from molpy.engine import CP2KEngine


def test_identity_and_extension():
    engine = CP2KEngine(executable="cp2k.psmp", check_executable=False)
    assert engine.name == "CP2K"
    assert engine._get_default_extension() == ".inp"


def test_execute_without_a_script_raises(tmp_path):
    engine = CP2KEngine(executable="cp2k.psmp", check_executable=False)
    with pytest.raises(RuntimeError, match="No input script"):
        engine._execute(tmp_path)


def test_run_invokes_the_binary_with_input_and_log_flags(tmp_path):
    engine = CP2KEngine(
        executable="cp2k.psmp", check_executable=False, launcher=["mpirun", "-np", "2"]
    )
    script = Script.from_text(
        name="input", text="&GLOBAL\n&END GLOBAL\n", language="other"
    )
    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        engine.run(script, workdir=tmp_path, check=False)
    argv = run.call_args.args[0]
    assert argv[:4] == ["mpirun", "-np", "2", "cp2k.psmp"]
    assert argv[argv.index("-i") + 1].endswith("input.inp")
    assert argv[argv.index("-o") + 1] == "cp2k.out"
