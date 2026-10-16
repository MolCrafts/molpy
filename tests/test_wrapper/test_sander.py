"""SanderWrapper.run_from_mdin: the control file it writes and the command."""

import subprocess
from unittest.mock import patch

import pytest

from molpy.wrapper import SanderWrapper, ToolError

_MDIN = "minimize\n&cntrl\n  imin=1, maxcyc=100,\n/\n"


def test_requires_a_working_directory(tmp_path):
    with pytest.raises(ValueError, match="working directory"):
        SanderWrapper().run_from_mdin(
            _MDIN, tmp_path / "x.prmtop", tmp_path / "x.inpcrd"
        )


def test_writes_the_control_text_and_runs_sander_on_it(tmp_path):
    wrapper = SanderWrapper(tmp_path / "run")
    with patch(
        "subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")
    ) as run:
        wrapper.run_from_mdin(
            _MDIN, tmp_path / "x.prmtop", tmp_path / "x.inpcrd", name="min"
        )
    assert (tmp_path / "run" / "min.in").read_text(encoding="utf-8") == _MDIN
    argv = run.call_args.args[0]
    assert argv[:4] == ["sander", "-O", "-i", "min.in"]
    assert argv[-4:] == ["-r", "min.rst", "-o", "min.out"]
    assert run.call_args.kwargs["cwd"] == tmp_path / "run"


def test_a_failed_run_raises_with_the_tool_output(tmp_path):
    failed = subprocess.CompletedProcess([], 1, "", "bad prmtop")
    with (
        patch("subprocess.run", return_value=failed),
        pytest.raises(ToolError, match="bad prmtop") as raised,
    ):
        SanderWrapper(tmp_path).run_from_mdin(
            _MDIN, tmp_path / "x.prmtop", tmp_path / "x.inpcrd"
        )
    assert raised.value.tool == "wrapper.sander"
    assert raised.value.returncode == 1
