"""SanderWrapper.minimize: the control file it writes and its failure modes."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from molpy.wrapper import SanderWrapper


def test_requires_a_working_directory(tmp_path):
    wrapper = SanderWrapper()
    with pytest.raises(ValueError, match="working directory"):
        wrapper.minimize(tmp_path / "x.prmtop", tmp_path / "x.inpcrd")


def test_control_file_and_arguments(tmp_path):
    wrapper = SanderWrapper(tmp_path / "run")
    seen = {}

    def fake_run(*, args, check):
        seen["args"] = args
        (tmp_path / "run" / "min.rst").write_text("relaxed\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch.object(SanderWrapper, "run", side_effect=fake_run):
        rst = wrapper.minimize(
            tmp_path / "x.prmtop", tmp_path / "x.inpcrd", max_iter=100
        )

    assert rst == tmp_path / "run" / "min.rst"
    mdin = (tmp_path / "run" / "min.in").read_text(encoding="utf-8")
    assert "imin=1, maxcyc=100, ncyc=50," in mdin
    assert "ntxo=1" in mdin
    assert seen["args"][:3] == ["-O", "-i", "min.in"]
    assert seen["args"][-4:] == ["-r", "min.rst", "-o", "min.out"]


def test_failed_run_raises_with_the_tool_output(tmp_path):
    wrapper = SanderWrapper(tmp_path)
    failure = SimpleNamespace(returncode=1, stdout="", stderr="bad prmtop")
    with patch.object(SanderWrapper, "run", return_value=failure):
        with pytest.raises(RuntimeError, match="bad prmtop"):
            wrapper.minimize(tmp_path / "x.prmtop", tmp_path / "x.inpcrd")
