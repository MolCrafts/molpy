"""Unit tests for :mod:`molpy.wrapper.env` — EnvSpec infrastructure."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from molpy.wrapper import EnvSpec


class TestEnvSpecResolve:
    def test_system_default(self):
        spec = EnvSpec.resolve()
        assert spec.is_system
        assert spec.env is None
        assert spec.env_manager is None
        assert EnvSpec.system() == spec

    def test_both_or_neither(self):
        with pytest.raises(ValueError, match="incomplete"):
            EnvSpec.resolve(env="AmberTools25")
        with pytest.raises(ValueError, match="incomplete"):
            EnvSpec.resolve(env_manager="conda")

    def test_conda_name(self):
        spec = EnvSpec.resolve("AmberTools25", "conda")
        assert spec.env == "AmberTools25"
        assert spec.env_manager == "conda"
        assert not spec.is_system

    def test_conda_prefix_path(self):
        spec = EnvSpec.resolve("/opt/conda/envs/at", "conda")
        assert spec.env_manager == "conda"
        assert spec.command_prefix()[1:4] == ["run", "-p", "/opt/conda/envs/at"]

    def test_venv_aliases_normalise(self):
        for alias in ("venv", "pip", "virtualenv", "Venv", "PIP"):
            spec = EnvSpec.resolve("/path/to/.venv", alias)
            assert spec.env_manager == "venv"

    def test_unsupported_manager(self):
        with pytest.raises(ValueError, match="Unsupported env_manager"):
            EnvSpec.resolve("x", "uv")


class TestEnvSpecCommandPrefix:
    def test_system_and_venv_have_empty_prefix(self):
        assert EnvSpec.system().command_prefix() == []
        assert EnvSpec.resolve("/tmp/v", "venv").command_prefix() == []

    def test_conda_name_uses_n(self):
        prefix = EnvSpec.resolve("AmberTools25", "conda").command_prefix()
        assert prefix[1:4] == ["run", "-n", "AmberTools25"]

    def test_conda_path_object_uses_p(self):
        prefix = EnvSpec.resolve(Path("/opt/envs/at"), "conda").command_prefix()
        assert prefix[1:4] == ["run", "-p", "/opt/envs/at"]

    def test_no_capture_output_flag(self):
        prefix = EnvSpec.resolve("e", "conda").command_prefix(no_capture_output=True)
        assert "--no-capture-output" in prefix
        assert prefix[1:3] == ["run", "--no-capture-output"]


class TestEnvSpecMergeEnviron:
    def test_venv_injects_path_and_virtual_env(self, tmp_path: Path):
        venv = tmp_path / "venv"
        spec = EnvSpec.resolve(venv, "venv")
        merged = spec.merge_environ(base={"PATH": "/usr/bin", "HOME": "/home"})
        bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
        assert merged["PATH"].split(os.pathsep)[0] == str(bin_dir)
        assert merged["VIRTUAL_ENV"] == str(venv)
        assert merged["HOME"] == "/home"

    def test_extra_overrides(self):
        merged = EnvSpec.system().merge_environ(
            base={"A": "1"}, extra={"A": "2", "B": "3"}
        )
        assert merged["A"] == "2"
        assert merged["B"] == "3"

    def test_conda_does_not_mutate_path(self):
        base = {"PATH": "/usr/bin"}
        merged = EnvSpec.resolve("e", "conda").merge_environ(base=base)
        assert merged["PATH"] == "/usr/bin"


class TestEnvSpecResolveExecutable:
    def test_absolute_existing_file(self, tmp_path: Path):
        exe = tmp_path / "tool"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        assert EnvSpec.system().resolve_executable(str(exe)) == str(exe)

    def test_venv_bin_lookup(self, tmp_path: Path):
        bin_dir = tmp_path / ("Scripts" if os.name == "nt" else "bin")
        bin_dir.mkdir()
        tool = bin_dir / "antechamber"
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)
        found = EnvSpec.resolve(tmp_path, "venv").resolve_executable("antechamber")
        assert found == str(tool)

    def test_system_which(self):
        # `echo` is on PATH in all reasonable environments used for tests.
        found = EnvSpec.system().resolve_executable("echo")
        assert found is not None
