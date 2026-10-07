"""Unit tests for :class:`molpy.wrapper.EnvironmentSpec` — the environment infrastructure."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from molpy.wrapper import EnvironmentSpec


class TestEnvironmentSpecResolve:
    def test_system_default(self):
        spec = EnvironmentSpec.resolve()
        assert spec.is_system
        assert spec.env is None
        assert spec.env_manager is None
        assert EnvironmentSpec.system() == spec

    def test_both_or_neither(self):
        with pytest.raises(ValueError, match="incomplete"):
            EnvironmentSpec.resolve(env="AmberTools25")
        with pytest.raises(ValueError, match="incomplete"):
            EnvironmentSpec.resolve(env_manager="conda")

    def test_conda_name(self):
        spec = EnvironmentSpec.resolve("AmberTools25", "conda")
        assert spec.env == "AmberTools25"
        assert isinstance(spec.env, str)
        assert spec.env_manager == "conda"
        assert not spec.is_system

    def test_conda_prefix_str_becomes_path(self):
        spec = EnvironmentSpec.resolve("/opt/conda/envs/at", "conda")
        assert spec.env_manager == "conda"
        assert isinstance(spec.env, Path)
        assert spec.env == Path("/opt/conda/envs/at")
        prefix = spec.command_prefix()
        assert prefix[1:3] == ["run", "-p"]
        assert Path(prefix[3]) == Path("/opt/conda/envs/at")

    def test_venv_normalises_to_path(self):
        for name in ("venv", "Venv"):
            spec = EnvironmentSpec.resolve("/path/to/.venv", name)
            assert spec.env_manager == "venv"
            assert isinstance(spec.env, Path)
            assert spec.env == Path("/path/to/.venv")

    def test_venv_has_one_spelling(self):
        for alias in ("pip", "virtualenv"):
            with pytest.raises(ValueError, match="Unsupported env_manager"):
                EnvironmentSpec.resolve("/path/to/.venv", alias)

    def test_unsupported_manager(self):
        with pytest.raises(ValueError, match="Unsupported env_manager"):
            EnvironmentSpec.resolve("x", "uv")


class TestEnvironmentSpecCommandPrefix:
    def test_system_and_venv_have_empty_prefix(self):
        assert EnvironmentSpec.system().command_prefix() == []
        assert EnvironmentSpec.resolve("/tmp/v", "venv").command_prefix() == []

    def test_conda_name_uses_n(self):
        prefix = EnvironmentSpec.resolve("AmberTools25", "conda").command_prefix()
        assert prefix[1:4] == ["run", "-n", "AmberTools25"]

    def test_conda_path_object_uses_p(self):
        env_path = Path("/opt/envs/at")
        prefix = EnvironmentSpec.resolve(env_path, "conda").command_prefix()
        assert prefix[1:3] == ["run", "-p"]
        # Internal storage is Path; argv boundary is str(path) (OS-native).
        assert Path(prefix[3]) == env_path
        assert prefix[3] == str(env_path)

    def test_no_capture_output_flag(self):
        prefix = EnvironmentSpec.resolve("e", "conda").command_prefix(
            no_capture_output=True
        )
        assert "--no-capture-output" in prefix
        assert prefix[1:3] == ["run", "--no-capture-output"]


class TestEnvironmentSpecMergeEnviron:
    def test_venv_injects_path_and_virtual_env(self, tmp_path: Path):
        venv = tmp_path / "venv"
        spec = EnvironmentSpec.resolve(venv, "venv")
        assert isinstance(spec.env, Path)
        merged = spec.merge_environ(base={"PATH": "/usr/bin", "HOME": "/home"})
        bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
        assert merged["PATH"].split(os.pathsep)[0] == str(bin_dir)
        assert merged["VIRTUAL_ENV"] == str(venv)
        assert Path(merged["VIRTUAL_ENV"]) == venv
        assert merged["HOME"] == "/home"

    def test_venv_str_input_becomes_path(self, tmp_path: Path):
        venv = tmp_path / "venv"
        spec = EnvironmentSpec.resolve(str(venv), "venv")
        assert isinstance(spec.env, Path)
        assert spec.env == venv

    def test_extra_overrides(self):
        merged = EnvironmentSpec.system().merge_environ(
            base={"A": "1"}, extra={"A": "2", "B": "3"}
        )
        assert merged["A"] == "2"
        assert merged["B"] == "3"

    def test_conda_does_not_mutate_path(self):
        base = {"PATH": "/usr/bin"}
        merged = EnvironmentSpec.resolve("e", "conda").merge_environ(base=base)
        assert merged["PATH"] == "/usr/bin"


class TestEnvironmentSpecResolveExecutable:
    def test_absolute_existing_file(self, tmp_path: Path):
        exe = tmp_path / "tool"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        assert EnvironmentSpec.system().resolve_executable(str(exe)) == str(
            exe.resolve()
        )
        assert EnvironmentSpec.system().resolve_executable(exe) == str(exe.resolve())

    def test_venv_bin_lookup(self, tmp_path: Path):
        bin_dir = tmp_path / ("Scripts" if os.name == "nt" else "bin")
        bin_dir.mkdir()
        tool = bin_dir / "antechamber"
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)
        found = EnvironmentSpec.resolve(tmp_path, "venv").resolve_executable(
            "antechamber"
        )
        assert found == str(tool.resolve())

    def test_system_which(self):
        # `echo` is on PATH in all reasonable environments used for tests.
        found = EnvironmentSpec.system().resolve_executable("echo")
        assert found is not None
