"""molpy.config: the four layers, their precedence, and where each value came from."""

from __future__ import annotations

from pathlib import Path

import pytest
from molcfg import FrozenConfigError, ValidationError

from molpy.config import (
    DEFAULTS,
    LAYERS,
    PROJECT_CONFIG_NAME,
    load_config,
    tool_settings,
    user_config_path,
)


@pytest.fixture
def home(tmp_path: Path) -> dict[str, str]:
    """An environment whose ``MOLCRAFTS_HOME`` is under this test's tmp dir."""
    return {"MOLCRAFTS_HOME": str(tmp_path / "home")}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    path = tmp_path / "project"
    path.mkdir()
    return path


def _user(environ: dict[str, str], text: str) -> Path:
    path = user_config_path(environ)
    path.write_text(text, encoding="utf-8")
    return path


def _project(project: Path, text: str) -> Path:
    path = project / PROJECT_CONFIG_NAME
    path.write_text(text, encoding="utf-8")
    return path


class TestLayers:
    def test_layer_order(self) -> None:
        assert LAYERS == ("defaults", "user", "project", "run")

    def test_user_config_lives_under_molcrafts_home(self, home) -> None:
        path = user_config_path(home)
        assert path == Path(home["MOLCRAFTS_HOME"]) / "molpy" / "config" / "config.toml"
        assert path.parent.is_dir()

    def test_defaults_alone(self, home, project) -> None:
        config = load_config(project_dir=project, environ=home)
        assert config["engine.gromacs.executable"] == "gmx"
        assert config.meta("engine.gromacs.executable")["source"] == "defaults"
        assert config.to_dict() == DEFAULTS

    def test_each_layer_overrides_the_one_before(self, home, project) -> None:
        _user(
            home,
            '[engine.lammps]\nexecutable = "lmp_user"\ntimeout = 10\n'
            'launcher = ["mpirun"]\n',
        )
        _project(project, '[engine.lammps]\nexecutable = "lmp_project"\ntimeout = 20\n')
        config = load_config(
            {"engine": {"lammps": {"executable": "lmp_run"}}},
            project_dir=project,
            environ=home,
        )
        assert config["engine.lammps.executable"] == "lmp_run"
        assert config["engine.lammps.timeout"] == 20
        assert config["engine.lammps.launcher"] == ["mpirun"]
        assert config.meta("engine.lammps.executable") == {
            "source": "run",
            "history": ("defaults", "user", "project", "run"),
        }
        assert config.meta("engine.lammps.timeout")["source"] == "project"
        assert config.meta("engine.lammps.launcher")["source"] == "user"

    def test_an_absent_layer_is_skipped(self, home, project) -> None:
        _project(project, '[wrapper]\nenv = "AmberTools25"\nenv_manager = "conda"\n')
        config = load_config(project_dir=project, environ=home)
        assert config.meta("wrapper.env")["history"] == ("defaults", "project")

    def test_tables_merge_key_by_key(self, home, project) -> None:
        _user(home, '[engine.env_vars]\nOMP_NUM_THREADS = "8"\n')
        _project(project, '[engine.env_vars]\nOMP_PLACES = "cores"\n')
        config = load_config(project_dir=project, environ=home)
        assert config["engine.env_vars"].to_dict() == {
            "OMP_NUM_THREADS": "8",
            "OMP_PLACES": "cores",
        }

    def test_the_project_is_the_working_directory_by_default(
        self, home, project, monkeypatch
    ) -> None:
        _project(project, '[engine.cp2k]\nexecutable = "cp2k.popt"\n')
        monkeypatch.chdir(project)
        config = load_config(environ=home)
        assert config["engine.cp2k.executable"] == "cp2k.popt"
        assert config.meta("engine.cp2k.executable")["source"] == "project"

    def test_molcrafts_home_is_read_from_the_process_environment(
        self, tmp_path, project, monkeypatch
    ) -> None:
        monkeypatch.setenv("MOLCRAFTS_HOME", str(tmp_path / "elsewhere"))
        _user(
            {"MOLCRAFTS_HOME": str(tmp_path / "elsewhere")},
            '[conda]\nexecutable = "/c"\n',
        )
        config = load_config(project_dir=project)
        assert config["conda.executable"] == "/c"
        assert config.meta("conda.executable")["source"] == "user"

    def test_the_config_is_frozen(self, home, project) -> None:
        config = load_config(project_dir=project, environ=home)
        with pytest.raises(FrozenConfigError):
            config["engine.timeout"] = 5


class TestValidation:
    def test_an_unknown_key_is_refused_with_its_layer(self, home, project) -> None:
        path = _project(project, '[engine.lammps]\nexecutible = "lmp"\n')
        with pytest.raises(ValidationError) as caught:
            load_config(project_dir=project, environ=home)
        message = str(caught.value)
        assert "engine.lammps.executible: unexpected field" in message
        assert str(path) in message

    def test_an_unknown_tool_is_refused(self, home, project) -> None:
        with pytest.raises(ValidationError, match="engine.namd: unexpected field"):
            load_config(
                {"engine": {"namd": {"executable": "namd2"}}},
                project_dir=project,
                environ=home,
            )

    def test_a_wrong_type_is_refused(self, home, project) -> None:
        with pytest.raises(ValidationError, match="engine.launcher"):
            load_config(
                {"engine": {"launcher": "mpirun -np 4"}},
                project_dir=project,
                environ=home,
            )

    def test_an_unknown_env_manager_is_refused(self, home, project) -> None:
        with pytest.raises(ValidationError, match="wrapper.env_manager"):
            load_config(
                {"wrapper": {"env": "x", "env_manager": "uv"}},
                project_dir=project,
                environ=home,
            )

    def test_wrappers_have_no_launcher(self, home, project) -> None:
        with pytest.raises(ValidationError, match="wrapper.tleap.launcher"):
            load_config(
                {"wrapper": {"tleap": {"launcher": ["srun"]}}},
                project_dir=project,
                environ=home,
            )


class TestToolSettings:
    def test_package_defaults(self, home, project) -> None:
        settings = tool_settings(
            load_config(project_dir=project, environ=home), "engine.gromacs"
        )
        assert settings.tool == "engine.gromacs"
        assert settings.executable == "gmx"
        assert settings.env is None
        assert settings.env_manager is None
        assert settings.conda_executable == "conda"
        assert settings.launcher == ()
        assert settings.env_vars == {}
        assert settings.timeout is None
        assert settings.sources == {
            "env": ("defaults", "engine.env"),
            "env_manager": ("defaults", "engine.env_manager"),
            "env_vars": ("defaults", "engine.env_vars"),
            "timeout": ("defaults", "engine.timeout"),
            "launcher": ("defaults", "engine.launcher"),
            "executable": ("defaults", "engine.gromacs.executable"),
            "conda_executable": ("defaults", "conda.executable"),
        }

    def test_a_tool_key_beats_its_group_key(self, home, project) -> None:
        _user(home, '[engine]\nlauncher = ["srun"]\ntimeout = 100\n')
        _project(project, '[engine.lammps]\nlauncher = ["mpirun", "-np", "8"]\n')
        config = load_config(project_dir=project, environ=home)
        lammps = tool_settings(config, "engine.lammps")
        cp2k = tool_settings(config, "engine.cp2k")
        assert lammps.launcher == ("mpirun", "-np", "8")
        assert lammps.sources["launcher"] == ("project", "engine.lammps.launcher")
        assert lammps.timeout == 100.0
        assert lammps.sources["timeout"] == ("user", "engine.timeout")
        assert cp2k.launcher == ("srun",)
        assert cp2k.sources["launcher"] == ("user", "engine.launcher")

    def test_a_run_override_can_clear_a_group_environment(self, home, project) -> None:
        _project(project, '[wrapper]\nenv = "AmberTools25"\nenv_manager = "conda"\n')
        config = load_config(
            {"wrapper": {"sander": {"env": None, "env_manager": None}}},
            project_dir=project,
            environ=home,
        )
        assert tool_settings(config, "wrapper.tleap").env == "AmberTools25"
        sander = tool_settings(config, "wrapper.sander")
        assert sander.env is None
        assert sander.sources["env"] == ("run", "wrapper.sander.env")

    def test_wrappers_have_no_launcher(self, home, project) -> None:
        settings = tool_settings(
            load_config(project_dir=project, environ=home), "wrapper.tleap"
        )
        assert settings.launcher == ()
        assert "launcher" not in settings.sources

    def test_conda_executable(self, home, project) -> None:
        _user(home, '[conda]\nexecutable = "/opt/miniforge/bin/conda"\n')
        settings = tool_settings(
            load_config(project_dir=project, environ=home), "wrapper.antechamber"
        )
        assert settings.conda_executable == "/opt/miniforge/bin/conda"
        assert settings.sources["conda_executable"] == ("user", "conda.executable")

    @pytest.mark.parametrize("tool", ["engine.namd", "wrapper.lammps", "lammps", ""])
    def test_an_unknown_tool_raises(self, home, project, tool) -> None:
        config = load_config(project_dir=project, environ=home)
        with pytest.raises(KeyError, match="unknown tool"):
            tool_settings(config, tool)

    @pytest.mark.parametrize("timeout", [0, -5])
    def test_a_timeout_must_be_positive(self, home, project, timeout) -> None:
        config = load_config(
            {"engine": {"timeout": timeout}}, project_dir=project, environ=home
        )
        with pytest.raises(ValueError, match="engine.timeout"):
            tool_settings(config, "engine.lammps")


class TestLogging:
    def test_loading_logs_the_layers_read(self, home, project, molpy_records) -> None:
        _project(project, "[engine]\ntimeout = 5\n")
        load_config({"engine": {"timeout": 6}}, project_dir=project, environ=home)
        (record,) = (r for r in molpy_records if r.message == "config loaded")
        assert record.logger_name == "molpy.config"
        assert record.extra["layers"] == ["defaults", "project", "run"]
        assert record.extra["project_config"] == str(project / PROJECT_CONFIG_NAME)

    def test_resolving_logs_each_settings_source(
        self, home, project, molpy_records
    ) -> None:
        config = load_config(
            {"engine": {"lammps": {"timeout": 9}}}, project_dir=project, environ=home
        )
        tool_settings(config, "engine.lammps")
        (record,) = (r for r in molpy_records if r.message == "tool settings resolved")
        assert record.extra["tool"] == "engine.lammps"
        assert record.extra["sources"]["timeout"] == "run:engine.lammps.timeout"
        assert record.extra["sources"]["env"] == "defaults:engine.env"
