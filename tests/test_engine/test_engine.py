"""Unit tests for the engine base class — configuration, scripts, mocked subprocess."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from molpy.config import load_config
from molpy.engine import Cp2kEngine, GromacsEngine, LammpsEngine, OpenmmEngine, Script


def _completed(returncode: int = 0) -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = ""
    result.stderr = ""
    return result


def _lammps(overrides=None, **kwargs) -> LammpsEngine:
    lammps = {"executable": "lmp", **(overrides or {})}
    config = load_config({"engine": {"lammps": lammps}})
    return LammpsEngine(config=config, check_executable=False, **kwargs)


class TestEngineSettings:
    """An engine runs with what its resolved configuration says."""

    def test_package_defaults(self):
        engine = _lammps()
        assert engine.executable == "lmp"
        assert engine.work_dir is None
        assert engine.launcher == []
        assert engine.env_vars == {}
        assert engine.timeout is None
        assert engine.environment.is_system
        assert engine.settings.tool == "engine.lammps"

    @pytest.mark.parametrize(
        ("cls", "tool", "executable"),
        [
            (LammpsEngine, "lammps", "lmp_mpi"),
            (GromacsEngine, "gromacs", "gmx_mpi"),
            (OpenmmEngine, "openmm", "python3"),
            (Cp2kEngine, "cp2k", "cp2k.popt"),
        ],
    )
    def test_every_engine_reads_its_own_table(self, cls, tool, executable):
        config = load_config(
            {
                "engine": {
                    tool: {
                        "executable": executable,
                        "launcher": ["srun", "-n", "4"],
                        "env_vars": {"OMP_NUM_THREADS": "2"},
                        "timeout": 60,
                    }
                }
            }
        )
        engine = cls(config=config, check_executable=False)
        assert engine.executable == executable
        assert engine.launcher == ["srun", "-n", "4"]
        assert engine.env_vars == {"OMP_NUM_THREADS": "2"}
        assert engine.timeout == 60.0
        assert engine.settings.sources["executable"] == (
            "run",
            f"engine.{tool}.executable",
        )

    def test_group_settings_reach_every_engine(self):
        config = load_config({"engine": {"env": "md", "env_manager": "conda"}})
        for cls in (LammpsEngine, GromacsEngine, OpenmmEngine, Cp2kEngine):
            engine = cls(config=config, check_executable=False)
            assert engine.environment.env == "md"
            assert engine.environment.env_manager == "conda"

    def test_default_executables(self):
        assert GromacsEngine(check_executable=False).executable == "gmx"
        assert OpenmmEngine(check_executable=False).executable == "python"
        assert Cp2kEngine(check_executable=False).executable == "cp2k.psmp"

    def test_incomplete_environment_raises(self):
        with pytest.raises(ValueError, match="incomplete"):
            _lammps({"env": "myenv"})
        with pytest.raises(ValueError, match="incomplete"):
            _lammps({"env_manager": "conda"})

    def test_conda_environment_prefixes_the_command(self):
        config = load_config(
            {
                "conda": {"executable": "/opt/conda/bin/conda"},
                "engine": {
                    "lammps": {
                        "executable": "lmp",
                        "env": "lammps-env",
                        "env_manager": "conda",
                        "launcher": ["mpirun", "-np", "2"],
                    }
                },
            }
        )
        engine = LammpsEngine(config=config, check_executable=False)
        assert engine._build_full_command(["-in", "in.lmp"]) == [
            "/opt/conda/bin/conda",
            "run",
            "--no-capture-output",
            "-n",
            "lammps-env",
            "mpirun",
            "-np",
            "2",
            "lmp",
            "-in",
            "in.lmp",
        ]

    def test_check_executable_missing(self):
        with pytest.raises(FileNotFoundError, match="engine.lammps.executable"):
            LammpsEngine(
                config=load_config(
                    {"engine": {"lammps": {"executable": "no_such_lammps_xyz123"}}}
                )
            )

    def test_repr(self, tmp_path):
        engine = _lammps({"env": "myenv", "env_manager": "conda"}, workdir=tmp_path)
        text = repr(engine)
        assert "LammpsEngine" in text
        assert "lmp" in text
        assert str(tmp_path) in text
        assert "myenv" in text


class TestEngineRun:
    """engine.run writes scripts; subprocess is mocked — never a real binary."""

    def test_run_no_scripts_raises(self):
        with pytest.raises(ValueError, match="At least one script is required"):
            _lammps().run()

    def test_run_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one script is required"):
            _lammps().run([])

    def test_run_with_script_saves_files(self, tmp_path):
        script = Script.from_text("input", "units real\natom_style full\n")
        engine = _lammps(workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()) as mock_run:
            engine.run(script, capture_output=True, check=False)
        assert (tmp_path / "input.lmp").exists()
        assert mock_run.call_args.args[0][0] == "lmp"

    def test_run_with_string(self, tmp_path):
        engine = _lammps(workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            engine.run("units real\n", capture_output=True, check=False)
        assert (
            (tmp_path / "input.lmp")
            .read_text(encoding="utf-8")
            .startswith("units real")
        )

    def test_run_with_path(self, tmp_path):
        script_file = tmp_path / "my_script.lmp"
        script_file.write_text("units real\natom_style full\n", encoding="utf-8")
        engine = _lammps(workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            engine.run(script_file, capture_output=True, check=False)
        assert len(engine.scripts) == 1
        assert engine.scripts[0].path.name == "my_script.lmp"

    def test_run_with_multiple_scripts(self, tmp_path):
        script1 = Script.from_text("main", "units real\n")
        script1.tags.add("input")
        script2 = Script.from_text("data", "# data file\n")
        engine = _lammps(workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            engine.run([script1, script2], capture_output=True, check=False)
        assert len(engine.scripts) == 2
        assert engine.input_script == script1
        assert (tmp_path / "main.lmp").read_text(encoding="utf-8") == "units real\n"
        assert (tmp_path / "data.lmp").read_text(encoding="utf-8") == "# data file\n"

    def test_run_with_workdir_override(self, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        engine = _lammps(workdir=first)
        with patch("subprocess.run", return_value=_completed()):
            engine.run("units real\n", workdir=second, capture_output=True, check=False)
        assert not (first / "input.lmp").exists()
        assert (second / "input.lmp").exists()

    def test_run_passes_the_configured_timeout_and_env_vars(self, tmp_path):
        engine = _lammps(
            {"timeout": 90, "env_vars": {"OMP_NUM_THREADS": "4"}}, workdir=tmp_path
        )
        with patch("subprocess.run", return_value=_completed()) as mock_run:
            engine.run("units real\n", capture_output=True, check=False)
        assert mock_run.call_args.kwargs["timeout"] == 90.0
        assert mock_run.call_args.kwargs["env"]["OMP_NUM_THREADS"] == "4"

    def test_run_has_no_timeout_unless_configured(self, tmp_path):
        engine = _lammps(workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()) as mock_run:
            engine.run("units real\n", capture_output=True, check=False)
        assert mock_run.call_args.kwargs["timeout"] is None
        assert mock_run.call_args.kwargs["env"] is None  # inherits the parent's


class TestEngineLogging:
    """Every subprocess an engine starts is a structured mollog record."""

    def test_a_finished_run_logs_start_and_finish(self, tmp_path, molpy_records):
        engine = _lammps({"launcher": ["mpirun", "-np", "2"]}, workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            engine.run("units real\n", check=False)
        started, finished = (
            r for r in molpy_records if r.logger_name == "molpy.engine.lammps"
        )
        assert started.message == "process started"
        assert str(started.level) == "DEBUG"
        assert finished.message == "process finished"
        assert str(finished.level) == "INFO"
        expected = ["mpirun", "-np", "2", "lmp", "-in", "input.lmp"]
        assert finished.extra["command"][: len(expected)] == expected
        assert finished.extra["cwd"] == str(tmp_path)
        assert finished.extra["returncode"] == 0
        assert finished.extra["elapsed_s"] >= 0.0

    def test_a_failed_run_logs_an_error(self, tmp_path, molpy_records):
        engine = _lammps(workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed(3)):
            engine.run("units real\n", check=False)
        (failed,) = (r for r in molpy_records if r.message == "process failed")
        assert str(failed.level) == "ERROR"
        assert failed.extra["returncode"] == 3

    def test_a_raising_run_logs_and_reraises(self, tmp_path, molpy_records):
        engine = _lammps(workdir=tmp_path)
        error = subprocess.CalledProcessError(2, ["lmp"])
        with (
            patch("subprocess.run", side_effect=error),
            pytest.raises(subprocess.CalledProcessError),
        ):
            engine.run("units real\n")
        (failed,) = (r for r in molpy_records if r.message == "process failed")
        assert failed.extra["returncode"] == 2

    def test_a_timeout_is_logged(self, tmp_path, molpy_records):
        engine = _lammps({"timeout": 5}, workdir=tmp_path)
        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("lmp", 5)),
            pytest.raises(subprocess.TimeoutExpired),
        ):
            engine.run("units real\n")
        (record,) = (r for r in molpy_records if r.message == "process timed out")
        assert record.extra["timeout_s"] == 5.0

    def test_a_missing_executable_is_logged(self, tmp_path, molpy_records):
        engine = _lammps(workdir=tmp_path)
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("lmp")),
            pytest.raises(FileNotFoundError),
        ):
            engine.run("units real\n")
        (record,) = (r for r in molpy_records if r.message == "process not started")
        assert "lmp" in record.extra["error"]

    def test_gromacs_logs_each_step(self, tmp_path, molpy_records):
        config = load_config({"engine": {"gromacs": {"launcher": ["mpirun"]}}})
        engine = GromacsEngine(config=config, check_executable=False)
        script = Script.from_text("em", "integrator = steep\n")
        with patch("subprocess.run", return_value=_completed()):
            engine.run(script, workdir=tmp_path)
        finished = [r for r in molpy_records if r.message == "process finished"]
        assert [r.logger_name for r in finished] == ["molpy.engine.gromacs"] * 2
        assert [r.extra["step"] for r in finished] == ["grompp", "mdrun"]
        assert finished[1].extra["command"][:2] == ["mpirun", "gmx"]

    @pytest.mark.parametrize(
        ("cls", "logger"),
        [
            (Cp2kEngine, "molpy.engine.cp2k"),
            (OpenmmEngine, "molpy.engine.openmm"),
        ],
    )
    def test_each_engine_logs_under_its_namespace(
        self, cls, logger, tmp_path, molpy_records
    ):
        engine = cls(check_executable=False, workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            engine.run("input\n", check=False)
        assert {
            r.logger_name for r in molpy_records if r.message.startswith("process")
        } == {logger}


def test_cp2k_identity():
    engine = Cp2kEngine(check_executable=False)
    assert engine.name == "CP2K"
    assert engine._get_default_extension() == ".inp"


def test_lammps_identity():
    engine = _lammps()
    assert engine.name == "LAMMPS"
    assert engine._get_default_extension() == ".lmp"
    assert Path(engine.executable).name == "lmp"
