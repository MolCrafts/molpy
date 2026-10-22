"""Unit tests for the engine base class — configuration, scripts, mocked subprocess."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import molpy as mp


def _completed(returncode: int = 0) -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = ""
    result.stderr = ""
    return result


def _lammps(overrides=None, **kwargs) -> mp.engine.LammpsEngine:
    lammps = {"executable": "lmp", **(overrides or {})}
    config = mp.config.load_config({"engine": {"lammps": lammps}})
    return mp.engine.LammpsEngine(config=config, check_executable=False, **kwargs)


class TestEngineSettings:
    """An engine runs with what its resolved configuration says."""

    def test_package_defaults(self):
        engine = _lammps()
        assert engine.executable == "lmp"
        assert engine.key == "engine.lammps"
        assert engine.workdir is None
        assert engine.launcher == []
        assert engine.env_vars == {}
        assert engine.timeout is None
        assert engine.environment.is_system
        assert engine.settings.tool == "engine.lammps"

    @pytest.mark.parametrize(
        ("cls", "tool", "executable"),
        [
            (mp.engine.LammpsEngine, "lammps", "lmp_mpi"),
            (mp.engine.GromacsEngine, "gromacs", "gmx_mpi"),
            (mp.engine.OpenmmEngine, "openmm", "python3"),
            (mp.engine.Cp2kEngine, "cp2k", "cp2k.popt"),
        ],
    )
    def test_every_engine_reads_its_own_table(self, cls, tool, executable):
        config = mp.config.load_config(
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
        config = mp.config.load_config(
            {"engine": {"env": "md", "env_manager": "conda"}}
        )
        for cls in (
            mp.engine.LammpsEngine,
            mp.engine.GromacsEngine,
            mp.engine.OpenmmEngine,
            mp.engine.Cp2kEngine,
        ):
            engine = cls(config=config, check_executable=False)
            assert engine.environment.env == "md"
            assert engine.environment.env_manager == "conda"

    def test_default_executables(self):
        assert mp.engine.GromacsEngine(check_executable=False).executable == "gmx"
        assert mp.engine.OpenmmEngine(check_executable=False).executable == "python"
        assert mp.engine.Cp2kEngine(check_executable=False).executable == "cp2k.psmp"

    def test_incomplete_environment_raises(self):
        with pytest.raises(ValueError, match="incomplete"):
            _lammps({"env": "myenv"})
        with pytest.raises(ValueError, match="incomplete"):
            _lammps({"env_manager": "conda"})

    def test_conda_environment_prefixes_the_command(self):
        config = mp.config.load_config(
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
        engine = mp.engine.LammpsEngine(config=config, check_executable=False)
        assert engine.command(["-in", "in.lmp"]) == [
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
        with pytest.raises(
            mp.tool.ToolError, match="engine.lammps.executable"
        ) as raised:
            mp.engine.LammpsEngine(
                config=mp.config.load_config(
                    {"engine": {"lammps": {"executable": "no_such_lammps_xyz123"}}}
                )
            )
        assert raised.value.tool == "engine.lammps"

    def test_repr(self, tmp_path):
        engine = _lammps({"env": "myenv", "env_manager": "conda"}, workdir=tmp_path)
        text = repr(engine)
        assert "LammpsEngine" in text
        assert "lmp" in text
        assert repr(str(tmp_path)) in text
        assert "myenv" in text


class TestEngineRun:
    """engine.run writes scripts; subprocess is mocked — never a real binary."""

    def test_run_needs_scripts(self):
        with pytest.raises(TypeError):
            _lammps().run()

    def test_run_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one script is required"):
            _lammps().run([])

    def test_run_rejects_unknown_keywords(self, tmp_path):
        with pytest.raises(TypeError, match="timeout"):
            _lammps(workdir=tmp_path).run("units real\n", timeout=60)

    def test_run_keeps_no_state(self, tmp_path):
        engine = _lammps(workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            engine.run("units real\n")
        assert not hasattr(engine, "scripts")
        assert not hasattr(engine, "input_script")
        with pytest.raises(TypeError):
            engine.run()

    def test_run_does_not_change_the_callers_script(self, tmp_path):
        source = tmp_path / "src" / "in.lmp"
        source.parent.mkdir()
        source.write_text("units real\n", encoding="utf-8")
        script = mp.engine.Script.from_path(source)
        with patch("subprocess.run", return_value=_completed()):
            _lammps().run(script, workdir=tmp_path / "run")
        assert script.path == source
        assert (tmp_path / "run" / "in.lmp").read_text(
            encoding="utf-8"
        ) == "units real\n"

    def test_run_with_script_saves_files(self, tmp_path):
        script = mp.engine.Script.from_text("input", "units real\natom_style full\n")
        engine = _lammps(workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()) as mock_run:
            engine.run(script, capture_output=True)
        assert (tmp_path / "input.lmp").exists()
        assert mock_run.call_args.args[0][0] == "lmp"

    def test_run_with_string(self, tmp_path):
        engine = _lammps(workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            engine.run("units real\n", capture_output=True)
        assert (
            (tmp_path / "input.lmp")
            .read_text(encoding="utf-8")
            .startswith("units real")
        )

    def test_run_with_path_writes_it_verbatim(self, tmp_path):
        script_file = tmp_path / "src" / "my_script.lmp"
        script_file.parent.mkdir()
        text = "  units real\natom_style full"
        script_file.write_text(text, encoding="utf-8")
        engine = _lammps(workdir=tmp_path / "run")
        with patch("subprocess.run", return_value=_completed()) as mock_run:
            engine.run(script_file, capture_output=True)
        assert (tmp_path / "run" / "my_script.lmp").read_text(encoding="utf-8") == text
        assert mock_run.call_args.args[0][:3] == ["lmp", "-in", "my_script.lmp"]

    def test_run_with_multiple_scripts_runs_the_first(self, tmp_path):
        script1 = mp.engine.Script.from_text("main", "units real\n")
        script2 = mp.engine.Script.from_text("data", "# data file\n")
        engine = _lammps(workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()) as mock_run:
            engine.run([script1, script2], capture_output=True)
        assert mock_run.call_args.args[0][:3] == ["lmp", "-in", "main.lmp"]
        assert (tmp_path / "main.lmp").read_text(encoding="utf-8") == "units real\n"
        assert (tmp_path / "data.lmp").read_text(encoding="utf-8") == "# data file\n"

    def test_run_with_workdir_override(self, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        engine = _lammps(workdir=first)
        with patch("subprocess.run", return_value=_completed()):
            engine.run("units real\n", workdir=second, capture_output=True)
        assert not (first / "input.lmp").exists()
        assert (second / "input.lmp").exists()

    def test_run_passes_the_configured_timeout_and_env_vars(self, tmp_path):
        engine = _lammps(
            {"timeout": 90, "env_vars": {"OMP_NUM_THREADS": "4"}}, workdir=tmp_path
        )
        with patch("subprocess.run", return_value=_completed()) as mock_run:
            engine.run("units real\n", capture_output=True)
        assert mock_run.call_args.kwargs["timeout"] == 90.0
        assert mock_run.call_args.kwargs["env"]["OMP_NUM_THREADS"] == "4"

    def test_run_has_no_timeout_unless_configured(self, tmp_path):
        engine = _lammps(workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()) as mock_run:
            engine.run("units real\n", capture_output=True)
        assert mock_run.call_args.kwargs["timeout"] is None
        assert mock_run.call_args.kwargs["env"] is None  # inherits the parent's


class TestEngineLogging:
    """Every subprocess an engine starts is a structured mollog record."""

    def test_a_finished_run_logs_start_and_finish(self, tmp_path, molpy_records):
        engine = _lammps({"launcher": ["mpirun", "-np", "2"]}, workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            engine.run("units real\n")
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
            assert engine.run("units real\n", check=False).returncode == 3
        (failed,) = (r for r in molpy_records if r.message == "process failed")
        assert str(failed.level) == "ERROR"
        assert failed.extra["returncode"] == 3

    def test_a_failed_run_raises_with_the_log_tail(self, tmp_path, molpy_records):
        """-screen none: the failure's tail comes from log.lammps."""
        engine = _lammps(workdir=tmp_path)

        def fails(argv, *, cwd, **_):
            (cwd / "log.lammps").write_text(
                "LAMMPS (2 Aug 2023)\nERROR: Unknown command: bogus\n",
                encoding="utf-8",
            )
            return _completed(1)

        with (
            patch("subprocess.run", side_effect=fails),
            pytest.raises(
                mp.tool.ToolError, match="ERROR: Unknown command: bogus"
            ) as raised,
        ):
            engine.run("bogus\n")
        assert raised.value.tool == "engine.lammps"
        assert raised.value.returncode == 1
        (failed,) = (r for r in molpy_records if r.message == "process failed")
        assert failed.extra["returncode"] == 1

    def test_a_timeout_is_logged(self, tmp_path, molpy_records):
        engine = _lammps({"timeout": 5}, workdir=tmp_path)
        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("lmp", 5)),
            pytest.raises(mp.tool.ToolError, match="timed out after 5 s"),
        ):
            engine.run("units real\n")
        (record,) = (r for r in molpy_records if r.message == "process timed out")
        assert record.extra["timeout_s"] == 5.0

    def test_a_missing_executable_is_logged(self, tmp_path, molpy_records):
        engine = _lammps(workdir=tmp_path)
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("lmp")),
            pytest.raises(mp.tool.ToolError, match="could not start"),
        ):
            engine.run("units real\n")
        (record,) = (r for r in molpy_records if r.message == "process not started")
        assert "lmp" in record.extra["error"]

    def test_gromacs_logs_each_step(self, tmp_path, molpy_records):
        config = mp.config.load_config(
            {"engine": {"gromacs": {"launcher": ["mpirun"]}}}
        )
        engine = mp.engine.GromacsEngine(config=config, check_executable=False)
        script = mp.engine.Script.from_text("em", "integrator = steep\n")
        with patch("subprocess.run", return_value=_completed()):
            engine.run(script, workdir=tmp_path)
        finished = [r for r in molpy_records if r.message == "process finished"]
        assert [r.logger_name for r in finished] == ["molpy.engine.gromacs"] * 2
        assert [r.extra["step"] for r in finished] == ["grompp", "mdrun"]
        assert finished[1].extra["command"][:2] == ["mpirun", "gmx"]

    @pytest.mark.parametrize(
        ("cls", "logger"),
        [
            (mp.engine.Cp2kEngine, "molpy.engine.cp2k"),
            (mp.engine.OpenmmEngine, "molpy.engine.openmm"),
        ],
    )
    def test_each_engine_logs_under_its_namespace(
        self, cls, logger, tmp_path, molpy_records
    ):
        engine = cls(check_executable=False, workdir=tmp_path)
        with patch("subprocess.run", return_value=_completed()):
            engine.run("input\n")
        assert {
            r.logger_name for r in molpy_records if r.message.startswith("process")
        } == {logger}


def test_cp2k_identity():
    engine = mp.engine.Cp2kEngine(check_executable=False)
    assert engine.key == "engine.cp2k"
    assert engine._get_default_extension() == ".inp"


def test_lammps_identity():
    engine = _lammps()
    assert engine.key == "engine.lammps"
    assert engine._get_default_extension() == ".lmp"
    assert Path(engine.executable).name == "lmp"


# ---------------------------------------------------------------------------
# One input-writer contract
# ---------------------------------------------------------------------------


def test_every_deck_names_its_runnable_input(tmp_path, water, tip3p):
    import molpy as mp

    tip3p.get_style("pair", "lj/cut")["cutoff"] = 10.0
    tip3p.def_style("pair", "coul/cut", {"cutoff": 10.0})
    o, h1, h2 = list(water.atoms)
    ow = tip3p.get_style("atom", "full").get_type_by_name("OW")
    hw = tip3p.get_style("atom", "full").get_type_by_name("HW")
    tip3p.def_style("angle", "harmonic").def_type(
        "HW-OW-HW", hw, ow, hw, k=55.0, theta0=104.52
    )
    water.add_angle(h1, o, h2, type="HW-OW-HW")
    frame = water.to_frame()
    frame.box = mp.Box(mp.core.Cell.lengths([30.0] * 3))
    for engine_cls in (
        mp.engine.LammpsEngine,
        mp.engine.GromacsEngine,
        mp.engine.OpenmmEngine,
    ):
        out = tmp_path / engine_cls.tool
        paths = engine_cls(check_executable=False).generate_inputs(frame, tip3p, out)
        assert paths["input"].is_file()
        assert paths["input"].parent == out
