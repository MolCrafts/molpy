"""Unit tests for engine base classes."""

import tempfile
from pathlib import Path

import pytest

from molpy import Script
from molpy.engine import CP2KEngine, LAMMPSEngine


class TestEngineInit:
    """Test engine initialization."""

    def test_init_with_defaults(self):
        engine = LAMMPSEngine(executable="lmp", check_executable=False)
        assert engine.executable == "lmp"
        assert engine.work_dir is None
        assert engine.env_vars == {}
        assert engine.env is None
        assert engine.env_manager is None

    def test_init_with_workdir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LAMMPSEngine(
                executable="lmp", workdir=tmpdir, check_executable=False
            )
            assert engine.work_dir == Path(tmpdir)

    def test_init_with_env_vars(self):
        engine = LAMMPSEngine(
            executable="lmp",
            env_vars={"OMP_NUM_THREADS": "4"},
            check_executable=False,
        )
        assert engine.env_vars == {"OMP_NUM_THREADS": "4"}

    def test_init_env_validation(self):
        # Both None OK
        LAMMPSEngine(executable="lmp", check_executable=False)

        # Both set OK
        LAMMPSEngine(
            executable="lmp", env="myenv", env_manager="conda", check_executable=False
        )

        # Only env set -> raises
        with pytest.raises(ValueError, match="environment configuration is incomplete"):
            LAMMPSEngine(executable="lmp", env="myenv", check_executable=False)

        # Only env_manager set -> raises
        with pytest.raises(ValueError, match="environment configuration is incomplete"):
            LAMMPSEngine(executable="lmp", env_manager="conda", check_executable=False)

    def test_check_executable_missing(self):
        with pytest.raises(FileNotFoundError):
            LAMMPSEngine(executable="nonexistent_lammps_binary_xyz123")

    def test_repr(self):
        engine = LAMMPSEngine(executable="lmp", check_executable=False)
        assert "lmp" in repr(engine)
        assert "LAMMPSEngine" in repr(engine)

    def test_repr_with_workdir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LAMMPSEngine(
                executable="lmp", workdir=tmpdir, check_executable=False
            )
            repr_str = repr(engine)
            assert "lmp" in repr_str
            assert tmpdir in repr_str

    def test_repr_with_env(self):
        engine = LAMMPSEngine(
            executable="lmp",
            env="myenv",
            env_manager="conda",
            check_executable=False,
        )
        repr_str = repr(engine)
        assert "lmp" in repr_str
        assert "myenv" in repr_str
        assert "conda" in repr_str

    def test_merged_env_vars(self):
        engine = LAMMPSEngine(
            executable="lmp",
            env_vars={"VAR1": "value1"},
            check_executable=False,
        )
        merged = engine._merged_env({"VAR2": "value2"})
        assert merged["VAR1"] == "value1"
        assert merged["VAR2"] == "value2"

    def test_prepare_removed(self):
        """Engine.prepare() no longer exists."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)
        assert not hasattr(engine, "prepare")


class TestEngineRun:
    """Test engine run method."""

    def test_run_no_scripts_raises(self):
        engine = LAMMPSEngine(executable="lmp", check_executable=False)
        with pytest.raises(ValueError, match="At least one script is required"):
            engine.run()

    def test_run_empty_list_raises(self):
        engine = LAMMPSEngine(executable="lmp", check_executable=False)
        with pytest.raises(ValueError, match="At least one script is required"):
            engine.run([])

    def test_run_with_script_saves_files(self):
        script = Script.from_text("input", "units real\natom_style full\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LAMMPSEngine(
                executable="lmp", workdir=tmpdir, check_executable=False
            )

            try:
                engine.run(script, capture_output=True, check=False)
            except FileNotFoundError:
                pass  # LAMMPS not installed

            assert (Path(tmpdir) / "input.lmp").exists()

    def test_run_with_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LAMMPSEngine(
                executable="lmp", workdir=tmpdir, check_executable=False
            )

            try:
                engine.run("units real\n", capture_output=True, check=False)
            except FileNotFoundError:
                pass

            assert (Path(tmpdir) / "input.lmp").exists()

    def test_run_with_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            script_file = tmpdir_path / "my_script.lmp"
            script_file.write_text("units real\natom_style full\n")

            engine = LAMMPSEngine(
                executable="lmp", workdir=tmpdir, check_executable=False
            )

            try:
                result = engine.run(script_file, capture_output=True, check=False)
                assert len(engine.scripts) == 1
                assert engine.scripts[0].path.name == "my_script.lmp"
            except FileNotFoundError:
                pass

    def test_run_with_multiple_scripts(self):
        script1 = Script.from_text("main", "units real\n")
        script1.tags.add("input")
        script2 = Script.from_text("data", "# data file\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LAMMPSEngine(
                executable="lmp", workdir=tmpdir, check_executable=False
            )

            try:
                engine.run([script1, script2], capture_output=True, check=False)
                assert len(engine.scripts) == 2
                assert (Path(tmpdir) / "main.lmp").exists()
                assert (Path(tmpdir) / "data.lmp").exists()
                assert engine.input_script == script1
            except FileNotFoundError:
                pass

    def test_run_with_workdir_override(self):
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                engine = LAMMPSEngine(
                    executable="lmp", workdir=tmpdir1, check_executable=False
                )

                try:
                    engine.run(
                        "units real\n",
                        workdir=tmpdir2,
                        capture_output=True,
                        check=False,
                    )
                    assert not (Path(tmpdir1) / "input.lmp").exists()
                    assert (Path(tmpdir2) / "input.lmp").exists()
                except FileNotFoundError:
                    pass


class TestCP2KEngine:
    """Test CP2K engine specifics."""

    def test_name(self):
        engine = CP2KEngine(executable="cp2k", check_executable=False)
        assert engine.name == "CP2K"

    def test_extension(self):
        engine = CP2KEngine(executable="cp2k", check_executable=False)
        assert engine._get_default_extension() == ".inp"

    def test_run_with_script(self):
        script = Script.from_text("input", "&GLOBAL\n  PROJECT water\n&END GLOBAL\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CP2KEngine(
                executable="cp2k.psmp", workdir=tmpdir, check_executable=False
            )

            try:
                engine.run(script, capture_output=True, check=False)
                assert engine.work_dir == Path(tmpdir)
                assert len(engine.scripts) == 1
                assert (Path(tmpdir) / "input.inp").exists()
            except (FileNotFoundError, PermissionError):
                pass


class TestLAMMPSEngine:
    """Test LAMMPS engine specifics."""

    def test_name(self):
        engine = LAMMPSEngine(executable="lmp", check_executable=False)
        assert engine.name == "LAMMPS"

    def test_extension(self):
        engine = LAMMPSEngine(executable="lmp", check_executable=False)
        assert engine._get_default_extension() == ".lmp"

    def test_run_accepts_timeout(self):
        """Timeout parameter is forwarded to subprocess.run without TypeError."""
        script = Script.from_text("input", "units real\natom_style full\n")
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = LAMMPSEngine(
                executable="lmp", workdir=tmpdir, check_executable=False
            )
            try:
                engine.run(script, capture_output=True, check=False, timeout=1)
            except (FileNotFoundError, TimeoutError):
                pass  # Expected: LAMMPS not installed or timed out

            # Script was saved regardless
            assert (Path(tmpdir) / "input.lmp").exists()
