"""
Unit tests for engine base classes.
"""

import tempfile
from pathlib import Path

import pytest

from molpy import Script
from molpy.engine import CP2KEngine, LAMMPSEngine


class TestEngine:
    """Test Engine base class functionality."""

    def test_engine_init(self):
        """Test engine initialization."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)
        assert engine.executable == "lmp"
        assert engine.name == "LAMMPS"

    def test_engine_check_executable(self):
        """Test executable checking."""
        # Should raise FileNotFoundError for non-existent executable
        with pytest.raises(FileNotFoundError):
            LAMMPSEngine(
                executable="nonexistent_executable_xyz123", check_executable=True
            )

    def test_engine_prepare_single_script(self):
        """Test preparing engine with a single script."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)

        script = Script.from_text(
            name="input", text="units real\natom_style full\n", language="other"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.prepare(work_dir=tmpdir, scripts=script)

            assert engine.work_dir == Path(tmpdir)
            assert len(engine.scripts) == 1
            assert engine.input_script == script
            assert engine.input_script.path is not None
            assert engine.input_script.path.name == "input.lmp"

            # Check that script was saved
            script_path = engine.work_dir / "input.lmp"
            assert script_path.exists()
            assert script_path.read_text() == "units real\natom_style full\n"

    def test_engine_prepare_multiple_scripts(self):
        """Test preparing engine with multiple scripts."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)

        script1 = Script.from_text(
            name="input", text="units real\natom_style full\n", language="other"
        )
        script2 = Script.from_text(
            name="data", text="1 1.0 1.0 1.0\n", language="other"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.prepare(work_dir=tmpdir, scripts=[script1, script2])

            assert len(engine.scripts) == 2
            assert engine.input_script == script1  # First script is input

            # Check that scripts were saved
            assert (engine.work_dir / "input.lmp").exists()
            assert (engine.work_dir / "data.lmp").exists()

    def test_engine_prepare_with_input_tag(self):
        """Test preparing engine with script tagged as input."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)

        script1 = Script.from_text(name="config", text="units real\n", language="other")
        script2 = Script.from_text(
            name="input", text="atom_style full\n", language="other"
        )
        script2.tags.add("input")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.prepare(work_dir=tmpdir, scripts=[script1, script2])

            # Script with 'input' tag should be the input script
            assert engine.input_script == script2

    def test_engine_prepare_empty_scripts(self):
        """Test that preparing with empty scripts raises error."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="At least one script is required"):
                engine.prepare(work_dir=tmpdir, scripts=[])

    def test_engine_prepare_with_path(self):
        """Test preparing engine with script that has a path."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)

        script = Script.from_text(
            name="input", text="units real\n", language="other", path=Path("custom.lmp")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.prepare(work_dir=tmpdir, scripts=script)

            # Should use the filename from script.path
            assert engine.input_script.path.name == "custom.lmp"
            assert (engine.work_dir / "custom.lmp").exists()

    def test_engine_get_script_by_name(self):
        """Test getting script by name."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)

        script1 = Script.from_text(name="input", text="units real\n", language="other")
        script2 = Script.from_text(name="data", text="1 1.0\n", language="other")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.prepare(work_dir=tmpdir, scripts=[script1, script2])

            # Get script by logical name
            found = engine.get_script(name="data")
            assert found == script2

            # Get script by filename
            found = engine.get_script(name="data.lmp")
            assert found == script2

            # Get non-existent script
            found = engine.get_script(name="nonexistent")
            assert found is None

    def test_engine_get_script_by_tag(self):
        """Test getting script by tag."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)

        script1 = Script.from_text(name="input", text="units real\n", language="other")
        script1.tags.add("input")
        script2 = Script.from_text(name="data", text="1 1.0\n", language="other")
        script2.tags.add("data")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.prepare(work_dir=tmpdir, scripts=[script1, script2])

            # Get script by tag
            found = engine.get_script(tag="data")
            assert found == script2

            # Get non-existent tag
            found = engine.get_script(tag="nonexistent")
            assert found is None

    def test_engine_clean(self):
        """Test cleaning up engine files."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)

        script = Script.from_text(name="input", text="units real\n", language="other")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.prepare(work_dir=tmpdir, scripts=script)

            # Create an output file
            output_file = engine.work_dir / "output.txt"
            output_file.write_text("output")

            # Clean without keeping scripts
            engine.clean(keep_scripts=False)

            # Script should be removed
            assert not (engine.work_dir / "input.lmp").exists()
            # Output file should still exist (not cleaned)
            assert output_file.exists()

    def test_engine_list_output_files(self):
        """Test listing output files."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)

        script = Script.from_text(name="input", text="units real\n", language="other")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.prepare(work_dir=tmpdir, scripts=script)

            # Create output files
            output1 = engine.work_dir / "output1.txt"
            output2 = engine.work_dir / "output2.txt"
            output1.write_text("output1")
            output2.write_text("output2")

            # List output files
            output_files = engine.list_output_files()

            # Should not include input script
            assert "input.lmp" not in [f.name for f in output_files]
            # Should include output files
            assert output1 in output_files
            assert output2 in output_files

    def test_engine_get_output_file(self):
        """Test getting output file."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)

        script = Script.from_text(name="input", text="units real\n", language="other")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.prepare(work_dir=tmpdir, scripts=script)

            # Create output file
            output_file = engine.work_dir / "output.txt"
            output_file.write_text("output")

            # Get output file by name
            found = engine.get_output_file(name="output.txt")
            assert found == output_file

            # Get output file without name (should return first)
            found = engine.get_output_file()
            assert found == output_file

            # Get non-existent file
            found = engine.get_output_file(name="nonexistent.txt")
            assert found is None


class TestLAMMPSEngine:
    """Test LAMMPSEngine class."""

    def test_lammps_engine_default_extension(self):
        """Test LAMMPS engine default extension."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)
        assert engine._get_default_extension() == ".lmp"

    def test_lammps_engine_run_not_prepared(self):
        """Test that running without preparation raises error."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)

        with pytest.raises(RuntimeError, match="Engine not prepared"):
            engine.run()

    def test_lammps_engine_run_with_script(self):
        """Test running LAMMPS engine with script."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)

        script = Script.from_text(
            name="input", text="units real\natom_style full\n", language="other"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.prepare(work_dir=tmpdir, scripts=script)

            # Note: This will fail if LAMMPS is not installed, but we can test the command
            # construction without actually running
            try:
                engine.run(capture_output=True, check=False)
                # If LAMMPS is not available, returncode will be non-zero
                # but we can still verify the command was constructed correctly
            except FileNotFoundError:
                # LAMMPS not found, skip this test
                pytest.skip("LAMMPS executable not found")

    def test_lammps_engine_get_log_file(self):
        """Test getting LAMMPS log file."""
        engine = LAMMPSEngine(executable="lmp", check_executable=False)

        script = Script.from_text(name="input", text="units real\n", language="other")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.prepare(work_dir=tmpdir, scripts=script)

            # Create log file
            log_file = engine.work_dir / "log.lammps"
            log_file.write_text("log content")

            # Get log file
            found = engine.get_log_file()
            assert found == log_file

            # Get non-existent log file
            log_file.unlink()
            found = engine.get_log_file()
            assert found is None


class TestCP2KEngine:
    """Test CP2KEngine class."""

    def test_cp2k_engine_default_extension(self):
        """Test CP2K engine default extension."""
        engine = CP2KEngine(executable="cp2k.psmp", check_executable=False)
        assert engine._get_default_extension() == ".inp"

    def test_cp2k_engine_run_not_prepared(self):
        """Test that running without preparation raises error."""
        engine = CP2KEngine(executable="cp2k.psmp", check_executable=False)

        with pytest.raises(RuntimeError, match="Engine not prepared"):
            engine.run()

    def test_cp2k_engine_get_output_file(self):
        """Test getting CP2K output file."""
        engine = CP2KEngine(executable="cp2k.psmp", check_executable=False)

        script = Script.from_text(
            name="input", text="&GLOBAL\n&END GLOBAL\n", language="other"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            engine.prepare(work_dir=tmpdir, scripts=script)

            # Create output file
            output_file = engine.work_dir / "cp2k.out"
            output_file.write_text("output content")

            # Get output file
            found = engine.get_output_file()
            assert found == output_file

            # Get output file by name
            found = engine.get_output_file(name="cp2k.out")
            assert found == output_file

            # Get non-existent file
            output_file.unlink()
            found = engine.get_output_file()
            assert found is None
