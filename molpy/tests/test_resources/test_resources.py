"""Tests for ``molpy.resources``: the files molpy ships."""

import gc
import zipfile
from pathlib import Path

import pytest

import molpy as mp


class TestResourceAccess:
    """``get_path``, ``list_files`` and ``exists`` over the shipped files."""

    def test_get_path_forcefield(self):
        """Test getting path to a forcefield file."""
        path = mp.resources.get_path("forcefield/tip3p.xml")
        assert path.exists()
        assert path.name == "tip3p.xml"
        assert "forcefield" in str(path)

    def test_get_path_nonexistent(self):
        """A missing file raises FileNotFoundError naming the available files."""
        with pytest.raises(FileNotFoundError, match="Available files"):
            mp.resources.get_path("forcefield/nonexistent.xml")

    def test_list_files_forcefield(self):
        """Test listing files in forcefield directory."""
        files = list(mp.resources.list_files("forcefield"))
        assert len(files) > 0
        assert any("tip3p.xml" in f for f in files)
        # Should not include Python files
        assert not any("__init__.py" in f for f in files)

    def test_list_files_exclude_python(self):
        """Test that list_files excludes Python files by default."""
        files = list(mp.resources.list_files("forcefield", exclude_python=True))
        assert not any(f.endswith(".py") for f in files)
        assert not any("__init__" in f for f in files)

    def test_list_files_include_python(self):
        """Test that list_files can include Python files if requested."""
        files = list(mp.resources.list_files("forcefield", exclude_python=False))
        # Should include __init__.py if it exists
        assert any("__init__.py" in f for f in files)

    def test_get_path_from_a_zip_install_stays_valid(self, tmp_path, monkeypatch):
        """A zipped install extracts the file; the path outlives the call."""
        archive = tmp_path / "resources.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("res/forcefield/tip3p.xml", "<ForceField/>")
        root = zipfile.Path(archive, "res/")
        monkeypatch.setattr(mp.resources, "files", lambda _package: root)

        path = mp.resources.get_path("forcefield/tip3p.xml")
        gc.collect()
        assert path.is_file()
        assert path.read_text(encoding="utf-8") == "<ForceField/>"
        assert mp.resources.exists("forcefield/tip3p.xml")
        assert not mp.resources.exists("forcefield/missing.xml")

    def test_exists(self):
        """``exists`` says whether a shipped file is there."""
        assert mp.resources.exists("forcefield/tip3p.xml")
        assert not mp.resources.exists("forcefield/nonexistent.xml")

    def test_the_bundled_forcefields(self):
        """The force-field files molpy ships, found through ``list_files``."""
        names = {Path(f).name for f in mp.resources.list_files("forcefield")}
        assert names == {"clp.xml", "tip3p.xml"}
