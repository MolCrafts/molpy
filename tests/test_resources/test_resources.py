"""Tests for ``molpy.resources``: the files molpy ships."""

from pathlib import Path

import pytest

from molpy.resources import exists, get_path, list_files


class TestResourceAccess:
    """``get_path``, ``list_files`` and ``exists`` over the shipped files."""

    def test_get_path_forcefield(self):
        """Test getting path to a forcefield file."""
        path = get_path("forcefield/tip3p.xml")
        assert path.exists()
        assert path.name == "tip3p.xml"
        assert "forcefield" in str(path)

    def test_get_path_nonexistent(self):
        """A missing file raises FileNotFoundError naming the available files."""
        with pytest.raises(FileNotFoundError, match="Available files"):
            get_path("forcefield/nonexistent.xml")

    def test_list_files_forcefield(self):
        """Test listing files in forcefield directory."""
        files = list(list_files("forcefield"))
        assert len(files) > 0
        assert any("tip3p.xml" in f for f in files)
        # Should not include Python files
        assert not any("__init__.py" in f for f in files)

    def test_list_files_exclude_python(self):
        """Test that list_files excludes Python files by default."""
        files = list(list_files("forcefield", exclude_python=True))
        assert not any(f.endswith(".py") for f in files)
        assert not any("__init__" in f for f in files)

    def test_list_files_include_python(self):
        """Test that list_files can include Python files if requested."""
        files = list(list_files("forcefield", exclude_python=False))
        # Should include __init__.py if it exists
        assert any("__init__.py" in f for f in files)

    def test_exists(self):
        """``exists`` says whether a shipped file is there."""
        assert exists("forcefield/tip3p.xml")
        assert not exists("forcefield/nonexistent.xml")

    def test_the_bundled_forcefields(self):
        """The force-field files molpy ships, found through ``list_files``."""
        names = {Path(f).name for f in list_files("forcefield")}
        assert names == {"clp.xml", "tip3p.xml"}


class TestResourcesModule:
    """``molpy.resources`` exposes its three functions."""

    def test_import_resources_module(self):
        """The module carries ``get_path``, ``list_files`` and ``exists``."""
        import molpy.resources

        assert hasattr(molpy.resources, "get_path")
        assert hasattr(molpy.resources, "list_files")
        assert hasattr(molpy.resources, "exists")
