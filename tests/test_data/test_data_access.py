"""
Tests for data file access module.
"""

from pathlib import Path

import pytest

from molpy.data import (
    exists,
    get_forcefield_path,
    get_path,
    list_files,
    list_forcefields,
)


class TestDataAccess:
    """Test basic data file access functions."""

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
        """Test checking if a data file exists."""
        assert exists("forcefield/tip3p.xml")
        assert exists("forcefield/tip3p.xml")
        assert not exists("forcefield/nonexistent.xml")

    def test_get_forcefield_path(self):
        """Test getting forcefield path using convenience function."""
        path = get_forcefield_path("tip3p.xml")
        assert Path(path).exists()
        assert Path(path).name == "tip3p.xml"

    def test_get_forcefield_path_nonexistent(self):
        """Test getting nonexistent forcefield path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_forcefield_path("nonexistent.xml")

    def test_list_forcefields(self):
        """Test listing available forcefields."""
        assert set(list_forcefields()) == {"clp.xml", "tip3p.xml"}


class TestDataModuleImport:
    """Test that data module can be imported and used."""

    def test_import_data_module(self):
        """Test importing the data module."""
        import molpy.data

        assert hasattr(molpy.data, "get_path")
        assert hasattr(molpy.data, "list_files")
        assert hasattr(molpy.data, "exists")
        assert hasattr(molpy.data, "get_forcefield_path")
        assert hasattr(molpy.data, "list_forcefields")
