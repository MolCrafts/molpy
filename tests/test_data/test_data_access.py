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
        path = get_path("forcefield/oplsaa.xml")
        assert path.exists()
        assert path.name == "oplsaa.xml"
        assert "forcefield" in str(path)

    def test_get_path_nonexistent(self):
        """Test getting path to a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_path("forcefield/nonexistent.xml")

    def test_list_files_forcefield(self):
        """Test listing files in forcefield directory."""
        files = list(list_files("forcefield"))
        assert len(files) > 0
        assert any("oplsaa.xml" in f for f in files)
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
        assert exists("forcefield/oplsaa.xml")
        assert exists("forcefield/tip3p.xml")
        assert not exists("forcefield/nonexistent.xml")

    def test_get_forcefield_path(self):
        """Test getting forcefield path using convenience function."""
        path = get_forcefield_path("oplsaa.xml")
        assert Path(path).exists()
        assert Path(path).name == "oplsaa.xml"

    def test_get_forcefield_path_nonexistent(self):
        """Test getting nonexistent forcefield path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_forcefield_path("nonexistent.xml")

    def test_list_forcefields(self):
        """Test listing available forcefields."""
        forcefields = list_forcefields()
        assert len(forcefields) > 0
        assert "oplsaa.xml" in forcefields
        assert "tip3p.xml" in forcefields
        # Should not include Python files
        assert "__init__.py" not in forcefields


class TestForcefieldIntegration:
    """Test integration with forcefield loader."""

    def test_forcefield_loader_uses_data_module(self):
        """Test that forcefield loader uses the data module."""
        from molpy.io.forcefield.xml import _resolve_forcefield_path

        # Test with just filename
        path = _resolve_forcefield_path("oplsaa.xml")
        assert path.exists()
        assert path.name == "oplsaa.xml"

    def test_forcefield_loader_with_full_path(self):
        """Test that forcefield loader works with full paths."""
        from molpy.io.forcefield.xml import _resolve_forcefield_path

        # Get the actual path
        data_path = get_path("forcefield/tip3p.xml")
        path = _resolve_forcefield_path(str(data_path))
        assert path.exists()
        assert path.name == "tip3p.xml"

    def test_forcefield_loader_nonexistent(self):
        """Test that forcefield loader raises FileNotFoundError for nonexistent files."""
        from molpy.io.forcefield.xml import _resolve_forcefield_path

        with pytest.raises(FileNotFoundError) as exc_info:
            _resolve_forcefield_path("nonexistent.xml")

        # Check that error message includes available forcefields
        assert "Available built-in force fields" in str(exc_info.value)


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

    def test_import_data_forcefield_submodule(self):
        """Test importing the forcefield submodule."""
        from molpy.data.forcefield import get_forcefield_path, list_forcefields

        path = get_forcefield_path("oplsaa.xml")
        assert Path(path).exists()

        forcefields = list_forcefields()
        assert len(forcefields) > 0
