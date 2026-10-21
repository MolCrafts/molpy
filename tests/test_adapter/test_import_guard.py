"""molpy.adapter imports without RDKit; its names then fail with a clear ImportError."""

import importlib
import importlib.abc
import importlib.util
import sys

import pytest

import molpy
import molpy.adapter


@pytest.fixture
def reimport_adapter(monkeypatch):
    """Re-import ``molpy.adapter`` from scratch; the original is restored after."""
    for name in ("molpy.adapter", "molpy.adapter._rdkit"):
        if name in sys.modules:
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.delattr(molpy, "adapter", raising=False)
    return lambda: importlib.import_module("molpy.adapter")


def test_missing_rdkit_leaves_the_package_importable(monkeypatch, reimport_adapter):
    monkeypatch.setitem(sys.modules, "rdkit", None)  # import rdkit -> not found
    adapter = reimport_adapter()
    with pytest.raises(ImportError, match=r"needs RDKit.*molpy\[rdkit\]"):
        adapter.to_rdkit
    with pytest.raises(ImportError, match="needs RDKit"):
        from molpy.adapter import from_rdkit  # noqa: F401


class _RdkitMissingItsOwnDependency(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds ``rdkit``, whose import then fails on a module RDKit itself needs."""

    def find_spec(self, name, path=None, target=None):
        return importlib.util.spec_from_loader(name, self) if name == "rdkit" else None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ModuleNotFoundError("No module named 'libfoo'", name="libfoo")


def test_a_broken_rdkit_install_is_not_mistaken_for_a_missing_one(
    monkeypatch, reimport_adapter
):
    for name in [m for m in sys.modules if m == "rdkit" or m.startswith("rdkit.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(
        sys, "meta_path", [_RdkitMissingItsOwnDependency(), *sys.meta_path]
    )
    with pytest.raises(ModuleNotFoundError) as info:
        reimport_adapter()
    assert info.value.name == "libfoo"


def test_an_unknown_name_is_an_attribute_error():
    with pytest.raises(AttributeError):
        molpy.adapter.RdkitAdapter
