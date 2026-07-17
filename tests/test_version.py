"""Exact molpy/molrs release-contract tests."""

from __future__ import annotations

import importlib
import importlib.metadata

import pytest


version_module = importlib.import_module("molpy.version")


def test_exact_molrs_version_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib.metadata, "version", lambda _name: version_module.version
    )
    assert version_module.check_molrs_version() == version_module.version


def test_mismatched_molrs_version_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.6.0")
    with pytest.raises(ImportError, match="Version mismatch"):
        version_module.check_molrs_version()


def test_missing_molrs_metadata_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError("molcrafts-molrs")

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(ImportError, match="package metadata is missing"):
        version_module.check_molrs_version()


def test_permissive_strict_switch_is_absent() -> None:
    with pytest.raises(TypeError):
        version_module.check_molrs_version(strict=False)
