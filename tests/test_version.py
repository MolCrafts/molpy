"""molpy/molrs minor-version compatibility tests."""

from __future__ import annotations

import importlib
import importlib.metadata

import pytest


version_module = importlib.import_module("molpy.version")


def test_matching_minor_same_patch_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.metadata, "version", lambda _name: version_module.version
    )
    assert version_module.check_molrs_version() == version_module.version


def test_same_minor_different_patch_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # molpy 0.10.0 may pair with molrs 0.10.1
    major, minor, *_ = version_module.version.split(".")
    newer_patch = f"{major}.{minor}.99"
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: newer_patch)
    assert version_module.check_molrs_version() == newer_patch


def test_different_minor_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.6.0")
    with pytest.raises(ImportError, match="Minor-version mismatch"):
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
