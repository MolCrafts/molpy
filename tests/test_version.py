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
    # molpy 0.12.1 may pair with molrs 0.12.0 or 0.12.99 — patch is ignored
    major, minor, *_ = version_module.version.split(".")
    for patch in ("0", "1", "99"):
        molrs_ver = f"{major}.{minor}.{patch}"
        monkeypatch.setattr(importlib.metadata, "version", lambda _n, v=molrs_ver: v)
        assert version_module.check_molrs_version() == molrs_ver


def test_older_patch_molrs_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """molpy 0.12.1 + molrs 0.12.0 must work (patch not in the check)."""
    major, minor, *_ = version_module.version.split(".")
    older = f"{major}.{minor}.0"
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: older)
    assert version_module.check_molrs_version() == older


def test_different_minor_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.6.0")
    with pytest.raises(ImportError, match="Minor-version mismatch"):
        version_module.check_molrs_version()


def test_different_major_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    major, minor, *_ = version_module.version.split(".")
    other_major = f"{int(major) + 1}.{minor}.0"
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: other_major)
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
