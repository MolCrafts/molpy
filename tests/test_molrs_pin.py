"""Declared molrs pin and installed package share the 0.14 minor line."""

from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

import molrs


class TestMolrsPin:
    def test_declared_pin_is_the_0_14_line(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        deps = data["project"]["dependencies"]
        pin = next(d for d in deps if d.startswith("molcrafts-molrs"))
        assert pin == "molcrafts-molrs>=0.14.0,<0.15"

    def test_installed_molrs_is_0_14(self) -> None:
        installed = version("molcrafts-molrs")
        major, minor, *_ = installed.split(".")
        assert (major, minor) == ("0", "14")

    def test_record_is_the_public_name(self) -> None:
        assert hasattr(molrs, "Record")
        assert not hasattr(molrs, "MolRec")
