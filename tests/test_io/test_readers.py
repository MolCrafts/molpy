"""Unit tests for :mod:`molpy.io.readers` entry points."""

from __future__ import annotations

import pytest

import molpy as mp


class TestReadSmiles:
    def test_multi_component_names_the_components_door(self):
        with pytest.raises(ValueError) as excinfo:
            mp.io.read_smiles("[Li+].[F-]")

        message = str(excinfo.value)
        assert "components()" in message
        assert "adopt" not in message
