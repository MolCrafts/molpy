"""Unit tests for :mod:`molpy.builder.polymer.ambertools.amber_utils`."""

from unittest.mock import patch

import pytest

from molpy.builder.polymer.ambertools.amber_utils import (
    check_amber_tools_available,
    configure_amber_wrappers,
)


class TestAmberUtilities:
    def test_default_uses_system_environment(self, tmp_path):
        wrappers = configure_amber_wrappers(tmp_path)
        assert [wrapper.name for wrapper in wrappers] == [
            "antechamber",
            "prepgen",
            "tleap",
        ]
        assert all(wrapper.workdir == tmp_path for wrapper in wrappers)
        assert all(wrapper.env is None for wrapper in wrappers)
        assert all(wrapper.env_manager is None for wrapper in wrappers)

    def test_wrapper_configuration_requires_manager_with_env(self, tmp_path):
        with pytest.raises(ValueError, match="incomplete"):
            configure_amber_wrappers(tmp_path, env="AmberTools25")

    def test_wrapper_configuration_uses_named_env(self, tmp_path):
        wrappers = configure_amber_wrappers(
            tmp_path, env="AmberTools25", env_manager="conda"
        )
        assert all(wrapper.env == "AmberTools25" for wrapper in wrappers)
        assert all(wrapper.env_manager == "conda" for wrapper in wrappers)

    def test_availability_requires_every_wrapper(self):
        wrappers = [
            type("Stub", (), {"is_available": lambda self: True})() for _ in range(3)
        ]
        wrappers[-1].is_available = lambda: False
        with patch(
            "molpy.builder.polymer.ambertools.amber_utils.configure_amber_wrappers",
            return_value=tuple(wrappers),
        ):
            assert check_amber_tools_available() is False
