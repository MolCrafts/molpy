"""Unit tests for :mod:`molpy.builder.polymer.ambertools.amber_utils`."""

from unittest.mock import patch

from molpy.builder.polymer.ambertools.amber_utils import (
    check_amber_tools_available,
    configure_amber_wrappers,
)


class TestAmberUtilities:
    def test_wrapper_configuration_uses_one_environment_and_workdir(self, tmp_path):
        wrappers = configure_amber_wrappers(tmp_path, "AmberTools25")
        assert [wrapper.name for wrapper in wrappers] == [
            "antechamber",
            "prepgen",
            "tleap",
        ]
        assert all(wrapper.workdir == tmp_path for wrapper in wrappers)

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
