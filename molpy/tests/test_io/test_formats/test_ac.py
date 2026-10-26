"""``mp.io.read_ac``: the native reader plus the ``q`` -> ``charge`` rename."""

import pytest

import molpy as mp


def test_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        mp.io.read_ac(tmp_path / "nope.ac")
