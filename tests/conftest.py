from pathlib import Path

import pytest


@pytest.fixture(scope="session", name="TEST_DATA_DIR")
def test_data_dir() -> Path:
    """Fixture files committed under tests/tests-data."""
    return Path(__file__).resolve().parent / "tests-data"
