from collections.abc import Iterator
from pathlib import Path

import mollog
import pytest


@pytest.fixture(scope="session", name="TEST_DATA_DIR")
def test_data_dir() -> Path:
    """Fixture files committed under tests/tests-data."""
    return Path(__file__).resolve().parent / "tests-data"


@pytest.fixture(autouse=True)
def _hermetic_config(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """molpy's configuration sees no developer's files.

    The user layer lives under ``MOLCRAFTS_HOME`` (an empty directory here),
    and the project layer is ``molpy.toml`` in the working directory, which
    the checkout does not have. A test that wants a layer writes it under its
    own ``tmp_path``.
    """
    home = tmp_path_factory.mktemp("molcrafts-home")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MOLCRAFTS_HOME", str(home))
        yield


class _Capture(mollog.Handler):
    """Keeps every mollog record it is handed."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[mollog.LogRecord] = []

    def emit(self, record: mollog.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def molpy_records() -> Iterator[list[mollog.LogRecord]]:
    """The records molpy's loggers emit during the test (the ``molpy`` tree)."""
    handler = _Capture()
    logger = mollog.get_logger("molpy")
    logger.add_handler(handler)
    try:
        yield handler.records
    finally:
        logger.remove_handler(handler)
