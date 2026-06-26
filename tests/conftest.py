import contextlib
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import mollog
import pytest

_REPO_URL = "https://github.com/molcrafts/tests-data.git"
_DEFAULT_DIR = Path(__file__).resolve().parent / "tests-data"


class _RecordingHandler(mollog.Handler):
    """mollog handler that collects records for assertions in tests."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[mollog.LogRecord] = []

    def emit(self, record: mollog.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def mollog_capture() -> Callable[[str], "contextlib.AbstractContextManager"]:
    """Factory fixture to capture records on a named mollog logger.

    MolPy logs through :mod:`mollog`, not the stdlib ``logging`` module, so
    pytest's ``caplog`` does not see its records. Usage::

        with mollog_capture("molpy.reacter.bond_react") as records:
            do_something()
        assert any(r.level >= mollog.Level.WARNING for r in records)
    """

    @contextlib.contextmanager
    def _capture(name: str) -> Iterator[list[mollog.LogRecord]]:
        logger = mollog.get_logger(name)
        handler = _RecordingHandler()
        logger.add_handler(handler)
        try:
            yield handler.records
        finally:
            logger.remove_handler(handler)

    return _capture


@pytest.fixture(scope="session", name="TEST_DATA_DIR")
def find_test_data() -> Path:
    """
    Ensure the tests-data repository is present and up-to-date.
    * If the directory already contains a `.git` folder -> `git pull`.
      Otherwise clone afresh.
    """
    if (_DEFAULT_DIR / ".git").exists():
        subprocess.run(["git", "pull", "--ff-only"], cwd=_DEFAULT_DIR)
    else:
        _DEFAULT_DIR.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", _REPO_URL, str(_DEFAULT_DIR)],
            cwd=_DEFAULT_DIR.parent,
        )
        if result.returncode != 0:
            pytest.skip(f"Cannot clone tests-data (exit {result.returncode})")
    if not _DEFAULT_DIR.exists():
        pytest.skip("tests-data directory not available")
    return _DEFAULT_DIR


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark tests requiring external software.

    We use the `external` marker for tests that depend on third-party software
    outside the Python environment (e.g. RDKit, simulation engines).
    """

    for item in items:
        # Engine suite (may require external simulation engines / heavier runtime)
        try:
            fspath = Path(str(item.fspath))
        except Exception:  # pragma: no cover
            continue
        if "test_engine" in fspath.parts:
            item.add_marker(pytest.mark.external)

        # RDKit-dependent tests
        if "test_adapter" in fspath.parts and "rdkit" in fspath.name:
            item.add_marker(pytest.mark.external)
        # AmberTools-dependent tests
        if "test_wrapper" in fspath.parts and fspath.name in (
            "test_antechamber.py",
            "test_parmchk2.py",
            "test_tleap.py",
        ):
            item.add_marker(pytest.mark.external)
        if "test_builder" in fspath.parts and "amber" in fspath.name:
            item.add_marker(pytest.mark.external)
