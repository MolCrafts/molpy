import contextlib
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import mollog
import pytest
from filelock import FileLock

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


_SENTINEL = _DEFAULT_DIR / "README.md"


def _ensure_test_data() -> Path:
    """Clone tests-data (skipping ``con/``) if absent; skip data tests if absent.

    tests-data (a fork of chemfiles/tests-data) ships EON fixtures under ``con/``,
    a reserved device name Windows cannot create — a plain clone aborts there. We
    leave tests-data untouched and skip con/ at checkout instead. Two settings are
    BOTH required on Windows: ``core.protectNTFS=false`` lets git process the con/
    index entry without rejecting the name, and sparse-checkout keeps con/ off disk
    so the OS is never asked to create it (nothing reads con/ data). If tests-data
    still isn't available (e.g. offline), skip the data-dependent tests via a
    ``README.md`` sentinel rather than failing.

    Deliberately does NOT ``git pull`` a present checkout: under ``-n auto`` the
    session fixture runs once per worker, so pulling would mutate the shared
    working tree while other workers read from it. CI clones fresh each run (in a
    serial pre-test step); to refresh a local copy, delete tests/tests-data.
    """
    if not (_DEFAULT_DIR / ".git").exists():
        _DEFAULT_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--no-checkout",
                _REPO_URL,
                str(_DEFAULT_DIR),
            ],
            cwd=_DEFAULT_DIR.parent,
        )
        subprocess.run(
            ["git", "-C", str(_DEFAULT_DIR), "config", "core.protectNTFS", "false"]
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(_DEFAULT_DIR),
                "sparse-checkout",
                "set",
                "--no-cone",
                "/*",
                "!/con/",
            ],
        )
        subprocess.run(["git", "-C", str(_DEFAULT_DIR), "checkout"])
    if not _SENTINEL.exists():
        pytest.skip("tests-data checkout unavailable or incomplete")
    return _DEFAULT_DIR


@pytest.fixture(scope="session", name="TEST_DATA_DIR")
def find_test_data(tmp_path_factory, worker_id) -> Path:
    """Ensure the tests-data repository is present and up-to-date.

    xdist-safe: the session fixture runs once **per worker**, so without a lock
    all workers would `git clone`/`pull` the *same* directory concurrently and
    corrupt each other's checkout (Windows is strict about concurrent file
    access — this manifested as spurious "path not found" file-read failures
    under ``-n auto``). Serialize the git operation across workers with a
    cross-process lock on a shared path; ``git pull --ff-only`` is idempotent so
    running it once per worker (in turn) is harmless.
    """
    if worker_id == "master":
        # Not running under xdist — no other workers to race with.
        return _ensure_test_data()
    lock_path = tmp_path_factory.getbasetemp().parent / "tests_data.lock"
    with FileLock(str(lock_path)):
        return _ensure_test_data()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark tests requiring external software.

    We use the `external` marker for tests that depend on third-party software
    outside the Python environment (e.g. simulation engines, AmberTools).
    """

    for item in items:
        # Engine suite (may require external simulation engines / heavier runtime)
        try:
            fspath = Path(str(item.fspath))
        except Exception:  # pragma: no cover
            continue
        if "test_engine" in fspath.parts:
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
