import contextlib
import io
import tarfile
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

import mollog
import pytest
from filelock import FileLock

_TARBALL_URL = (
    "https://github.com/molcrafts/tests-data/archive/refs/heads/master.tar.gz"
)
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

        with mollog_capture("molpy.io.data.lammps_bond_react") as records:
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
    """Download tests-data (minus ``con/``) if absent; skip data tests if it isn't.

    Fetches the archive and extracts everything except the ``con/`` directory,
    rather than git-cloning. tests-data (a fork of chemfiles/tests-data) ships EON
    fixtures under ``con/`` — a reserved device name Windows cannot create, so a
    git checkout aborts there ("cannot create directory at 'con'") and
    sparse-checkout can't reliably exclude it on Windows. Extracting the tarball
    without con/ never asks the OS to create it, so this works on every platform
    (nothing reads con/ data) and leaves tests-data itself untouched. If the
    download fails (e.g. offline), skip the data-dependent tests via a
    ``README.md`` sentinel rather than failing.

    An existing checkout (git clone or a previous extract) is reused as-is and
    deliberately NOT refreshed: under ``-n auto`` the session fixture runs once
    per worker, and a concurrent refresh would corrupt the shared tree. CI
    fetches fresh each run (in a serial pre-test step); to refresh a local copy,
    delete tests/tests-data.
    """
    if not _SENTINEL.exists():
        try:
            with urllib.request.urlopen(_TARBALL_URL, timeout=60) as resp:
                raw = resp.read()
        except OSError:
            pytest.skip("tests-data unavailable (download failed)")
        _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            members = []
            for member in tar.getmembers():
                # Archive paths are "tests-data-<ref>/<rel>"; drop the root
                # component and skip the Windows-reserved con/ directory.
                _, _, rel = member.name.partition("/")
                if not rel or rel == "con" or rel.startswith("con/"):
                    continue
                member.name = rel
                members.append(member)
            tar.extractall(_DEFAULT_DIR, members=members, filter="data")
    if not _SENTINEL.exists():
        pytest.skip("tests-data checkout unavailable or incomplete")
    return _DEFAULT_DIR


@pytest.fixture(scope="session", name="TEST_DATA_DIR")
def find_test_data(tmp_path_factory, worker_id) -> Path:
    """Ensure the tests-data repository is present.

    xdist-safe: the session fixture runs once **per worker**, so without a lock
    all workers would download/extract into the *same* directory concurrently and
    corrupt each other's tree. Serialize the fetch across workers with a
    cross-process lock on a shared path; ``_ensure_test_data`` is a no-op once the
    tree is present, so running it once per worker (in turn) is harmless.
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
    """Mark tests that need third-party *executables* as ``external``.

    Policy: the default gate (``pytest -m "not external"``) never requires
    third-party scientific software — not RDKit, AmberTools, LAMMPS, Packmol,
    OpenMM, freud, etc. Wrapper unit tests that mock ``subprocess`` stay in the
    default gate. Only suites that must talk to a real external binary are
    marked here (or with an explicit ``@pytest.mark.external``).
    """

    for item in items:
        try:
            fspath = Path(str(item.fspath))
        except Exception:  # pragma: no cover
            continue
        # Engine suite talks to real MD engines when present.
        if "test_engine" in fspath.parts:
            item.add_marker(pytest.mark.external)
        # AmberTools polymer builder integration (real antechamber/tleap).
        if "test_builder" in fspath.parts and "amber" in fspath.name:
            item.add_marker(pytest.mark.external)
        # Nested ambertools builder package under polymer/ (same rule).
        if "test_builder" in fspath.parts and "ambertools" in fspath.parts:
            item.add_marker(pytest.mark.external)
