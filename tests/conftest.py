import subprocess
from pathlib import Path

import pytest

_REPO_URL = "https://github.com/molcrafts/tests-data.git"
_DEFAULT_DIR = Path(__file__).resolve().parent / "tests-data"


@pytest.fixture(scope="session", name="TEST_DATA_DIR")
def find_test_data() -> Path:
    """
    Ensure the tests-data repository is present and up-to-date.
    * If the directory already contains a `.git` folder -> `git pull`.
      Otherwise clone afresh.
    """
    if (_DEFAULT_DIR / ".git").exists():
        subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=_DEFAULT_DIR,
        )
    else:
        _DEFAULT_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", _REPO_URL, str(_DEFAULT_DIR)],
            cwd=_DEFAULT_DIR.parent,
        )
    return _DEFAULT_DIR
