# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-29
# version: 0.0.1

from pathlib import Path
import pytest
import subprocess
import os

_REPO_URL = "https://github.com/molcrafts/chemfile-testcases.git"
_DEFAULT_DIR = Path(__file__).parent / "chemfile-testcases"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    """Subprocess helper with shorter syntax and error surfacing."""
    subprocess.run(cmd, cwd=cwd, check=True, text=True)


@pytest.fixture(scope="session", name="TEST_DATA_DIR")
def find_test_data() -> Path:
    """
    Ensure the chemfile-testcases repository is present and up-to-date.

    * Set env var **CHEMFILE_DATA_DIR** to override the clone location.
    * If the directory already contains a `.git` folder → `git pull`.
      Otherwise clone afresh.
    """
    data_dir = Path(os.getenv("CHEMFILE_DATA_DIR", _DEFAULT_DIR)).expanduser()
    print("[test-data] Using test data directory:", data_dir)
    if (data_dir / ".git").exists():
        # Already a repo → update
        print(f"[test-data] Updating repo in {data_dir} …")
        _run(["git", "pull", "--ff-only"], cwd=data_dir)
    else:
        # Fresh clone
        print(f"[test-data] Cloning repo into {data_dir} …")
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", _REPO_URL, str(data_dir)])

    return data_dir