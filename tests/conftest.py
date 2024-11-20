# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-29
# version: 0.0.1

from pathlib import Path
import pytest
import subprocess

@pytest.fixture(name="test_data_path", scope="session")
def find_test_data() -> Path:

    data_path = Path(__file__).parent / "chemfile-testcases"
    if not data_path.exists():
        print("Downloading test data...")
        subprocess.run(
            f"git clone https://github.com/molcrafts/chemfile-testcases.git {data_path.parent}/chemfile-testcases",
            shell=True,
            check=True,
        )
    else:
        print("Test data already exists; updating...")
        subprocess.run(f"git pull", shell=True, cwd=data_path, check=True)

    return data_path

