"""Entry example: linear PEO (delegates to the topology catalogue).

Full catalogue: examples/topology/
Guide: docs/user-guide/topology/index.md
Run:   python 02_build_polymer.py
"""

from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent / "topology"


def main() -> None:
    sys.path.insert(0, str(HERE))
    runpy.run_path(str(HERE / "01_linear.py"), run_name="__main__")


if __name__ == "__main__":
    main()
