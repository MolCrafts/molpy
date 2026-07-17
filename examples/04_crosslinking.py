"""Entry example: statistical networks (gel, end-link, dual, agent).

Full catalogue: examples/topology/
Guide: docs/user-guide/topology/index.md
Run:   python 04_crosslinking.py
"""

from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent / "topology"


def main() -> None:
    sys.path.insert(0, str(HERE))
    for name in (
        "07_gel_exhaustive.py",
        "08_gel_random.py",
        "09_end_linked.py",
        "10_dual_network.py",
        "11_prepolymer_agent.py",
    ):
        print(f"\n--- {name} ---")
        runpy.run_path(str(HERE / name), run_name="__main__")


if __name__ == "__main__":
    main()
