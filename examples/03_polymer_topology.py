"""Entry example: ruled topologies (linear, ring, star, comb, telechelic).

Full catalogue: examples/topology/
Guide: docs/user-guide/topology/index.md
Run:   python 03_polymer_topology.py
"""

from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent / "topology"


def main() -> None:
    sys.path.insert(0, str(HERE))
    for name in (
        "01_linear.py",
        "02_block.py",
        "03_ring.py",
        "04_star.py",
        "05_comb.py",
        "06_telechelic.py",
    ):
        print(f"\n--- {name} ---")
        runpy.run_path(str(HERE / name), run_name="__main__")


if __name__ == "__main__":
    main()
