"""Run every topology example in order (smoke suite).

Run from ``molpy/examples/``::

    python topology/run_all.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "01_linear.py",
    "02_block.py",
    "03_ring.py",
    "04_star.py",
    "05_comb.py",
    "06_telechelic.py",
    "07_gel_exhaustive.py",
    "08_gel_random.py",
    "09_end_linked.py",
    "10_dual_network.py",
    "11_prepolymer_agent.py",
]


def main() -> None:
    sys.path.insert(0, str(HERE))
    failed: list[str] = []
    for name in SCRIPTS:
        path = HERE / name
        print(f"\n======== {name} ========")
        try:
            runpy.run_path(str(path), run_name="__main__")
        except Exception as exc:  # noqa: BLE001 — smoke suite collects failures
            failed.append(f"{name}: {exc}")
            print(f"FAILED: {exc}")
    print("\n======== summary ========")
    if failed:
        print(f"{len(failed)} failed:")
        for line in failed:
            print(" ", line)
        raise SystemExit(1)
    print(f"all {len(SCRIPTS)} topology examples passed")


if __name__ == "__main__":
    main()
