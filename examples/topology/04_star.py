"""T6 — Star: multifunctional core + arms via build_star.

Guide: docs/user-guide/topology/04_star.md
Run:   python topology/04_star.py
"""

from eo_kit import eo_builder, report, trifunctional_core


def main() -> None:
    builder = eo_builder(extra={"X3": trifunctional_core()})
    star = builder.build_star("X3", "EO", n_arms=3, arm_length=4)
    report("star-3x4", star)
    print("  core=X3, arms=3 × EO4  (shortcut → branched CGSmiles → build)")


if __name__ == "__main__":
    main()
