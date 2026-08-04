"""Linear homopolymer: build_linear / linear_topology.

Guide: docs/user-guide/topology/01_linear.md
Run:   python topology/01_linear.py
"""

from eo_kit import eo_builder, report
from molpy.builder.assembly import linear_topology


def main() -> None:
    builder = eo_builder()
    chain = builder.build_linear("EO", 10)
    report("linear-10", chain)
    same = builder.build(linear_topology(["EO"] * 10))
    print(f"  build_linear ≡ build(topology): {chain.n_atoms == same.n_atoms}")


if __name__ == "__main__":
    main()
