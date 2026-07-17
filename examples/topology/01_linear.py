"""Linear homopolymer: build_linear → build(\"{[#EO]|n}\").

Guide: docs/user-guide/topology/01_linear.md
Run:   python topology/01_linear.py
"""

from eo_kit import eo_builder, report


def main() -> None:
    builder = eo_builder()
    chain = builder.build_linear("EO", 10)
    report("linear-10", chain)
    # Sole entry is still build():
    same = builder.build("{[#EO]|10}")
    print(f"  build_linear ≡ build: {chain.n_atoms == same.n_atoms}")


if __name__ == "__main__":
    main()
