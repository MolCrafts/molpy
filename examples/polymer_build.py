"""Build polymers with the declarative entry functions.

Demonstrates the two one-call entry points:

- ``polymer()`` — a single linear PEO-like chain from G-BigSMILES
- ``polymer_system()`` — a small polydisperse system

Requires RDKit for 3D embedding (``pip install rdkit``).
"""

from molpy.builder import polymer, polymer_system


def main() -> None:
    # Single chain: monomer repeat unit + chain length in one string.
    chain = polymer("{[<]CCO[>]}|5|", optimize=False, random_seed=42)
    print(f"single chain: {len(list(chain.atoms))} atoms")

    # Multi-chain system (fixed DP here; use e.g. "|schulz_zimm(1500,3000)||5e5|"
    # for a polydisperse molar-mass distribution).
    chains = polymer_system("{[<]CCO[>]}|3|", optimize=False, random_seed=42)
    print(f"system: {len(chains)} chain(s)")
    for index, built in enumerate(chains, start=1):
        print(f"  chain {index}: {len(list(built.atoms))} atoms")


if __name__ == "__main__":
    main()
