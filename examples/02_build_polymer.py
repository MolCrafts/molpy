"""Build a polymer chain by composing the real builder classes.

MolPy has no ``polymer()`` dispatcher — you assemble a chain yourself, which
keeps the data flow explicit: parse a repeat-unit monomer, embed it in 3D, then
hand a monomer library + a reaction to ``PolymerBuilder`` and feed it a label
sequence. Each piece (parser, adapter, builder) is a real class you call directly.

Guide: docs/user-guide/02_polymer_stepwise.md
Run:   python 02_build_polymer.py
"""

from collections import Counter

from molpy.builder.polymer import PolymerBuilder, ReactionPresets
from molpy.conformer import Conformer
from molpy.parser import parse_monomer


def main() -> None:
    # 1. Repeat unit: BigSMILES {[<]CCO[>]} is one ethylene-oxide unit with a
    #    head (<) and tail (>) bonding port. Conformer (molpy's native molrs
    #    embedder) adds hydrogens and 3D coordinates the reaction + placement need.
    eo, _ = Conformer(add_hydrogens=True, seed=42).generate(
        parse_monomer("{[<]CCO[>]}")
    )

    # 2. Assemble PEO-10. The "dehydration" preset joins the tail O of one unit
    #    to the head C of the next, dropping one H from each. build_sequence takes
    #    the label list directly — no CGSmiles string, no notation round-trip.
    builder = PolymerBuilder({"EO": eo}, reacter=ReactionPresets.get("dehydration"))
    chain = builder.build_sequence(["EO"] * 10).polymer

    atoms = list(chain.atoms)
    counts = Counter(a.get("element") for a in atoms)
    print(f"PEO-10 chain: {len(atoms)} atoms, {len(list(chain.bonds))} bonds")
    print(f"  composition: {dict(counts)}")  # {'C': 20, 'O': 10, 'H': 42}

    # The result is a plain Atomistic — the same structure parse_molecule returns —
    # so every downstream step (crosslink, typify, optimize, export) accepts it.
    print(f"  type: {type(chain).__name__}")


if __name__ == "__main__":
    main()
