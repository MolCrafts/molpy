"""SMILES notation — ``molrs::io::smiles``.

SMILES is :mod:`molpy.io`'s because it is a *format*: text in, molecule out,
exactly like PDB or XYZ. :func:`read_smiles` reads SMILES text into a
:class:`SmilesIr`, whose ``to_atomistic()`` turns it into a molecule;
:func:`write_smiles` writes a molecule as SMILES text. An IR is handed out by
a ``read_*`` function and never constructed by the caller. SMARTS is not a format — a pattern is a query over a
perceived graph — so it lives, wholly, in :mod:`molpy.perceive`.

:func:`read_smiles_fragment` parses a fragment body with bonding
descriptors (``[<]OCC[>]``) into a :class:`FragmentIr`, whose
``to_atomistic()`` returns the graph beside its descriptors and whose
``to_template()`` builds the ported unit. A :class:`BondingDescriptor` is one
such joining-site marker, as the CGsmiles records (:mod:`molpy.io.cgsmiles`)
hand it out.

Every refusal raised by this family of notations — by the parser, by the
expansion of a CGsmiles string, or by an emit — is a :class:`SmilesError`, one
class carrying the four facts the Rust error owns: ``kind``, the variant name
of the rule that was broken (``"UnclosedBranch"``, ``"UnexpectedEnd"``,
``"CgNotExpandable"``, …); ``span``, the byte range of the offending text as a
``(start, end)`` pair whose end is clamped to ``len(input)``; ``input``, the
offending string, empty for the errors raised past the parser, which are handed
an IR and never see the text it came from; and ``notation``, lowercase
``"smiles"``, ``"smarts"`` or ``"cgsmiles"``. It subclasses
:class:`ValueError`, so ``except ValueError`` keeps catching it, and
``str(e)`` is the message Rust renders, caret line included.
"""

from molpy._core import (
    BondingDescriptor,
    FragmentIr,
    SmilesError,
    SmilesIr,
    read_smiles,
    read_smiles_fragment,
    write_smiles,
)

__all__ = [
    "BondingDescriptor",
    "FragmentIr",
    "SmilesError",
    "SmilesIr",
    "read_smiles",
    "read_smiles_fragment",
    "write_smiles",
]
