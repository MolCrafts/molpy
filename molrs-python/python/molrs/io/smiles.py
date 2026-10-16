"""SMILES notation — ``molrs::io::smiles``.

:class:`SmilesIr` is :mod:`molrs.io`'s because SMILES is a *format*: text in,
molecule out, exactly like PDB or XYZ. :meth:`SmilesIr.parse` reads SMILES
text and ``to_atomistic()`` turns it into a molecule;
:meth:`SmilesIr.from_atomistic` builds the IR of a molecule and ``str(ir)``
writes it as SMILES. SMARTS is not a format — a pattern is a query over a
perceived graph — so it lives, wholly, in :mod:`molrs.perceive`.

:meth:`SmilesIr.from_fragment` parses a fragment body with bonding
descriptors (``[<]OCC[>]``) into a :class:`FragmentIr`, whose
``to_atomistic()`` returns the graph beside its descriptors and whose
``to_template()`` builds the ported unit. A :class:`BondingDescriptor` is one
such joining-site marker, as the CGsmiles records (:mod:`molrs.io.cgsmiles`)
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

from .._native import BondingDescriptor, FragmentIr, SmilesError, SmilesIr

__all__ = [
    "BondingDescriptor",
    "FragmentIr",
    "SmilesError",
    "SmilesIr",
]
