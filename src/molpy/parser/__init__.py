"""Parsing façade — SMILES / SMARTS from the native core; moltemplate stays local.

Chemistry notation is parsed by the native core only, and it is parsed by **types**,
not by helper functions:

* ``SmilesIR`` — ``SmilesIR("CCO")`` parses; ``.to_atomistic()`` /
  ``.components()`` / ``.n_components`` read the result.
* :class:`~molpy.core.atomistic.Atomistic` — ``mp.io.read_smiles("CCO")``
  when a molpy graph is what you want.
* ``SmartsPattern`` — ``SmartsPattern("[#6]")`` compiles a query.
* ``CGSmilesIR`` — ``CGSmilesIR(text)`` parses CGsmiles; ``.to_coarsegrain()``
  gives the bead graph, ``.to_fragment()`` the fragment, and
  ``.to_atomistic()`` the all-atom graph. ``CGGraph``, ``CGNode``, ``CGEdge``,
  ``CGFragmentDef``, ``ResolvedPair``, ``PairEnd`` and ``BondingDescriptor``
  are the parsed pieces; ``SmilesError`` is the parse error. There is no molpy
  ``CGSmilesReader``: the type is the parser.

There is deliberately nothing else here. ``parse_smiles`` / ``parse_smarts`` /
``parse_molecule`` / ``parse_mixture`` / ``smiles_to_atomistic`` /
``smilesir_to_atomistic`` were wrappers whose bodies were a constructor call —
one was literally ``return SmartsPattern(pattern)``, and two were aliases of a
third. A free function that only forwards to a constructor is a second name for
that constructor, and a second name is a thing to keep in sync.

``parse_mixture`` also *split the string on* ``'.'`` and re-parsed each piece,
which decides what a separator is before the parser has said so;
``SmilesIR.components()`` splits the parsed components instead.

:mod:`molpy.parser.moltemplate` is a separate, non-Lark ``.lt`` reader and is
not chemistry-notation parsing.

Migration:

=============================  ==============================================
was                            now
=============================  ==============================================
``parse_smiles(s)``            ``SmilesIR(s)``
``parse_smarts(p)``            ``SmartsPattern(p)``
``parse_molecule(s)``          ``mp.io.read_smiles(s)``
``smiles_to_atomistic(s)``     ``mp.io.read_smiles(s)``
``smilesir_to_atomistic(ir)``  ``ir.to_atomistic()``
``parse_mixture(s)``           ``SmilesIR(s).components()``
=============================  ==============================================
"""

from __future__ import annotations

from molrs.io import (
    BondingDescriptor,
    CGEdge,
    CGFragmentDef,
    CGGraph,
    CGNode,
    CGSmilesIR,
    PairEnd,
    ResolvedPair,
    SmilesError,
    SmilesIR,
)
from molrs.perceive import SmartsMatch, SmartsPattern

__all__ = [
    "BondingDescriptor",
    "CGEdge",
    "CGFragmentDef",
    "CGGraph",
    "CGNode",
    "CGSmilesIR",
    "PairEnd",
    "ResolvedPair",
    "SmartsMatch",
    "SmartsPattern",
    "SmilesError",
    "SmilesIR",
]
