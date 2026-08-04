"""Parsing façade — SMILES / SMARTS from molrs; moltemplate stays local.

Chemistry notation is parsed by ``molrs`` only, and it is parsed by **types**,
not by helper functions:

* :class:`molrs.io.SmilesIR` — ``SmilesIR("CCO")`` parses; ``.to_atomistic()`` /
  ``.components()`` / ``.n_components`` read the result.
* :class:`~molpy.core.atomistic.Atomistic` — ``mp.io.read_smiles("CCO")``
  when a molpy graph is what you want.
* :class:`molrs.perceive.SmartsPattern` — ``SmartsPattern("[#6]")`` compiles a query.

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
``smilesir_to_atomistic(ir)``  ``Atomistic.adopt(ir.to_atomistic())``
``parse_mixture(s)``           ``[Atomistic.adopt(m) for m in``
                               ``SmilesIR(s).components()]``
=============================  ==============================================
"""

from __future__ import annotations

from molrs.io import SmilesIR
from molrs.perceive import SmartsMatch, SmartsPattern

__all__ = [
    "SmartsMatch",
    "SmartsPattern",
    "SmilesIR",
]
