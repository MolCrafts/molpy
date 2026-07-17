"""Force-field typification: ``MolGraph -> MolGraph``.

A typifier completes the graph's truncated valences, matches it, and writes the
annotations back. Only the *match* differs between them, so
:meth:`~molpy.typifier.base.Typifier.typify` is written once and
:meth:`~molpy.typifier.base.Typifier.match` is the single abstract method.

Typifiers are named after the force field or the tool that decides the types.
:class:`~molpy.typifier.forcefield.ForceFieldParams` is **not** one: it spends a
node type rather than deciding it, and is the second half of every force-field
typifier.
"""

from molrs.typifier import MMFF94Typifier as MMFFTypifier, OPLSAATypifier

from .ambertools import AmberToolsTypifier
from .base import Match, Typifier
from .clp import ClpTypifier
from .forcefield import ForceFieldParams

__all__ = [
    # the contract
    "Typifier",
    "Match",
    # typifiers, named after their force field or tool
    "ClpTypifier",
    "AmberToolsTypifier",
    "OPLSAATypifier",
    "MMFFTypifier",
    # the component every force-field typifier ends with
    "ForceFieldParams",
]
