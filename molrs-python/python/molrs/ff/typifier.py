"""Typifier base, its ``TypeAssignment``, and the built-in force-field typifiers.

A typifier implements one hook, ``assign(graph) -> TypeAssignment``: it reads a graph and
returns the annotations to write, positional against the graph's atoms and
against each relation kind's own rows (``graph.bonds`` … ``graph.impropers``,
``graph.terms(block)``). The
base :class:`Typifier` owns the rest. ``typify(mol)`` copies ``mol`` (as
``copy.copy(mol)``), calls
``assign`` on the copy, stamps the match onto the copy and defines its types in
the output force field — ``forcefield()``, of which ``typify`` is the only
writer. A subclass must not define ``typify``.

The built-in typifiers accept and return ``Atomistic``. :class:`ElementTypifier`
labels by element symbol alone and defines no force field.

:func:`assign_cmaps` builds a typed frame's ``cmaps`` block from its
dihedrals, against a force field's CMAP types.
"""

from .._native import (
    AtdTypifier,
    ElementTypifier,
    GaffTypifier,
    TypeAssignment,
    Mmff94sTypifier,
    Mmff94Typifier,
    OplsAaTypifier,
    Typifier,
    assign_cmaps,
)

__all__ = [
    "AtdTypifier",
    "ElementTypifier",
    "GaffTypifier",
    "Mmff94sTypifier",
    "Mmff94Typifier",
    "TypeAssignment",
    "OplsAaTypifier",
    "Typifier",
    "assign_cmaps",
]
