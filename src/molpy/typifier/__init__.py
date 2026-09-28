"""Force-field typification: the native typifiers.

Each name here is an identity re-export of :mod:`molrs.ff.typifier`
(``mp.typifier.ElementTypifier is molrs.ff.typifier.ElementTypifier``).
``MMFFTypifier`` is the molpy public name for
:class:`molrs.ff.typifier.MMFF94Typifier`.

The molpy typifier modules in this package (``base``, ``clp``, ``smarts``,
``ambertools``, ``scope``, ``forcefield``, ``region``, ``affected_region``,
``cache``, ``_matching``) are not imported here: they target retired molrs
APIs and await deletion or porting (bounded debt, ``.claude/notes/notes.md``).
"""

from molrs.ff.typifier import (
    ElementTypifier,
    MMFF94Typifier as MMFFTypifier,
    OPLSAATypifier,
)

__all__ = [
    "ElementTypifier",
    "MMFFTypifier",
    "OPLSAATypifier",
]
