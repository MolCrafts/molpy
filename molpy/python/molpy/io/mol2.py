"""MOL2, the Tripos molecule format — ``molrs::io::mol2``.

The one-frame doors are :func:`molpy.io.read_mol2` / :func:`molpy.io.write_mol2`;
this module holds the format's classes: :class:`Mol2Reader`, every molecule
of a file, and :class:`Mol2Writer`, one molecule per ``write``.
"""

from molpy._core import (
    Mol2Reader,
    Mol2Writer,
)

__all__ = [
    "Mol2Reader",
    "Mol2Writer",
]
