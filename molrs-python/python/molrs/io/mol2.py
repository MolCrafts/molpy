"""MOL2, the Tripos molecule format — ``molrs::io::mol2``.

The one-frame doors are :func:`molrs.io.read_mol2` / :func:`molrs.io.write_mol2`;
this module holds the format's classes: :class:`Mol2Reader`, every molecule
of a file, and :class:`Mol2Writer`, one molecule per ``write``.
"""

from .._native import (
    Mol2Reader,
    Mol2Writer,
)

__all__ = ["Mol2Reader", "Mol2Writer"]
