"""PDB, the Protein Data Bank coordinate format — ``molrs::io::pdb``.

The one-frame doors are :func:`molpy.io.read_pdb` / :func:`molpy.io.write_pdb`;
this module holds the format's classes: :class:`PdbReader`, every ``MODEL``
of a file, and :class:`PdbWriter`, one ``MODEL`` per ``write``.
"""

from molpy._core import (
    PdbReader,
    PdbWriter,
)

__all__ = [
    "PdbReader",
    "PdbWriter",
]
