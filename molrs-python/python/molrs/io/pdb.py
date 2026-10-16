"""PDB, the Protein Data Bank coordinate format — ``molrs::io::pdb``.

The one-frame doors are :func:`molrs.io.read_pdb` / :func:`molrs.io.write_pdb`;
this module holds the format's classes: :class:`PdbReader`, every ``MODEL``
of a file, and :class:`PdbWriter`, one ``MODEL`` per ``write``.
"""

from .._native import (
    PdbReader,
    PdbWriter,
)

__all__ = ["PdbReader", "PdbWriter"]
