"""CIF, the Crystallographic Information File — ``molrs::io::cif``.

The one-frame doors are :func:`molrs.io.read_cif` / :func:`molrs.io.write_cif`;
this module holds the format's classes: :class:`CifReader`, every ``data_``
block of a file, and :class:`CifWriter`, one block per ``write``.
"""

from .._native import (
    CifReader,
    CifWriter,
)

__all__ = ["CifReader", "CifWriter"]
