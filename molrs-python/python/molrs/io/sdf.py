"""SDF, the MDL structure-data format — ``molrs::io::sdf``.

The one-frame door is :func:`molrs.io.read_sdf`; this module holds the
format's reader, :class:`SdfReader`, every record of a file.
"""

from .._native import (
    SdfReader,
)

__all__ = ["SdfReader"]
