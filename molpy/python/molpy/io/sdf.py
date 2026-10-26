"""SDF, the MDL structure-data format — ``molrs::io::sdf``.

The one-frame door is :func:`molpy.io.read_sdf`; this module holds the
format's reader, :class:`SdfReader`, every record of a file.
"""

from molpy._core import (
    SdfReader,
)

__all__ = [
    "SdfReader",
]
