"""XTC, the GROMACS compressed trajectory format — ``molrs::io::xtc``.

The one-frame doors are :func:`molpy.io.read_xtc` / :func:`molpy.io.write_xtc`;
this module holds the format's classes: :class:`XtcReader` and
:class:`XtcWriter`, one frame per ``write``.
"""

from molpy._core import (
    XtcReader,
    XtcWriter,
)

__all__ = [
    "XtcReader",
    "XtcWriter",
]
