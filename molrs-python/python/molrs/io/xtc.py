"""XTC, the GROMACS compressed trajectory format — ``molrs::io::xtc``.

The one-frame doors are :func:`molrs.io.read_xtc` / :func:`molrs.io.write_xtc`;
this module holds the format's classes: :class:`XtcReader` and
:class:`XtcWriter`, one frame per ``write``.
"""

from .._native import (
    XtcReader,
    XtcWriter,
)

__all__ = ["XtcReader", "XtcWriter"]
