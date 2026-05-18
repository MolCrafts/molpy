"""Boundary helper: molpy.Frame → molrs.Frame for molrs compute kernels.

All molrs Python compute kernels accept ``Frame | list[Frame]``. molpy
wraps ``molrs.Frame`` inside ``molpy.Frame._inner`` for storage; this
helper unwraps either form for zero-copy hand-off to a molrs kernel.
"""

from __future__ import annotations

import molrs


def to_molrs_frames(frames):
    """Return frames as ``molrs.Frame | list[molrs.Frame]`` (zero-copy).

    Accepts a single ``molpy.Frame`` / ``molrs.Frame`` or a list of either.
    """
    if isinstance(frames, (list, tuple)):
        return [_unwrap(f) for f in frames]
    return _unwrap(frames)


def _unwrap(frame) -> molrs.Frame:
    inner = getattr(frame, "_inner", None)
    if isinstance(inner, molrs.Frame):
        return inner
    if isinstance(frame, molrs.Frame):
        return frame
    raise TypeError(f"Expected molpy.Frame or molrs.Frame, got {type(frame).__name__}")
