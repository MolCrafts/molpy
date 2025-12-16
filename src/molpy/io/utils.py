"""IO utility functions and decorators.

Provides type conversion helpers and reader utilities for IO operations.
"""

from functools import wraps

from molpy.core.frame import Frame
from molpy.core.segment import Segment
from molpy.core.struct import Struct
from molpy.core.system import System

FrameLike = Frame | System | Struct
"""Type alias for frame-like objects."""


def to_system(func):
    """
    Decorator to convert first argument to System.

    Automatically converts Frame, Struct, or Segment to System
    before calling the decorated function.

    Args:
        func: Function expecting System as first argument

    Returns:
        Wrapped function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            system = args[0]
            args = args[1:]
        else:
            system = kwargs.pop("system", None)

        if isinstance(system, Frame):
            system = System(frame=system)
        elif isinstance(system, (Struct, Segment)):
            system = System(frame=system.to_frame())

        assert isinstance(
            system, System
        ), f"Expected system to be a molpy System object, got {type(system)}"
        return func(system, *args, **kwargs)

    return wrapper


def to_frame(framelike: FrameLike) -> Frame:
    """
    Convert frame-like object to Frame.

    Args:
        framelike: System, Frame, or Struct

    Returns:
        Frame instance
    """
    if isinstance(framelike, System):
        frame = framelike.frame
    elif isinstance(framelike, Frame):
        frame = framelike
    elif isinstance(framelike, Struct):
        frame = framelike.to_frame()

    return frame


class ZipReader:
    """
    Zip multiple readers together for parallel iteration.

    Context manager that yields tuples of frames from multiple readers.

    Args:
        *readers: Variable number of reader objects
    """

    def __init__(self, *readers):
        self.readers = readers

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close all readers."""
        for reader in self.readers:
            reader.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        """Iterate over zipped frames from all readers."""
        yield from zip(*self.readers)
