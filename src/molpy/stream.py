"""Live frame streaming — :mod:`molrs.stream`, mirrored by identity.

The transport only: ``Publisher`` (native builds) and ``ControlCommand``.
The wire encoding of one frame is :mod:`molpy.io`'s
(``mp.io.read_msgpack_frame_bytes`` / ``write_msgpack_frame_bytes``,
``read_json_frame_str`` / ``write_json_frame_str``).
"""

from molrs.stream import *  # noqa: F403
from molrs.stream import __all__ as __all__
