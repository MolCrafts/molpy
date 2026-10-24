"""GROMACS topologies (``.top`` / ``.itp``) — ``molrs::io::gromacs``.

:class:`GromacsTopForcefieldReader` reads a topology's force-field
directives (``read``) or the whole system (``read_system``);
:class:`GromacsTopForcefieldWriter` is the inverse (``write`` /
``write_system``). Structures are :func:`molpy.io.read_gro`'s.
"""

from molpy._core import (
    GromacsTopForcefieldReader,
    GromacsTopForcefieldWriter,
)

__all__ = [
    "GromacsTopForcefieldReader",
    "GromacsTopForcefieldWriter",
]
