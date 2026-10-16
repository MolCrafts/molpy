"""GROMACS topologies (``.top`` / ``.itp``) — ``molrs::io::gromacs``.

:class:`GromacsTopForcefieldReader` reads a topology's force-field
directives (``read``) or the whole system (``read_system``);
:class:`GromacsTopForcefieldWriter` is the inverse (``write`` /
``write_system``). Structures are :func:`molrs.io.read_gro`'s.
"""

from .._native import (
    GromacsTopForcefieldReader,
    GromacsTopForcefieldWriter,
)

__all__ = ["GromacsTopForcefieldReader", "GromacsTopForcefieldWriter"]
