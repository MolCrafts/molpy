"""AMBER force-field files — ``molrs::io::amber``.

:class:`AmberPrmtopForcefieldReader` reads a prmtop's force field
(``read``) or force field and structure (``read_system``);
:class:`AmberFrcmodWriter` writes a ``frcmod``. The structure doors are
functions of :mod:`molpy.io` (:func:`~molpy.io.read_prmtop`,
:func:`~molpy.io.read_inpcrd`, :func:`~molpy.io.read_ac`,
:func:`~molpy.io.read_prep` / :func:`~molpy.io.write_prep`).
"""

from molpy._core import (
    AmberFrcmodWriter,
    AmberPrmtopForcefieldReader,
)

__all__ = [
    "AmberFrcmodWriter",
    "AmberPrmtopForcefieldReader",
]
