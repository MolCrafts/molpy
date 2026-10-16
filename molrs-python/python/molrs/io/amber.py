"""AMBER force-field files — ``molrs::io::amber``.

:class:`AmberPrmtopForcefieldReader` reads a prmtop's force field
(``read``) or force field and structure (``read_system``);
:class:`AmberFrcmodWriter` writes a ``frcmod``. The structure doors are
functions of :mod:`molrs.io` (:func:`~molrs.io.read_prmtop`,
:func:`~molrs.io.read_inpcrd`, :func:`~molrs.io.read_ac`,
:func:`~molrs.io.read_prep` / :func:`~molrs.io.write_prep`).
"""

from .._native import (
    AmberFrcmodWriter,
    AmberPrmtopForcefieldReader,
)

__all__ = ["AmberFrcmodWriter", "AmberPrmtopForcefieldReader"]
