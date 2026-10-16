"""Scientific records (``*.mrec``) — ``molrs::io::mrec``.

A record is one self-describing store: ``meta`` plus a snapshot (``frame``),
a topology (``system``), a time-ordered frame sequence (``trajectory``)
and/or a force field (``forcefield``). :class:`~molrs.core.Frame` and
:class:`~molrs.core.Trajectory` are the in-memory objects.

A whole record is written like every other format, by a function at the top
of :mod:`molrs.io`: :func:`~molrs.io.write_mrec` (a path is a directory
store, a ``bytearray`` receives the packed record). This module holds the
rest:

* :class:`MrecFile` — a record opened for its sections: ``sections()``,
  ``meta()``, ``frame()``, ``system()``, ``forcefield()``, each decoding only
  the section it names
* :class:`MrecReader` — lazy frame cursor over the ``trajectory`` section
  (``len()``, ``reader[i]``, iteration, ``frame_columns``, ``.step`` /
  ``.time`` labels, ``has_block``, ``to_trajectory()``)
* :class:`SequenceSchema` / :class:`MrecWriter` — pin a schema and write a run
  frame by frame
* :class:`ForceFieldSection` — the ``forcefield`` section as data: the
  document and one ``Block`` per style table, units as stored
  (:meth:`ForceFieldSection.from_forcefield` /
  :meth:`ForceFieldSection.to_forcefield` map it onto a force field)
* :func:`pack_mrec_zip` — collapse a closed store into one ``*.mrec.zip``
* :mod:`molrs.io.mrec.validation` — runtime check of a record's frames

The names are those of ``molrs::io::mrec``.
"""

try:
    from ..._native import mrec as _mrec
except ImportError as exc:
    raise ImportError(
        "molrs.io.mrec needs the fs feature; this build does not include it"
    ) from exc
from . import validation

ForceFieldSection = _mrec.ForceFieldSection
MrecFile = _mrec.MrecFile
MrecReader = _mrec.MrecReader
MrecWriter = _mrec.MrecWriter
SequenceSchema = _mrec.SequenceSchema
pack_mrec_zip = _mrec.pack_mrec_zip

__all__ = [
    "ForceFieldSection",
    "MrecFile",
    "MrecReader",
    "MrecWriter",
    "SequenceSchema",
    "pack_mrec_zip",
    "validation",
]
