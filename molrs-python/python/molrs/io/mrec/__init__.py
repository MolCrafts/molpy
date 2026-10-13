"""Scientific records (``*.mrec``) — ``molrs::io::mrec``.

A record is one self-describing store on disk: ``meta`` plus a snapshot
(``frame``), a topology (``system``), a time-ordered frame sequence
(``trajectory``) and/or a force field (``forcefield``).
:class:`~molrs.core.Frame` and :class:`~molrs.core.Trajectory` are the
in-memory objects.

Whole records are read and written like every other format, by functions at
the top of :mod:`molrs.io`: :func:`~molrs.io.read_mrec_frame` /
:func:`~molrs.io.write_mrec_frame` (Structure: ``meta`` + ``frame/``), their
``_system`` / ``_trajectory`` / ``_forcefield`` partners, and
:func:`~molrs.io.read_mrec_meta` (the identity document).

This module holds the rest:

* :class:`MrecReader` — lazy frame cursor over a store (one frame per
  :meth:`MrecReader.read_frame`; also ``len()``, ``seq[i]``, iteration,
  ``.step`` / ``.time`` labels and ``has_block``)
* :class:`SequenceSchema` / :class:`MrecWriter` — pin a schema and write a run
  frame by frame
* :class:`ForceFieldSection` — the ``forcefield`` section as data: the
  document and one ``Block`` per style table, units as stored
  (:meth:`ForceFieldSection.from_forcefield` /
  :meth:`ForceFieldSection.to_forcefield` map it onto a force field)
* :func:`section_names` — which sections a store holds
* :func:`pack_mrec_zip` — collapse a closed store into one ``*.mrec.zip``
* :mod:`molrs.io.mrec.validation` — runtime check of a record's frames

The names are those of ``molrs::io::mrec`` — ``MrecReader``, ``MrecWriter``,
``SequenceSchema``, ``section_names``.
"""

try:
    from ..._native import mrec as _mrec
except ImportError as exc:
    raise ImportError(
        "molrs.io.mrec needs the fs feature; this build does not include it"
    ) from exc
from . import validation

ForceFieldSection = _mrec.ForceFieldSection
MrecReader = _mrec.MrecReader
MrecWriter = _mrec.MrecWriter
SequenceSchema = _mrec.SequenceSchema
pack_mrec_zip = _mrec.pack_mrec_zip
section_names = _mrec.section_names

__all__ = [
    "ForceFieldSection",
    "MrecReader",
    "MrecWriter",
    "SequenceSchema",
    "pack_mrec_zip",
    "section_names",
    "validation",
]
